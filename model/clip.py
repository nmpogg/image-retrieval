import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import math

class ImageEncoderResNet(nn.Module):
    def __init__(self, embed_dim=512, resnet_version='resnet50'):
        super().__init__()
        # Khởi tạo ResNet cơ bản (không dùng pre-trained để code từ đầu)
        if resnet_version == 'resnet50':
            self.backbone = models.resnet50(weights=None)
        else:
            self.backbone = models.resnet18(weights=None)
            
        # Kích thước feature map trước lớp FC của ResNet50 là 2048 (ResNet18 là 512)
        feature_dim = self.backbone.fc.in_features
        
        # Loại bỏ lớp FC cũ và thay bằng lớp Identity để lấy feature thô
        self.backbone.fc = nn.Identity()
        
        # Lớp Projection đưa feature về không gian chia sẻ (shared embedding space)
        self.projection = nn.Linear(feature_dim, embed_dim, bias=False)

    def forward(self, x):
        # x shape: [batch_size, 3, 224, 224]
        features = self.backbone(x)
        # Chiếu xuống embed_dim
        embeddings = self.projection(features)
        return embeddings
    
class ImageEncoderViT(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # Khởi tạo ViT cơ bản (ViT-B/32)
        self.backbone = models.vit_b_32(weights=None)
        
        # Kích thước hidden size của ViT-B là 768
        feature_dim = self.backbone.heads.head.in_features
        
        # Loại bỏ classification head
        self.backbone.heads = nn.Identity()
        
        # Thêm Projection layer
        self.projection = nn.Linear(feature_dim, embed_dim, bias=False)

    def forward(self, x):
        # Lấy feature từ token [class]
        features = self.backbone(x)
        embeddings = self.projection(features)
        return embeddings
    
class TextEncoder(nn.Module):
    def __init__(self, vocab_size=49152, embed_dim=512, context_length=77, 
                 transformer_width=512, transformer_layers=8, transformer_heads=8):
        super().__init__()
        self.context_length = context_length
        
        # Token embedding & Positional embedding
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        # Khối Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_width, 
            nhead=transformer_heads, 
            dim_feedforward=transformer_width * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Lớp Projection
        self.projection = nn.Linear(transformer_width, embed_dim, bias=False)

    def forward(self, text):
        # text shape: [batch_size, context_length]
        
        x = self.token_embedding(text) # [batch_size, context_length, width]
        x = x + self.positional_embedding
        
        # Đưa qua Transformer
        x = self.transformer(x)
        
        # CLIP gốc lấy feature tại vị trí của token [EOS]. 
        # Giả sử text được pad bằng 0 và token cuối cùng là argmax của text.
        # Để đơn giản trong code từ đầu, ta có thể lấy token có vị trí cao nhất trước vùng padding,
        # hoặc dùng công thức gather sau:
        batch_size = x.shape[0]
        
        # Tìm vị trí thực tế của [EOS] (giả sử [EOS] là token lớn nhất trong index của tokenizer)
        # Trong thực tế, bạn sẽ truyền eos_indices vào, ở đây giả lập bằng cách lấy vị trí cuối cùng.
        eos_indices = text.argmax(dim=-1) 
        
        # Trích xuất feature tại [EOS]
        x = x[torch.arange(batch_size), eos_indices]
        
        # Chiếu xuống embed_dim
        embeddings = self.projection(x)
        return embeddings
    
class CLIP(nn.Module):
    def __init__(self, embed_dim=512, image_encoder_type='vit', vocab_size=49152):
        super().__init__()
        
        # 1. Khởi tạo Image Encoder
        if image_encoder_type == 'resnet':
            self.image_encoder = ImageEncoderResNet(embed_dim=embed_dim)
        elif image_encoder_type == 'vit':
            self.image_encoder = ImageEncoderViT(embed_dim=embed_dim)
        else:
            raise ValueError("Chỉ hỗ trợ 'resnet' hoặc 'vit'")
            
        # 2. Khởi tạo Text Encoder
        self.text_encoder = TextEncoder(vocab_size=vocab_size, embed_dim=embed_dim)
        
        # 3. Tham số Logit Scale (Temperature)
        # Khởi tạo giá trị bằng ln(1/0.07) theo bài báo gốc
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, image, text):
        # Trích xuất đặc trưng
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        
        # L2 Normalization (Chuẩn hóa vector)
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # Tính toán ma trận tương đồng (Cosine Similarity)
        # Tránh logit_scale quá lớn bằng cách clamp
        logit_scale = self.logit_scale.exp() 
        logit_scale = torch.clamp(logit_scale, max=100.0)
        
        # Image -> Text logits
        logits_per_image = logit_scale * image_features @ text_features.t()
        # Text -> Image logits
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

def contrastive_loss(logits_per_image, logits_per_text):
    """
    Symmetric Cross-Entropy Loss (InfoNCE)
    """
    batch_size = logits_per_image.shape[0]
    device = logits_per_image.device
    
    # Ground truth (Nhãn) chính là đường chéo của ma trận (0, 1, 2, ..., N-1)
    labels = torch.arange(batch_size, dtype=torch.long, device=device)
    
    # Tính Loss cho cả 2 chiều
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    # Trung bình cộng loss
    return (loss_i + loss_t) / 2