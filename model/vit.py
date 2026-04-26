import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2 # h = w
        
        self.proj = nn.Conv2d(in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        B, C, H, W = x.shape # batch size, channels, height, width
        assert H == self.img_size and W == self.img_size, "Input image size must match the defined img_size"
        
        x = self.proj(x)  # (B, embed_dim, num_patches**0.5, num_patches**0.5)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return x
    

class ViTMLP(nn.Module):
    def __init__(self, embed_dim=768, mlp_dim=3072, dropout=0.1):
        super(ViTMLP, self).__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
    
    def forward(self, x):
        x = self.fc1(x)  # (B, num_patches, mlp_dim)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, num_patches, embed_dim)
        return x
    

class ViTBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, mlp_dim=3072, dropout=0.1):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = ViTMLP(embed_dim=embed_dim, mlp_dim=mlp_dim, dropout=dropout)
    
    def forward(self, x):
        # Multi-head Self-Attention
        x_norm = self.norm1(x)  # (B, num_patches, embed_dim)
        attn_output, _ = self.attn(x_norm.transpose(0, 1), x_norm.transpose(0, 1), x_norm.transpose(0, 1))  # (num_patches, B, embed_dim)
        attn_output = attn_output.transpose(0, 1)  # (B, num_patches, embed_dim)
        
        # Residual connection
        x = x + attn_output
        
        # MLP
        x_norm = self.norm2(x)  # (B, num_patches, embed_dim)
        mlp_output = self.mlp(x_norm)  # (B, num_patches, embed_dim)
        
        # Residual connection
        x = x + mlp_output
        
        return x
    

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=8, mlp_dim=3072, num_blocks=12, num_classes=1000, dropout=0.1):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)  # (B, num_patches, embed_dim)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        
        x = x + self.pos_embedding[:, :x.size(1), :]  # Add positional embedding
        x = self.dropout(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)  # (B, num_patches + 1, embed_dim)
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_output)  # (B, num_classes)
        
        return logits