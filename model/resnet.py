import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, num_channels, use_1x1_conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)

        if use_1x1_conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class Bottleneck(nn.Module):
    expansion = 4 # Output channels sẽ gấp 4 lần input channels nội bộ

    def __init__(self, num_channels, use_1x1_conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=1)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv3 = nn.LazyConv2d(num_channels * self.expansion, kernel_size=1)

        if use_1x1_conv:
            self.conv4 = nn.LazyConv2d(num_channels * self.expansion, kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        
        if self.conv4:
            X = self.conv4(X)
            
        Y += X
        return F.relu(Y)
    
class ResNet(nn.Module):

    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    
    def block(self, block_class, num_residuals, num_channels, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(block_class(num_channels=num_channels, use_1x1_conv=True, strides=2))
            # Với Bottleneck, đôi khi layer đầu tiên dù stride=1 vẫn cần 1x1 conv để đồng bộ số kênh (do expansion=4)
            elif i == 0 and first_block and block_class.expansion == 4:
                blk.append(block_class(num_channels=num_channels, use_1x1_conv=True, strides=1))
            else:
                blk.append(block_class(num_channels=num_channels))
        return nn.Sequential(*blk)
    
    def last(self, num_class):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(out_features=num_class)
        )
    
    def __init__(self, block_class, blocks_info, num_class=10):
        super().__init__()
        blocks = [self.b1()]

        for i, block_info in enumerate(blocks_info):
            # unpack block_info: (num_residuals, num_channels)
            blocks.append(self.block(block_class, *block_info, first_block=(i==0)))
        
        blocks.append(self.last(num_class=num_class))
        self.net = nn.Sequential(*blocks)

    def forward(self, X):
        return self.net(X)
    

class ResNet18(ResNet):
    """
    Phiên bản ResNet-18 sử dụng BasicBlock.
    Cấu hình block: [2, 2, 2, 2]
    """
    def __init__(self, num_classes=10):
        # Truyền BasicBlock và tuple cấu hình vào class cha (ResNet)
        super().__init__(BasicBlock, ((2, 64), (2, 128), (2, 256), (2, 512)), num_classes)

class ResNet50(ResNet):
    """
    Phiên bản ResNet-50 sử dụng Bottleneck.
    Cấu hình block: [3, 4, 6, 3]
    """
    def __init__(self, num_classes=10):
        # Truyền Bottleneck và tuple cấu hình vào class cha (ResNet)
        super().__init__(Bottleneck, ((3, 64), (4, 128), (6, 256), (3, 512)), num_classes)