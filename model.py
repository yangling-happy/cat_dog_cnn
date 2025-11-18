# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CatDogCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(CatDogCNN, self).__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate/2)
        )
        
        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate/2)
        )
        
        # 第三个卷积块
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate/2)
        )
        
        # 第四个卷积块
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate/2)
        )
        
        # 添加自适应池化层，固定输出尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 修正全连接层输入尺寸
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 7 * 7, 512),  # 修正后的尺寸：256 * 7 * 7 = 12544
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(128, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)  # [batch, 32, 112, 112]
        x = self.conv2(x)  # [batch, 64, 56, 56]
        x = self.conv3(x)  # [batch, 128, 28, 28]
        x = self.conv4(x)  # [batch, 256, 14, 14]
        
        x = self.adaptive_pool(x)  # [batch, 256, 7, 7]
        x = x.view(x.size(0), -1)  # 展平 [batch, 256*7*7=12544]
        x = self.classifier(x)
        return x

def test_model():
    """测试模型结构"""
    model = CatDogCNN()
    x = torch.randn(2, 3, 224, 224)
    print(f"输入尺寸: {x.shape}")
    
    # 逐步测试
    x1 = model.conv1(x)
    print(f"conv1后: {x1.shape}")
    
    x2 = model.conv2(x1)
    print(f"conv2后: {x2.shape}")
    
    x3 = model.conv3(x2)
    print(f"conv3后: {x3.shape}")
    
    x4 = model.conv4(x3)
    print(f"conv4后: {x4.shape}")
    
    x_pool = model.adaptive_pool(x4)
    print(f"自适应池化后: {x_pool.shape}")
    
    x_flat = x_pool.view(x_pool.size(0), -1)
    print(f"展平后: {x_flat.shape}")
    
    output = model.classifier(x_flat)
    print(f"输出尺寸: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    test_model()