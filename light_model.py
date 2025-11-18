# light_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightCatDogCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(LightCatDogCNN, self).__init__()
        
        # 更轻量级的结构
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate/3),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate/3),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate/3),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate/3),
        )
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, num_classes)
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
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def test_light_model():
    """测试轻量模型"""
    model = LightCatDogCNN()
    print("轻量模型结构:")
    print(model)
    
    x = torch.randn(4, 3, 224, 224)
    print(f"\n输入: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
        print(f"输出: {output.shape}")
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 检查输出多样性
        probs = torch.softmax(output, dim=1)
        print(f"\n样本预测概率:")
        for i, prob in enumerate(probs):
            print(f"  样本{i}: cat={prob[0]:.3f}, dog={prob[1]:.3f}")

if __name__ == "__main__":
    test_light_model()