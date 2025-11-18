# evaluate_light.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix
from light_model import LightCatDogCNN
from utils import load_checkpoint, plot_confusion_matrix

def evaluate_light_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据预处理
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据
    test_dataset = datasets.ImageFolder('./data/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"测试样本数: {len(test_dataset)}")
    print(f"类别: {test_dataset.classes}")
    
    # 加载轻量模型
    model = LightCatDogCNN(num_classes=2)
    model = model.to(device)
    
    # 修正：直接使用文件名，因为 load_checkpoint 已经在 models 目录中查找
    checkpoint = load_checkpoint('light_best_model.pth', model)
    if checkpoint is None:
        return
    
    print(f"模型加载成功！训练轮次: {checkpoint.get('epoch', 'N/A')}")
    print(f"验证集最佳准确率: {checkpoint.get('best_val_acc', 0):.4f}")
    
    model.eval()
    
    all_preds = []
    all_labels = []
    test_running_corrects = 0
    
    print("测试中...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            test_running_corrects += torch.sum(preds == labels.data)
    
    test_acc = test_running_corrects.double() / len(test_dataset)
    
    print("\n" + "=" * 50)
    print("轻量模型测试结果")
    print("=" * 50)
    print(f"测试准确率: {test_acc:.4f} ({test_running_corrects}/{len(test_dataset)})")
    
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, 
                              target_names=test_dataset.classes,
                              digits=4))
    
    # 绘制混淆矩阵
    plot_confusion_matrix(all_labels, all_preds, test_dataset.classes)
    
    return test_acc

if __name__ == "__main__":
    evaluate_light_model()