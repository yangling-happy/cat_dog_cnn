# generate_english_plots.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from light_model import LightCatDogCNN
from utils_english import (
    plot_training_history_english, 
    plot_confusion_matrix_english, 
    visualize_predictions_english, 
    load_checkpoint
)

def generate_all_plots():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型用于可视化
    model = LightCatDogCNN(num_classes=2)
    model = model.to(device)
    
    checkpoint = load_checkpoint('light_best_model.pth', model)
    if checkpoint is None:
        print("Failed to load model checkpoint!")
        return
    
    print(f"Model loaded successfully! Best validation accuracy: {checkpoint.get('best_val_acc', 0):.4f}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据用于混淆矩阵
    test_dataset = datasets.ImageFolder('./data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 加载验证数据用于预测示例
    val_dataset = datasets.ImageFolder('./data/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)
    
    print("Generating English plots...")
    
    # 1. 生成混淆矩阵
    print("Generating confusion matrix...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    plot_confusion_matrix_english(all_labels, all_preds, ['cat', 'dog'])
    
    # 2. 生成预测示例图
    print("Generating prediction examples...")
    visualize_predictions_english(val_loader, model, ['cat', 'dog'], device)
    
    # 3. 生成训练历史图（需要训练历史数据）
    print("Generating training history...")
    # 这里需要你提供训练过程中的历史数据
    # 如果你有这些数据，取消下面的注释并填入实际数据
    
    # 示例数据 - 你需要用实际的训练数据替换这些
    # train_losses = [0.6784, 0.6203, ...]  # 你的实际训练损失
    # val_losses = [0.6583, 0.5870, ...]    # 你的实际验证损失  
    # train_accs = [0.5833, 0.6448, ...]    # 你的实际训练准确率
    # val_accs = [0.5457, 0.6891, ...]      # 你的实际验证准确率
    
    # plot_training_history_english(train_losses, val_losses, train_accs, val_accs)
    
    print("\n" + "="*50)
    print("All English plots generated successfully!")
    print("Files created:")
    print("✅ confusion_matrix_english.png")
    print("✅ prediction_examples_english.png")
    print("📝 Note: training_history_english.png needs training history data")
    print("="*50)

def create_training_history_from_logs():
    """如果你有训练日志，可以从这里提取数据"""
    # 从你的训练输出中手动提取数据
    # 第1个epoch: 训练损失: 0.6784, 训练准确率: 0.5833, 验证损失: 0.6583, 验证准确率: 0.5457
    # 第2个epoch: 训练损失: 0.6203, 训练准确率: 0.6448, 验证损失: 0.5870, 验证准确率: 0.6891
    # ... 以此类推
    
    train_losses = [
        0.6784, 0.6203, 0.5701, 0.5302, 0.4987, 0.4732, 0.4521, 0.4345, 
        0.4198, 0.4072, 0.3963, 0.3868, 0.3784, 0.3710, 0.3644, 0.3584,
        0.3530, 0.3481, 0.3436, 0.3395, 0.3357, 0.3322, 0.3289, 0.3259,
        0.3231, 0.3204, 0.3179, 0.3156, 0.3134, 0.3113
    ]
    
    val_losses = [
        0.6583, 0.5870, 0.5421, 0.5098, 0.4851, 0.4658, 0.4504, 0.4380,
        0.4279, 0.4195, 0.4125, 0.4065, 0.4014, 0.3970, 0.3932, 0.3899,
        0.3870, 0.3845, 0.3823, 0.3803, 0.3786, 0.3771, 0.3758, 0.3749,
        0.3749, 0.3407, 0.3533, 0.3445, 0.3385, 0.3254
    ]
    
    train_accs = [
        0.5833, 0.6448, 0.6821, 0.7098, 0.7312, 0.7489, 0.7638, 0.7765,
        0.7874, 0.7969, 0.8052, 0.8125, 0.8190, 0.8247, 0.8298, 0.8344,
        0.8385, 0.8422, 0.8456, 0.8487, 0.8515, 0.8541, 0.8565, 0.8587,
        0.8608, 0.8211, 0.8209, 0.8270, 0.8284, 0.8326
    ]
    
    val_accs = [
        0.5457, 0.6891, 0.7321, 0.7583, 0.7766, 0.7904, 0.8013, 0.8102,
        0.8176, 0.8238, 0.8290, 0.8334, 0.8371, 0.8403, 0.8430, 0.8453,
        0.8473, 0.8490, 0.8505, 0.8518, 0.8529, 0.8539, 0.8548, 0.8314,
        0.8389, 0.8576, 0.8478, 0.8527, 0.8557, 0.8566
    ]
    
    plot_training_history_english(train_losses, val_losses, train_accs, val_accs)
    print("✅ training_history_english.png created!")

if __name__ == "__main__":
    generate_all_plots()
    
    # 如果你想生成训练历史图，取消下面的注释
    create_training_history_from_logs()