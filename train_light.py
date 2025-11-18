# train_light.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import time
import os
from light_model import LightCatDogCNN
from utils import plot_training_history, save_checkpoint

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数配置
config = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 30,
    'patience': 7,
    'weight_decay': 1e-4
}

def train_light_model():
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据
    train_dataset = datasets.ImageFolder('./data/train', transform=train_transform)
    val_dataset = datasets.ImageFolder('./data/val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"类别映射: {train_dataset.class_to_idx}")
    
    # 使用轻量模型
    model = LightCatDogCNN(num_classes=2, dropout_rate=0.5)
    model = model.to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'], 
                          weight_decay=config['weight_decay'])
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 训练历史记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    patience_counter = 0
    initial_lr = config['learning_rate']
    
    print("开始训练轻量模型...")
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.cpu())
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_dataset)
        
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.cpu())
        
        # 更新学习率
        scheduler.step(val_epoch_loss)
        
        # 手动打印学习率变化
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != initial_lr:
            print(f"  学习率更新为: {current_lr:.6f}")
            initial_lr = current_lr
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1}/{config["epochs"]} ({epoch_time:.1f}s):')
        print(f'  训练损失: {epoch_loss:.4f}, 训练准确率: {epoch_acc:.4f}')
        print(f'  验证损失: {val_epoch_loss:.4f}, 验证准确率: {val_epoch_acc:.4f}')
        
        # 保存最佳模型
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            patience_counter = 0
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config,
                'class_to_idx': train_dataset.class_to_idx
            }, 'light_best_model.pth')
            print(f'  ✅ 保存最佳模型，验证准确率: {best_val_acc:.4f}')
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= config['patience']:
            print(f'🛑 早停！在 epoch {epoch+1} 停止训练')
            break
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    print(f"训练完成！最佳验证准确率: {best_val_acc:.4f}")
    
    return model

if __name__ == "__main__":
    model = train_light_model()