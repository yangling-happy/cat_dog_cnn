# utils_english.py
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def plot_training_history_english(train_losses, val_losses, train_accs, val_accs):
    """绘制训练历史图表（英文版）"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用英文字体
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot([acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in train_accs], 
             'b-', label='Training Accuracy', linewidth=2)
    ax2.plot([acc.cpu().numpy() if torch.is_tensor(acc) else acc for acc in val_accs], 
             'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_english.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix_english(y_true, y_pred, classes, normalize=True):
    """绘制混淆矩阵（英文版）"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Percentage'})
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_english.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def visualize_predictions_english(dataloader, model, class_names, device, num_images=10):
    """可视化模型预测结果（英文版）"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    model.eval()
    images, labels = next(iter(dataloader))
    
    # Select first num_images
    images = images[:num_images]
    labels = labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
    
    # Plot results
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i in range(num_images):
        # Denormalize image
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_examples_english.png', dpi=300, bbox_inches='tight')
    plt.show()

# 其他工具函数保持不变
def save_checkpoint(state, filename='best_model.pth'):
    """Save model checkpoint"""
    os.makedirs('models', exist_ok=True)
    filepath = os.path.join('models', filename)
    torch.save(state, filepath)
    print(f"Model saved as {filepath}")

def load_checkpoint(filename, model, optimizer=None):
    """Load model checkpoint"""
    filepath = os.path.join('models', filename)
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint '{filepath}' (epoch {checkpoint['epoch']})")
        return checkpoint
    else:
        print(f"Checkpoint '{filepath}' not found")
        return None