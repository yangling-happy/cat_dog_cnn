import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import CatDogCNN
from utils import load_checkpoint
import argparse

def predict_single_image(image_path, model_path='models/best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = CatDogCNN(num_classes=2)
    checkpoint = load_checkpoint(model_path, model)
    
    if checkpoint is None:
        return
    
    model = model.to(device)
    model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载和预处理图像
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # 类别映射
    class_names = ['cat', 'dog']  # 根据你的数据顺序调整
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item()
    
    # 显示结果
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f'预测: {predicted_class} (置信度: {confidence_score:.4f})')
    plt.axis('off')
    plt.show()
    
    print(f"预测结果: {predicted_class}")
    print(f"置信度: {confidence_score:.4f}")
    print(f"各类别概率:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {probabilities[0][i].item():.4f}")
    
    return predicted_class, confidence_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='预测单张猫狗图片')
    parser.add_argument('image_path', type=str, help='输入图片路径')
    parser.add_argument('--model', type=str, default='models/best_model.pth', help='模型路径')
    
    args = parser.parse_args()
    
    predict_single_image(args.image_path, args.model)