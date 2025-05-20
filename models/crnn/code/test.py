import os
import json
import cv2
import torch
import numpy as np
from .model import SimpleCharClassifier
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

def check_image_quality(image):
    """
    检查图像质量
    返回: (bool, str) - (是否通过检查, 失败原因)
    """
    if image is None:
        return False, "图像读取失败"
    
    h, w = image.shape[:2]
    if h < 16 or w < 16:
        return False, f"图像尺寸过小: {h}x{w}"
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian < 30:
        return False, f"图像模糊: {laplacian:.2f}"
    
    brightness = np.mean(gray)
    if brightness < 20:
        return False, f"图像过暗: {brightness:.2f}"
    if brightness > 230:
        return False, f"图像过亮: {brightness:.2f}"
    
    contrast = np.std(gray)
    if contrast < 20:
        return False, f"图像对比度过低: {contrast:.2f}"
    
    return True, "图像质量正常"

def preprocess_image(image):
    """
    预处理图像
    返回: 预处理后的图像
    """
    if image is None:
        return None
    
    # 调整图像大小
    h, w = image.shape[:2]
    if h < 16 or w < 16:
        scale = max(16/h, 16/w)
        image = cv2.resize(image, (int(w*scale), int(h*scale)))
    
    # 调整亮度和对比度
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    if brightness < 20:
        alpha = 1.5
        beta = 30
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    elif brightness > 230:
        alpha = 0.8
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    
    if contrast < 20:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    return image

def load_model(model_path='../user_data/model_data/best_model.pth'):
    """
    加载训练好的模型
    Args:
        model_path: 模型文件路径
    Returns:
        加载好的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCharClassifier(num_classes=10)  # 0-9数字分类
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print(f"成功加载模型，验证损失: {checkpoint['val_loss']:.4f}, 验证准确率: {checkpoint['val_acc']:.4f}")
        return model
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None

from .data_loader import get_transforms

def predict_image(model, image_path):
    """
    预测单张图像（适配CRNN输入格式）
    Args:
        model: 加载好的模型
        image_path: 图像文件路径
    Returns:
        dict: 包含预测结果和置信度
    """
    if model is None:
        return {'text': '', 'confidence': 0.0, 'success': False}

    try:
        device = next(model.parameters()).device
    except StopIteration:
        return {'text': '', 'confidence': 0.0, 'success': False}

    # 读取图像为三通道
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 强制三通道
    if image is None:
        print(f"无法读取图像: {image_path}")
        return {'text': '', 'confidence': 0.0, 'success': False}

    # 转换为 RGB 格式（如果模型训练时用了 RGB）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 应用与训练时一致的预处理（包括归一化）
    transform = get_transforms(is_train=False)  # 使用与训练相同的 transforms
    transformed = transform(image=image)
    image_tensor = transformed['image']  # shape: [C, H, W]
    image_tensor = image_tensor.unsqueeze(0).to(device)  # shape: [1, C, H, W]

    # 推理
    with torch.no_grad():
        logits, confidence, predicted = model(image_tensor)
        pred_class = str(predicted.item())
        conf = confidence.item()
        print(f"Logits: {logits}")
        print(f"Confidence: {confidence}")
        return {
            'text': pred_class,
            'confidence': conf,
            'success': True
        }
    
def predict_batch(model, image_dir, output_dir):
    """
    批量预测图像
    Args:
        model: 加载好的模型
        image_dir: 图像目录
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    for img_name in tqdm(image_files, desc="处理图像"):
        img_path = os.path.join(image_dir, img_name)
        result = predict_image(model, img_path)
        
        if result and result['success']:
            results.append({
                'image': img_name,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
    
    # 保存预测结果
    output_file = os.path.join(output_dir, 'predictions.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"预测完成，结果已保存到: {output_file}")

def main():
    # 配置
    model_path = '../user_data/model_data/best_model.pth'
    test_data_dir = '../user_data/test_data'
    output_dir = '../user_data/predictions'
    
    # 加载模型
    model = load_model(model_path)
    if model is None:
        print("模型加载失败，程序退出")
        return
    
    # 执行预测
    predict_batch(model, test_data_dir, output_dir)

if __name__ == '__main__':
    main()