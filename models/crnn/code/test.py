import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import argparse
from street_number_recognition.models.crnn.code.model import CRNN  # 绝对导入路径
from street_number_recognition.utils.image_utils import strLabelConverter, resizeNormalize  # 绝对导入路径
#from street_number_recognition.data_loader.data_loader import get_transforms  # 绝对导入路径

def check_image_quality(image):
    """
    检查图像质量
    返回: (bool, str) - (是否通过检查, 失败原因)
    """
    # 检查图像是否为空
    if image is None:
        return False, "图像读取失败"
    
    # 检查图像尺寸
    h, w = image.shape[:2]
    if h < 16 or w < 16:  # 降低最小尺寸要求
        return False, f"图像尺寸过小: {h}x{w}"
    
    # 检查图像是否模糊
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian < 30:  # 降低模糊检测阈值
        return False, f"图像模糊: {laplacian:.2f}"
    
    # 检查图像亮度
    brightness = np.mean(gray)
    if brightness < 20:  # 降低暗度阈值
        return False, f"图像过暗: {brightness:.2f}"
    if brightness > 230:  # 提高亮度阈值
        return False, f"图像过亮: {brightness:.2f}"
    
    # 检查图像对比度
    contrast = np.std(gray)
    if contrast < 20:  # 添加对比度检查
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
    
    # 如果图像过暗
    if brightness < 20:
        alpha = 1.5  # 增加亮度
        beta = 30    # 增加偏移
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    # 如果图像过亮
    elif brightness > 230:
        alpha = 0.8  # 降低亮度
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    
    # 如果对比度过低
    if contrast < 20:
        # 使用直方图均衡化
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    return image

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_samples=None):
        self.data_dir = data_dir
        self.transform = transform
        # 获取所有图片文件并按数字顺序排序
        self.image_files = []
        for f in os.listdir(data_dir):
            if f.endswith('.png') or f.endswith('.jpg'):
                try:
                    # 验证文件名格式
                    num = int(f.split('.')[0])
                    self.image_files.append(f)
                except ValueError:
                    print(f"警告：跳过无效的文件名 {f}")
                    continue
        self.image_files.sort(key=lambda x: int(x.split('.')[0]))
        
        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]
        
        # 创建debug目录
        self.debug_dir = os.path.join(os.path.dirname(data_dir), 'debug_images')
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # 打印加载的图片数量
        print(f"成功加载 {len(self.image_files)} 张图片")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        try:
            # 读取图像
            image = cv2.imread(img_path)
            if image is None:
                print(f"警告：无法读取图像 {img_name}")
                # 返回一个占位图像
                image = np.zeros((128, 384, 3), dtype=np.uint8)
                return image, img_name, (128, 384), False
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 预处理图像
            image = preprocess_image(image)
            if image is None:
                print(f"警告：图像 {img_name} 预处理失败")
                image = np.zeros((128, 384, 3), dtype=np.uint8)
                return image, img_name, (128, 384), False
            
            # 保存原始图像尺寸
            orig_h, orig_w = image.shape[:2]
            
            # 检查图像质量
            is_valid, reason = check_image_quality(image)
            if not is_valid:
                print(f"警告：图像 {img_name} 质量检查未通过 - {reason}")
                # 保存问题图像用于调试
                debug_path = os.path.join(self.debug_dir, f"invalid_{img_name}")
                cv2.imwrite(debug_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                # 尝试再次预处理
                image = preprocess_image(image)
                if image is not None:
                    is_valid, _ = check_image_quality(image)
            
            # 应用变换
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image, img_name, (orig_h, orig_w), is_valid
            
        except Exception as e:
            print(f"处理图像 {img_name} 时出错: {str(e)}")
            # 返回一个占位图像
            image = np.zeros((128, 384, 3), dtype=np.uint8)
            return image, img_name, (128, 384), False

class strLabelConverter:
    """修正后的标签转换器"""
    def __init__(self, alphabet):
        self.alphabet = alphabet  # 字符集（如'0123456789'）
        self.blank = len(alphabet)  # 空白标签索引
    
    def decode(self, preds):
        """解码模型输出（支持批量处理）
        输入形状: (seq_len, batch_size)
        返回: 解码后的字符串列表
        """
        texts = []
        for i in range(preds.size(1)):  # 遍历批次
            char_indices = []
            previous = self.blank  # 初始化前一个字符为空白
            for t in range(preds.size(0)):  # 遍历序列
                current = preds[t, i].item()
                if current != previous and current != self.blank:
                    char_indices.append(current)
                previous = current
            text = ''.join([self.alphabet[idx] for idx in char_indices if idx < len(self.alphabet)])
            texts.append(text)
        return texts

def decode_predictions(logits, converter):
    """带置信度计算的解码函数"""
    # 输入形状: (seq_len, batch_size, num_classes)
    probs = torch.nn.functional.softmax(logits, dim=2)
    max_probs, max_indices = torch.max(probs, dim=2)
    
    batch_texts = []
    batch_confidences = []
    
    for i in range(max_indices.size(1)):  # 遍历批次
        # 解码文本
        preds = max_indices[:, i]
        text = converter.decode(preds.unsqueeze(1))[0]  # 保持维度
        
        # 计算置信度
        valid_probs = []
        previous = converter.blank
        for t in range(preds.size(0)):
            current = preds[t].item()
            if current != previous and current != converter.blank:
                valid_probs.append(max_probs[t, i].item())
            previous = current
        
        confidence = np.mean(valid_probs) if valid_probs else 0.0
        confidence = round(confidence, 4)
        
        batch_texts.append(text)
        batch_confidences.append(confidence)
    
    return batch_texts, batch_confidences

def predict(
    model_path: str, 
    test_data_dir: str, 
    output_dir: str,
    batch_size: int = 8,
    save_predictions: bool = True
) -> dict:
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # 数据转换
    transform = A.Compose([
        A.Resize(height=32, width=100),  # 根据模型输入尺寸调整
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 加载数据
    test_dataset = TestDataset(test_data_dir, transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: (
            torch.stack([item[0] for item in x]),
            [item[1] for item in x],
            [item[2] for item in x],
            torch.tensor([item[3] for item in x])
        )
    )

    # 正确初始化模型（参数与model.py定义一致）
    model = CRNN(
        num_classes=11,         # 实际类别数+1（包含空白标签）
        rnn_hidden_size=256     # 与训练时保持一致
    )

    try:
        # 加载检查点并处理多GPU训练权重
        checkpoint = torch.load(model_path, map_location=device)
        
        # 移除可能的"module."前缀（如果是多GPU训练保存的）
        state_dict = {k.replace('module.', ''): v 
                    for k, v in checkpoint['model_state_dict'].items()}
        
        # 加载权重并验证
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # 打印验证信息
        print("成功加载模型参数！")
        print(f"输入示例形状: {torch.randn(1,3,32,100).shape}")  # 验证输入尺寸
        print(f"输出示例形状: {model(torch.randn(1,3,32,100).shape)}")  # 验证输出
        
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {str(e)}\n"
                         "可能原因：\n"
                         "1. 模型定义与检查点不匹配\n"
                         "2. 输入尺寸不一致\n"
                         "3. 类别数设置错误")

    # 初始化转换器
    converter = strLabelConverter(alphabet='0123456789')

    predictions = {}
    with torch.no_grad():
        for batch_idx, (images, img_names, orig_sizes, is_valid) in enumerate(tqdm(test_loader, desc="Predicting")):
            try:
                images = images.to(device)
                
                # 前向传播
                logits = model(images)  # 形状: (seq_len, batch_size, nclass)
                
                # 解码预测
                batch_texts, batch_confidences = decode_predictions(logits, converter)
                
                # 保存结果
                for i, img_name in enumerate(img_names):
                    predictions[img_name] = {
                        "text": batch_texts[i],
                        "confidence": batch_confidences[i]
                    }

            except Exception as e:
                print(f"批次 {batch_idx} 处理失败: {str(e)}")
                # 填充默认值
                for img_name in img_names:
                    predictions[img_name] = {"text": "", "confidence": 0.0}

    # 保存结果
    if save_predictions:
        output_path = os.path.join(output_dir, "predictions.json")
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"结果已保存至 {output_path}")
    
    return predictions
# ---------------------- 命令行接口 ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test_data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    predict(
        model_path=args.model,
        test_data_dir=args.test_data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )



# 以上代码是一个完整的CRNN模型预测脚本，包含了图像预处理、数据加载、模型加载和预测等功能。
