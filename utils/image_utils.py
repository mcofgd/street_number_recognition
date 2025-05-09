import cv2
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import torch

print("utils.image_utils module loaded successfully!")

class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + '-'  # 添加空白字符
        self.dict = {char: i+1 for i, char in enumerate(alphabet)}  # 索引从1开始
        self.dict['-'] = 0  # 空白标签为0

    def encode(self, text):
        """将字符串转换为索引序列"""
        return [self.dict[char] for char in text]

    def decode(self, preds,probs=None, raw=False):
        """解码模型输出（关键修改）"""
        # 输入形状: (seq_len, batch_size)
        # 转换为字符索引
        char_list = []
        for i in range(preds.size(1)):  # 遍历批次
            pred = preds[:, i]
            char_indices = []
            previous = None
            for p in pred:  # 遍历序列
                if p != 0 and (previous != p or raw):
                    char_indices.append(p.item())
                previous = p
            text = ''.join([self.alphabet[i-1] for i in char_indices if i > 0])
            char_list.append(text)

        if probs is not None:
            confidence = np.mean([probs[i] for i in char_indices])
            return text, confidence
        return text

def resizeNormalize(size, img):
    """Resize and normalize an image."""
    h, w = size
    transform = Compose([
        Resize(height=h, width=w),
        Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])
    transformed = transform(image=img)
    return transformed['image']

def preprocess_for_crnn(image_path, target_height=32):
    """
    对图像进行预处理以适配CRNN输入
    
    Args:
        image_path: 图像路径
        target_height: 目标高度
        
    Returns:
        预处理后的图像张量
    """
    # 读取灰度图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像 {image_path}")
    
    # 保持宽高比调整大小
    h, w = image.shape
    target_w = int(w * (target_height / h))
    
    # 定义转换
    transform = Compose([
        Resize(height=target_height, width=target_w),
        Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])
    
    # 应用转换
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0)  # 添加批次维度

def crop_image(image, bbox):
    """
    根据边界框裁剪图像
    
    Args:
        image: 原始图像
        bbox: 边界框 [x1, y1, x2, y2]
        
    Returns:
        裁剪后的图像
    """
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]

def save_cropped_image(image, output_path):
    """
    保存裁剪后的图像
    
    Args:
        image: 要保存的图像
        output_path: 输出路径
    """
    cv2.imwrite(output_path, image)



