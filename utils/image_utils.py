# street_number_recognition/utils/image_utils.py

import cv2
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import torch
from torchvision import transforms

print("utils.image_utils module loaded successfully!")

class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + '-'  # 添加空白字符
        self.dict = {char: i+1 for i, char in enumerate(alphabet)}  # 索引从1开始
        self.dict['-'] = 0  # 空白标签为0

    def encode(self, text):
        """将字符串转换为索引序列"""
        return [self.dict[char] for char in text]

    def decode(self, preds, probs=None, raw=False):
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



import cv2
import numpy as np
import torch
from torchvision import transforms
import os
from albumentations import Compose, Resize, Normalize
'''
def preprocess_for_crnn(cropped_img, target_height=32, min_width=100):
    """
    预处理裁剪后的图像，使其适合输入CRNN模型。
    
    Args:
        cropped_img: 裁剪后的图像（numpy array）
        target_height: 目标高度，默认32
        min_width: 图像最小宽度，默认值为100
        
    Returns:
        归一化后的Tensor (B, C, H, W)
    """
    # 确保图像是灰度格式
    if len(cropped_img.shape) == 3 and cropped_img.shape[2] == 3:
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    h, w = cropped_img.shape
    scale = target_height / h
    new_w = max(int(w * scale), min_width)
    resized = cv2.resize(cropped_img, (new_w, target_height))  # shape: (H, W)

    # 添加通道维度并转换为 (C, H, W)
    resized = resized[np.newaxis, :, :]  # shape: (1, H, W)

    # 转换为 Tensor + 归一化
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = transforms.Normalize([0.5], [0.5])(tensor)

    return tensor.unsqueeze(0)  # shape: (1, 1, 32, >=100)
'''
def preprocess_for_crnn(cropped_img, target_height=32, min_width=100):
    """
    预处理裁剪后的图像，使其适合输入CRNN模型。
    
    Args:
        cropped_img: 裁剪后的图像（numpy array）
        target_height: 目标高度，默认32
        min_width: 图像最小宽度，默认值为100
        
    Returns:
        归一化后的Tensor (B, C, H, W)
    """
    # 如果图像是灰度，则转换为三通道
    if len(cropped_img.shape) == 2 or (len(cropped_img.shape) == 3 and cropped_img.shape[2] == 1):
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)

    h, w, c = cropped_img.shape  # 注意这里现在获取的是彩色图像的尺寸，包括通道数
    scale = target_height / h
    new_w = max(int(w * scale), min_width)
    resized = cv2.resize(cropped_img, (new_w, target_height))  # shape: (H, W, C)

    # 确保图像达到最小宽度
    if new_w < min_width:
        resized = ensure_min_width(resized, min_width=min_width)

    # 添加通道维度并转换为 (C, H, W)
    resized = resized.transpose((2, 0, 1))  # shape: (C, H, W)

    # 转换为 Tensor + 归一化
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = transforms.Normalize([0.5]*c, [0.5]*c)(tensor)  # 根据通道数调整归一化参数

    return tensor.unsqueeze(0)  # shape: (1, C, 32, >=100)

def preprocess_for_inference(image, transform):
    """
    推理阶段的预处理函数
    
    Args:
        image: 输入的原始图像 (numpy array)
        transform: 数据预处理变换
        
    Returns:
        归一化后的Tensor (1, C, H, W)
    """
    # 转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 应用预处理变换
    transformed = transform(image=image)
    tensor = transformed['image']

    return tensor.unsqueeze(0)  # 添加 batch 维度

def ensure_min_width(image, min_width=100):
    """
    修正后的宽度保障函数（兼容CHW/HWC格式）
    """
    # 如果是张量，转成numpy处理
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

    # 统一为 HWC 格式处理
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]  # 单通道变三维 (H, W, 1)

    h, w, c = image.shape

    if w >= min_width:
        return image

    # 创建填充图像
    padded = np.ones((h, min_width, c), dtype=image.dtype) * 255
    padded[:, :w] = image

    return padded

def save_processed_tensor(tensor, save_path):
    """
    保存经过归一化的图像张量（适用于字符分类器）
    
    Args:
        tensor: 输入的Tensor，可以是 (B, C, H, W) 或 (C, H, W)
        save_path: 保存路径
    """
    # 如果有batch维度，去掉
    if tensor.dim() == 4:
        tensor = tensor[0]

    # 反归一化 [0, 1] -> [0, 255]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # 反向操作：x * std + mean

    # 转换为 numpy 并调整格式
    img = tensor.mul(255).byte().cpu().numpy()  # (C, H, W)
    img = img.transpose(1, 2, 0)  # (H, W, C)

    # 如果是单通道，去掉多余维度
    if img.shape[2] == 1:
        img = img[:, :, 0]

    cv2.imwrite(str(save_path), img)

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.LongestMaxSize(max_size=32, p=1.0),
            A.PadIfNeeded(min_height=32, min_width=32,
                          border_mode=cv2.BORDER_CONSTANT,
                          fill_value=0, p=1.0),

            # 数据增强
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.RandomGamma(gamma_limit=(80, 120)),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 3)),
                A.MedianBlur(blur_limit=3),
            ], p=0.3),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.LongestMaxSize(max_size=32, p=1.0),
            A.PadIfNeeded(min_height=32, min_width=32,
                          border_mode=cv2.BORDER_CONSTANT,
                          fill_value=0, p=1.0),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def _save_single_tensor(tensor, save_path):
    """
    辅助函数，用于保存单个图像张量
    
    Args:
        tensor: 输入的Tensor，形状应为 (C, H, W)
        save_path: 保存路径
    """
    # 反归一化
    tensor = tensor * 0.5 + 0.5  # 反归一化 [0,1]
    tensor = tensor.clamp(0, 1)  # 确保在有效范围内
    
    # 转换为numpy并调整格式
    img = tensor.mul(255).byte().cpu().numpy()  # (C,H,W)
    img = img.transpose(1, 2, 0)  # 转为 (H,W,C)
    
    # 如果是单通道，去掉多余的维度
    if img.shape[2] == 1:
        img = img[:, :, 0]
    
    cv2.imwrite(str(save_path), img)

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

