import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DoorplateDataset(Dataset):
    def __init__(self, data_dir, json_file, transform=None, max_seq_length=10, is_test=False):
        self.data_dir = data_dir
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.is_test = is_test
        
        # 读取标注文件
        if not self.is_test:
            with open(json_file, 'r') as f:
                self.annotations = json.load(f)
            self.image_files = list(self.annotations.keys())
        else:
            self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.png') or f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # 读取图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if not self.is_test:
            # 获取标注信息
            ann = self.annotations[img_name]
            
            # 构建目标数据
            labels = ann['label']
            lefts = ann['left']
            tops = ann['top']
            widths = ann['width']
            heights = ann['height']
            
            # 创建目标序列和边界框
            seq = np.zeros(self.max_seq_length, dtype=np.int64)
            seq_len = min(len(labels), self.max_seq_length)
            
            # 填充序列
            for i in range(seq_len):
                seq[i] = labels[i]
            
            # 归一化边界框坐标
            h, w = image.shape[:2]
            bbox = np.array([lefts[0]/w, tops[0]/h, widths[0]/w, heights[0]/h], dtype=np.float32)
                
            # 应用变换
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image, seq, seq_len, bbox
        
        else:
            # 测试集只返回图像
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            
            return image, img_name

def get_transforms(is_train=True):
    if is_train:
        return A.Compose([
            A.Resize(height=128, width=384),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
                A.RandomGamma(),
                A.GaussNoise(),
            ], p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=128, width=384),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def get_dataloader(data_dir, json_file, batch_size, is_train=True, is_test=False):
    transform = get_transforms(is_train)
    dataset = DoorplateDataset(data_dir, json_file, transform, is_test=is_test)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4,
        pin_memory=True
    )