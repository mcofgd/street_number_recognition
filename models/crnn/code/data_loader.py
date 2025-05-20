import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CharImageDataset(Dataset):
    def __init__(self, data_dir, json_file=None, transform=None, mode='train'):
        """
        单字符图像分类任务的数据集类，支持从整张图片中裁剪出单个字符图像
        :param data_dir: 图像文件夹路径
        :param json_file: 标签文件路径 (JSON格式)
        :param transform: 数据增强操作
        :param mode: 'train' / 'val' / 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        # 加载图像列表和标签
        if mode in ['train', 'val']:
            with open(json_file, 'r') as f:
                self.annotations = json.load(f)
            self.image_files = list(self.annotations.keys())
        elif mode == 'test':
            self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg'))]
            self.annotations = None
        else:
            raise ValueError("mode should be 'train', 'val' or 'test'")

    def __len__(self):
        return len(self.image_files)

    # 在 CharImageDataset 类的 __getitem__ 方法中
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        if self.mode in ['train', 'val']:
            annotation = self.annotations[img_name]

            labels = annotation['label']
            lefts = annotation['left']
            tops = annotation['top']
            widths = annotation['width']
            heights = annotation['height']

            # 随机选择一个字符
            char_idx = np.random.randint(len(labels))
            label = int(labels[char_idx])

            # 获取并限制坐标范围
            x_min = max(0, int(lefts[char_idx]))
            y_min = max(0, int(tops[char_idx]))
            x_max = min(w, x_min + int(widths[char_idx]))
            y_max = min(h, y_min + int(heights[char_idx]))

            cropped_image = image[y_min:y_max, x_min:x_max]

            # 防止空图像
            if cropped_image.size == 0:
                print(f"[WARNING] Empty crop in {img_name}, using full image instead.")
                cropped_image = image

            if self.transform:
                transformed = self.transform(image=cropped_image)
                cropped_image = transformed['image']

            return cropped_image, label

        elif self.mode == 'test':
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            return image, img_name
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

def get_dataloader(data_dir, json_file, batch_size, mode='train'):
    transform = get_transforms(is_train=(mode == 'train'))

    dataset = CharImageDataset(
        data_dir=data_dir,
        json_file=json_file,
        transform=transform,
        mode=mode
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=4,
        pin_memory=True,
        drop_last=(mode == 'train')
    )

