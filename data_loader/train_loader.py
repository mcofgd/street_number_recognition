# 对训练集进行数据预处理，从当前目录下的mchar_train中读取图片，并读取mchar_train.json
# 处理后的图片放在images/train中,处理后的各个图片的边框信息分别以txt格式存储到了labels/train中

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import random

# 可配置参数（根据实际效果调整）
class PreprocessConfig:
    # 图像增强参数
    CLAHE_CLIP_LIMIT = 3.5          # 对比度受限自适应直方图均衡化强度
    ADAPTIVE_THRESH_BLOCK = 21     # 自适应阈值块大小（奇数）
    USM_STRENGTH = 1.2             # 反锐化掩模强度

    # 形态学操作参数
    MORPH_KERNEL_SHAPE = cv2.MORPH_ELLIPSE  # 核形状
    MORPH_KERNEL_SIZE = (2, 2)     # 形态学操作核尺寸

    # 数据增强概率
    AUGMENT_PROB = 0.5             # 数据增强应用概率
    MAX_ROTATION_ANGLE = 15        # 最大旋转角度
    NOISE_STD = 8                  # 高斯噪声标准差

    # 标注校验参数
    MIN_PIXEL_RATIO = 0.08         # 有效像素占比阈值
    MIN_BOX_AREA = 32              # 最小框面积（像素）

cfg = PreprocessConfig()

def enhance_digits(img):

    # 阶段1：自适应对比度增强
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 双重CLAHE增强
    clahe = cv2.createCLAHE(clipLimit=cfg.CLAHE_CLIP_LIMIT, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    l_combined = cv2.addWeighted(l, 0.3, l_clahe, 0.7, 0)

    # 阶段2：锐化处理（USM）
    blurred = cv2.GaussianBlur(l_combined, (0,0), 3)
    usm = cv2.addWeighted(l_combined, 1 + cfg.USM_STRENGTH,
                          blurred, -cfg.USM_STRENGTH, 0)

    # 合并通道
    merged_lab = cv2.merge((usm, a, b))
    enhanced = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # 阶段3：自适应二值化
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   cfg.ADAPTIVE_THRESH_BLOCK, 5)

    # 阶段4：形态学优化
    kernel = cv2.getStructuringElement(cfg.MORPH_KERNEL_SHAPE,
                                       cfg.MORPH_KERNEL_SIZE)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return enhanced, morph

def apply_augmentation(img):



    # 添加高斯噪声
    if random.random() < cfg.AUGMENT_PROB:
        noise = np.random.normal(0, cfg.NOISE_STD, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 适当使用主动模糊
    if random.random() < 0.3:  # 30%概率添加模糊
        blur_type = random.choice(["gaussian", "motion"])

        if blur_type == "gaussian":
            ksize = random.choice([3,5,7])  # 随机核大小
            img = cv2.GaussianBlur(img, (ksize,ksize), 0)

        elif blur_type == "motion":
            kernel_size = random.randint(15,25)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel /= kernel_size
            img = cv2.filter2D(img, -1, kernel)

    return img

def validate_annotation(bbox_mask, x_min, y_min, width, height):
    """验证标注有效性"""
    # 边界检查
    img_h, img_w = bbox_mask.shape
    x_max = x_min + width
    y_max = y_min + height
    if x_max > img_w or y_max > img_h:
        return False

    # 有效区域检查
    roi = bbox_mask[y_min:y_max, x_min:x_max]
    total_pixels = width * height
    if total_pixels < cfg.MIN_BOX_AREA:
        return False

    white_pixels = cv2.countNonZero(roi)
    return (white_pixels / total_pixels) > cfg.MIN_PIXEL_RATIO

def process_image(img_path, bboxes):

    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        return None, []

    # 图像增强
    enhanced_img, valid_mask = enhance_digits(img)

    # 数据增强
    enhanced_img = apply_augmentation(enhanced_img)

    # 转换为YOLO格式
    img_h, img_w = enhanced_img.shape[:2]
    valid_annotations = []

    for i in range(len(bboxes['label'])):
        # 坐标整数化
        x_min = int(round(bboxes['left'][i]))
        y_min = int(round(bboxes['top'][i]))
        width = int(round(bboxes['width'][i]))
        height = int(round(bboxes['height'][i]))

        # 校验标注有效性
        if not validate_annotation(valid_mask, x_min, y_min, width, height):
            continue

        # 计算归一化坐标
        x_center = (x_min + width/2) / img_w
        y_center = (y_min + height/2) / img_h
        w_norm = width / img_w
        h_norm = height / img_h

        valid_annotations.append(
            (bboxes['label'][i], x_center, y_center, w_norm, h_norm)
        )

    return enhanced_img, valid_annotations

def convert_to_yolo_format():
    # 创建目录结构
    os.makedirs('images/train', exist_ok=True)
    os.makedirs('labels/train', exist_ok=True)

    # 加载标注数据
    with open('mchar_train.json') as f:
        annotations = json.load(f)

    # 统计信息
    total_valid = 0
    skipped_images = []

    # 处理流程
    for img_name, bboxes in tqdm(annotations.items(), desc="Processing"):
        img_path = os.path.join('mchar_train', img_name)
        if not os.path.exists(img_path):
            skipped_images.append(img_name)
            continue

        # 处理图像
        processed_img, yolo_annos = process_image(img_path, bboxes)
        if not yolo_annos:
            skipped_images.append(img_name)
            continue

        # 保存处理结果
        txt_path = os.path.join('labels/train', os.path.splitext(img_name)[0] + '.txt')
        with open(txt_path, 'w') as f_txt:
            for anno in yolo_annos:
                f_txt.write(f"{anno[0]} {anno[1]:.6f} {anno[2]:.6f} {anno[3]:.6f} {anno[4]:.6f}\n")

        cv2.imwrite(os.path.join('images/train', img_name), processed_img)
        total_valid += 1

    # 输出统计信息
    print(f"\n处理完成！有效图片：{total_valid}，跳过图片：{len(skipped_images)}")
    if skipped_images:
        print("跳过的图片列表已保存至 skipped_images.txt")
        with open("skipped_images.txt", "w") as f:
            f.write("\n".join(skipped_images))

if __name__ == "__main__":
    convert_to_yolo_format()