# 对测试集进行数据预处理，从当前目录下的mchar_test中读取图片，
# 处理后的图片放在images/test中
import os
import cv2
import json
import numpy as np
from tqdm import tqdm

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

def enhance_blurry_digits(img):

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

def process_test_set(test_img_dir):
    # 创建目录结构
    os.makedirs('images/test', exist_ok=True)

    for img_name in tqdm(os.listdir(test_img_dir)):
        # 加载图像
        img_path = os.path.join(test_img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 处理
        processed_img, _ = enhance_blurry_digits(img)

        # 保存结果
        cv2.imwrite(os.path.join('images/test', img_name), processed_img)

if __name__ == "__main__":
    process_test_set('mchar_test')
    print("预处理完成！数据保存在 images/test")