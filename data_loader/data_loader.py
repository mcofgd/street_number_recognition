# 对训练集进行数据预处理，从当前目录下的mchar_train中读取图片，并读取mchar_train.json
# 处理后的图片放在images/train中,处理后的各个图片的边框信息分别以txt格式存储到了labels/train中
import os
import cv2
import json
import numpy as np
from tqdm import tqdm

# 配置参数
CLAHE_CLIP_LIMIT = 3.5          # 对比度增强强度（针对模糊文本）
EDGE_PRESERVE_SIGMA = 65        # 边缘保留滤波强度
MORPH_KERNEL_SIZE = (2, 2)      # 形态学操作核大小

def enhance_blurry_digits(img):

    # LAB空间对比度增强
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    merged = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # 边缘保留降噪
    filtered = cv2.edgePreservingFilter(enhanced, flags=cv2.RECURS_FILTER,
                                        sigma_s=EDGE_PRESERVE_SIGMA, sigma_r=0.35)

    # 二值化校验
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 形态学闭操作填充数字内部
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return filtered, morph

def convert_to_yolo_format():
    # 创建目录结构
    os.makedirs('images/train', exist_ok=True)
    os.makedirs('labels/train', exist_ok=True)

    # 加载标注数据
    with open('mchar_train.json') as f:
        annotations = json.load(f)

    # 处理每张图像
    for img_name, bboxes in tqdm(annotations.items(), desc="Processing 744-like Images"):
        img_path = os.path.join('mchar_train', img_name)
        if not os.path.exists(img_path):
            continue

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 处理图像
        processed_img, valid_mask = enhance_blurry_digits(img)
        img_h, img_w = processed_img.shape[:2]

        # 生成YOLO标签
        txt_path = os.path.join('labels/train', os.path.splitext(img_name)[0] + '.txt')
        valid_boxes = []

        with open(txt_path, 'w') as f_txt:
            for i in range(len(bboxes['label'])):
                # 坐标整数化处理
                x_min = int(round(bboxes['left'][i]))
                y_min = int(round(bboxes['top'][i]))
                width = int(round(bboxes['width'][i]))
                height = int(round(bboxes['height'][i]))

                # # 边界保护
                # x_max = min(x_min + width, valid_mask.shape[1])
                # y_max = min(y_min + height, valid_mask.shape[0])
                # if x_min >= x_max or y_min >= y_max:
                #     continue
                #
                # # 校验标注有效性
                # roi = valid_mask[y_min:y_max, x_min:x_max]
                # if cv2.countNonZero(roi) < max(10, width*height*0.05):  # 动态阈值
                #     continue

                # 计算归一化坐标
                x_center = (x_min + width/2) / img_w
                y_center = (y_min + height/2) / img_h
                w_norm = width / img_w
                h_norm = height / img_h

                f_txt.write(f"{bboxes['label'][i]} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                valid_boxes.append(i)

        # 保存处理后的图像
        if valid_boxes:
            cv2.imwrite(os.path.join('images/train', img_name), processed_img)

if __name__ == "__main__":
    convert_to_yolo_format()
    print("预处理完成！数据保存在 images/train 和 labels/train")