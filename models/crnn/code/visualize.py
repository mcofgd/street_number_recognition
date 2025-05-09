import os
import cv2
import json
import numpy as np
from pathlib import Path

def visualize_predictions(test_dir, pred_file, output_dir, num_samples=10):
    # 读取预测结果
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置颜色
    box_color = (0, 255, 0)  # 绿色边界框
    text_color = (255, 255, 255)  # 白色文字
    
    # 获取前num_samples个预测结果
    for i, (img_name, pred) in enumerate(list(predictions.items())[:num_samples]):
        # 读取原始图像
        img_path = os.path.join(test_dir, img_name)
        image = cv2.imread(img_path)
        
        # 获取边界框和标签
        x = pred['left'][0]
        y = pred['top'][0]
        w = pred['width'][0]
        h = pred['height'][0]
        labels = pred['label']
        
        # 绘制边界框
        cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
        
        # 将标签转换为字符串
        label_str = ''.join([str(l) for l in labels])
        
        # 计算文本大小以优化显示位置
        (text_w, text_h), baseline = cv2.getTextSize(
            label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        
        # 添加文本背景
        cv2.rectangle(image, 
                     (x, y - text_h - 10), 
                     (x + text_w, y),
                     box_color, -1)  # -1 表示填充矩形
        
        # 添加标签文本
        cv2.putText(image, label_str, 
                    (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, text_color, 2)
        
        # 保存结果
        output_path = os.path.join(output_dir, f'vis_{img_name}')
        cv2.imwrite(output_path, image)
        
        print(f'已处理图片 {i+1}/{num_samples}: {img_name}，检测到数字：{label_str}')

if __name__ == '__main__':
    test_dir = '../tcdata/mchar_test_a'
    pred_file = '../prediction_result/predictions.json'
    output_dir = '../visualization_result'
    
    visualize_predictions(test_dir, pred_file, output_dir, num_samples=10)
    print('可视化完成！结果保存在:', output_dir) 