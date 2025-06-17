import os
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import platform
import pathlib
import shutil

# 修复Windows路径问题
plt = platform.system()
if plt == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# 项目根目录
project_root = Path(__file__).parent.parent.parent

# 添加系统路径
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "models" / "yolov5"))
sys.path.insert(0, str(project_root / "models" / "crnn"))

# 配置导入
from street_number_recognition.config.config import (
    YOLOV5_WEIGHTS,
    CRNN_MODEL_PATH,
    INPUT_DIR,
    OUTPUT_DIR,
    YOLO_CONFIDENCE_THRESHOLD
)

# 工具函数导入
from street_number_recognition.utils.image_utils import crop_image, preprocess_for_crnn, save_processed_tensor, ensure_min_width, strLabelConverter,preprocess_for_inference,get_transforms

# 模型导入
from street_number_recognition.models.yolov5.detect import run as yolov5_detect
from street_number_recognition.models.crnn.code.test import predict_image as crnn_predict
import torchvision.transforms as transforms


def get_latest_detection_dir(output_dir: Path) -> Path:
    """获取最新的检测目录"""
    det_dirs = sorted(output_dir.glob("detections*"), 
                     key=lambda x: x.stat().st_ctime, 
                     reverse=True)
    return det_dirs[0] if det_dirs else None

def parse_yolo_detections(image_path: Path, output_dir: Path) -> List[Dict]:
    """
    YOLO结果解析
    添加类别映射验证
    """
    CLASS_MAP = {i: str(i) for i in range(10)}  # 假设类别0-9对应数字'0'-'9'
    
    det_dir = get_latest_detection_dir(output_dir)
    if not det_dir:
        print("未找到检测结果目录")
        return []
    
    label_path = det_dir / "labels" / f"{image_path.stem}.txt"
    print(f"[DEBUG] 预期标签路径: {label_path}")
    
    if not label_path.exists():
        print(f"错误: 标签文件不存在 {label_path}")
        return []

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError("无法读取图像")
        img_h, img_w = img.shape[:2]
    except Exception as e:
        print(f"图像读取失败: {str(e)}")
        return []

    detections = []
    with open(label_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 6:
                print(f"行 {line_num} 格式错误: {line}")
                continue

            try:
                class_id = int(parts[0])
                # 验证类别ID有效性
                if class_id not in CLASS_MAP:
                    print(f"行 {line_num} 无效类别ID: {class_id}")
                    continue
                
                # 坐标转换
                x_center = float(parts[1]) * img_w
                y_center = float(parts[2]) * img_h
                width = float(parts[3]) * img_w
                height = float(parts[4]) * img_h
                confidence = float(parts[5])
            except (ValueError, IndexError) as e:
                print(f"行 {line_num} 解析失败: {str(e)}")
                continue

            # 边界框计算
            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(img_w - 1, int(x_center + width / 2))
            y2 = min(img_h - 1, int(y_center + height / 2))

            if x2 <= x1 or y2 <= y1:
                print(f"行 {line_num} 无效边界框: {x1},{y1} {x2},{y2}")
                continue

            detections.append({
                'class': class_id,
                'text': CLASS_MAP[class_id],  # 使用映射表
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })

    print(f"[DEBUG] 解析到 {len(detections)} 个有效检测")
    return detections



from pathlib import Path
import cv2
import shutil

def process_single_image(image_path: str, yolo_weights: str, crnn_model_path: str) -> str:
    # 初始化路径
    image_path = Path(image_path)
    output_dir = Path(OUTPUT_DIR)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # === YOLO检测 ===
    print(f"\n=== 处理图像: {image_path} ===")
    try:
        
        yolov5_detect(
            weights=yolo_weights,
            source=str(image_path),
            project=str(output_dir),
            name='detections',
            save_txt=True,
            save_conf=True
        )
        '''
        from ultralytics import YOLO
       
        # 加载模型
        yolo_model = YOLO(yolo_weights)  # 支持 .pt / .onnx / 等格式

        # 推理
        results = yolo_model.predict(
            source=str(image_path),
            project=str(output_dir),
            name='detections',
            save=True,        # 保存图像
            save_txt=True,    # 保存标签文件
            save_conf=True,   # 保存置信度
            conf=0.5         # 可选：置信度阈值
        )
        '''
        
    except Exception as e:
        print(f"YOLOv5检测失败: {str(e)}")
        return ""

    # 解析YOLO检测结果
    detections = parse_yolo_detections(image_path, output_dir)
    if not detections:
        print(f"警告: 未检测到有效目标 {image_path.name}")
        return ""

    # === CRNN处理 ===
    crnn_input_dir = temp_dir / "crnn_input"
    crnn_input_dir.mkdir(exist_ok=True)
    
    from street_number_recognition.models.crnn.code.model import SimpleCharClassifier
    # 加载CRNN模型
    try:
        device = torch.device('cpu')  # 或者 'cuda' 如果有GPU
        checkpoint = torch.load(crnn_model_path, map_location=device)

        # 初始化模型结构
        model = SimpleCharClassifier(num_classes=10)
        model = model.to(device)

        # 加载权重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()  # 设置为评估模式

    except Exception as e:
        print(f"模型加载失败: {e}")
        raise

    # 保存预处理图像
    crnn_image_map = {}
    sorted_detections = sorted(detections, key=lambda x: x['bbox'][0])
    for i, detection in enumerate(sorted_detections):
        try:
            # 裁剪图像
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            cropped = crop_image(img, detection['bbox'])
            
            # 定义推理预处理变换
            transform = get_transforms(is_train=False)
            # 预处理
            processed = preprocess_for_inference(cropped, transform)  # 返回的是 (C,H,W)
            
            print("Input tensor shape:", processed.shape)  # 应该输出: torch.Size([1, 1, 32, W])，W >= 100

            # 保存图像
            save_path = crnn_input_dir / f"{image_path.stem}_{i}.png"
            save_processed_tensor(processed, save_path)  # 需要修改此函数以支持保存张量为图像
            
            # 在保存图像的同时记录其对应的检测项索引
            crnn_image_map[i] = {
                "path": save_path.name,
                "detection_index": i
            }
            
        except Exception as e:
            print(f"检测项 {i} 处理失败: {str(e)}")
            crnn_image_map[i] = None
    
    # CRNN预测
    crnn_results = {}
    if any(crnn_image_map.values()):
        for info in crnn_image_map.values():
            if info is not None:
                try:
                    print(f"[CRNN] 输入目录: {crnn_input_dir}")
                    
                    # 使用已加载好的 model 实例进行预测
                    result = crnn_predict(model, str(crnn_input_dir / info["path"]))  # 调用 predict_image 并传入模型和图像路径
                    
                    # 构造结果格式
                    crnn_results[info["detection_index"]] = {
                        "text": result["text"],
                        "confidence": result["confidence"]
                    }

                    print(f"[CRNN] 原始结果: '{result['text']}', 置信度: {result['confidence']:.4f}")
                except Exception as e:
                    print(f"CRNN预测异常: {str(e)}")

    # === 结果融合 ===
    final_text = ""
    for i, detection in enumerate(sorted_detections):
        yolo_text = detection['text']
        yolo_conf = detection['confidence']

        # 获取CRNN结果
        crnn_data = crnn_results.get(i, {})
        
        crnn_text = str(crnn_data.get('text', '')).strip()
        crnn_conf = crnn_data.get('confidence', 0.0)
        
        # 有效性检查
        def is_valid(text):
            return text.isdigit() and len(text) == 1
        
        valid_crnn = is_valid(crnn_text)
        valid_yolo = is_valid(yolo_text)
        
        # 决策逻辑
        if not valid_crnn and not valid_yolo:
            continue  # 跳过无效结果
        elif valid_crnn and valid_yolo:
            if yolo_text == crnn_text:
                choice = "一致"
                final_text += yolo_text
            else:
                # 加权评分
                total_score = yolo_conf * 0.6 + crnn_conf * 0.4
                 # 动态阈值
                threshold = 0.65 if crnn_conf > 0.8 else 0.55
                choice = "YOLO" if total_score > threshold else "CRNN"
                final_text += yolo_text if choice == "YOLO" else crnn_text
        elif valid_yolo:
            choice = "YOLO有效"
            final_text += yolo_text
        else:
            choice = "CRNN有效"
            final_text += crnn_text
        
        # 调试输出
        print(f"检测项 {i} 决策:")
        print(f"  YOLO: {yolo_text} (置信度 {yolo_conf:.2f})")
        print(f"  CRNN: '{crnn_text}' (置信度 {crnn_conf:.2f})")
        print(f"  选择: {choice}")

    # 清理临时文件
    shutil.rmtree(temp_dir, ignore_errors=True)
    return final_text

def main():
    results = []
    for img_name in os.listdir(INPUT_DIR):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = Path(INPUT_DIR) / img_name
        print(f"\n{'='*30}")
        print(f"Processing {img_name}...")
        
        try:
            result = process_single_image(
                img_path, 
                YOLOV5_WEIGHTS,
                CRNN_MODEL_PATH
            )
            results.append({'file_name': img_name, 'file_code': result})
        except Exception as e:
            print(f"处理失败: {str(e)}")
    
    # 保存结果
    df = pd.DataFrame(results)
    submission_path = OUTPUT_DIR / "submission.csv"
    df.to_csv(submission_path, index=False)
    print(f"\n结果已保存至: {submission_path}")

if __name__ == "__main__":
    main()



