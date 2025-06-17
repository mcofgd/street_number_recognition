import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent

# 模型路径
MODEL_DIR = ROOT_DIR / "models"
#YOLOV5_WEIGHTS = MODEL_DIR / "yolov5" / "weights" / "best.pt"
YOLOV5_WEIGHTS = MODEL_DIR / "yolov5" / "weights" /"weights2"/ "best.pt"
#YOLOV5_WEIGHTS = MODEL_DIR / "yolov5" / "weights" /"v8"/ "best.pt"
CRNN_MODEL_PATH = MODEL_DIR / "crnn" / "user_data" / "model_data" / "best_model.pth"

# 数据路径
DATA_DIR = ROOT_DIR / "data"
INPUT_DIR = DATA_DIR / "images" /"mchar_test_a"
OUTPUT_DIR = DATA_DIR / "output"

# 模型参数
YOLO_CONFIDENCE_THRESHOLD = 0.7
CRNN_IMG_HEIGHT = 32
CRNN_IMG_WIDTH = 100

# 创建必要的目录
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR / "yolov5" / "weights", exist_ok=True)
os.makedirs(MODEL_DIR / "crnn" / "user_data" / "model_data", exist_ok=True) 