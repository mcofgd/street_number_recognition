# 门牌号识别系统（Street Number Recognition）

本项目是一个基于深度学习的门牌号自动识别系统，集成了YOLOv5用于门牌号检测，CRNN用于数字识别。适合初学者学习目标检测与字符识别的完整流程。

---

## 项目组成部分介绍

本项目主要由以下几个部分组成：

1. **配置文件（config/）**  
   - `config.py`：定义模型路径、输入输出目录、模型参数等配置，方便统一管理。

2. **数据目录（data/）**  
   - `images/`：存放原始图片和处理后的图片。  
   - `output/`：存放检测与识别结果，包括YOLO格式的标签和可视化图片。

3. **数据加载与预处理（data_loader/）**  
   - `data_loader.py`：负责数据加载和预处理，将原始图片转换为适合模型训练的格式。  
   - `train_loader.py`：用于训练数据的加载。  
   - `test_loader.py`：用于测试数据的加载。

4. **模型目录（models/）**  
   - `yolov5/`：包含YOLOv5目标检测模型的代码和权重文件。  
   - `crnn/`：包含CRNN字符识别模型的代码和权重文件，其中：  
     - `code/test.py`：用于CRNN模型的测试和推理。  
     - `code/model.py`：定义CRNN模型结构。  
     - `code/utils.py`：提供CRNN模型所需的工具函数。

5. **主程序入口（src/）**  
   - `main.py`：项目的主程序入口，负责调用YOLOv5和CRNN模型进行门牌号检测和识别。

6. **工具函数（utils/）**  
   - `image_utils.py`：提供图像处理相关的工具函数，如裁剪、预处理等。  
   - `text_utils.py`：提供文本处理相关的工具函数，如标签转换等。

7. **依赖管理（requirements.txt）**  
   - 列出项目所需的Python库，方便用户安装依赖。

8. **项目说明文档（README.md）**  
   - 详细介绍项目的使用方法、目录结构、常见问题等，帮助初学者快速上手。

---

## 目录结构说明

```
street_number_recognition/
├── config/             # 配置文件（如模型路径、参数等）
├── data/               # 数据目录
│   ├── images/         # 存放原始和处理后的图片
│   └── output/         # 检测与识别结果输出
├── data_loader/        # 数据加载与预处理脚本
├── models/             # 模型目录
│   ├── yolov5/         # YOLOv5相关代码与权重
│   └── crnn/           # CRNN相关代码与权重
├── src/                # 主程序入口（main.py）
├── utils/              # 工具函数
├── requirements.txt    # 依赖包列表
└── README.md           # 项目说明文档
```

---

## 环境与依赖安装

建议使用Python 3.8及以上版本。

1. **克隆项目**
   ```bash
git clone [仓库地址]
cd street_number_recognition
```

2. **安装依赖**
   ```bash
pip install -r requirements.txt
```

---

## 数据准备与预处理

1. **原始数据**
   - 请将原始门牌号图片放入 `data/images/` 目录下。
   - 若有标注文件（如`mchar_train.json`），请一并放入项目根目录。

2. **数据预处理**
   - 运行数据预处理脚本，将原始图片处理为适合模型训练和推理的格式：
     ```bash
     python data_loader/data_loader.py
     ```
   - 处理后图片会保存在 `data/images/train/`，标签保存在 `labels/train/`。
   - 测试集处理后建议放入 `data/images/mchar_test_a_processed/`。

---

## 预训练模型权重获取

- **YOLOv5权重**：请将训练好的YOLOv5权重文件（如`best.pt`）放入 `models/yolov5/weights/` 目录。
- **CRNN权重**：请将训练好的CRNN权重文件（如`best_model.pth`）放入 `models/crnn/user_data/model_data/` 目录。

> 权重文件可通过自行训练或向项目作者获取。

---

## 运行方法

1. **配置参数**
   - 可在 `config/config.py` 中修改模型路径、输入输出目录等参数。

2. **执行主程序**
   ```bash
   python src/main.py
   ```
   - 默认会读取 `data/images/mchar_test_a_processed/` 下的图片，输出结果到 `data/output/`。

3. **参数说明**
   - YOLOv5置信度阈值、CRNN输入尺寸等可在 `config/config.py` 中调整。

---

## 输出结果说明

- 检测与识别结果保存在 `data/output/detectionsXXXX/` 目录下：
  - `labels/`：每张图片的检测框与类别标签（YOLO格式）
  - 可视化图片：带检测框和识别结果的图片
- 汇总结果（如`submission.csv`）可用于后续评测或提交。

---

## 常见问题与注意事项

1. **图片读取失败/无检测结果**
   - 请确保图片路径正确、图片清晰。
2. **CRNN未能正常识别**
   - 检查模型类别数、输入尺寸与训练时是否一致。
   - 若有报错，建议检查 `models/crnn/code/test.py` 与 `model.py` 的参数对接。
3. **数据集缺失**
   - 由于文件大小原因，部分图片未上传。请自行准备测试图片并放入指定目录。
4. **内存占用高**
   - 批量处理大规模图片时建议分批进行。

---

## 贡献与交流

- 欢迎提交Issue和Pull Request，帮助完善项目。
- 有问题可在Issue区留言，或通过邮箱联系作者（待补充）。

---

## 许可证

本项目许可证待定。