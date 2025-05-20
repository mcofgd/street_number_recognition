# 门牌号识别系统

这是一个基于深度学习的门牌号识别系统，使用YOLOv5进行目标检测和CRNN进行文字识别，能够从图像中准确识别出门牌号。

## 项目结构

```
street_number_recognition/
├── config/             # 配置文件目录
├── data/              # 数据目录
├── data_loader/       # 数据加载器
├── models/            # 模型目录
│   ├── yolov5/       # YOLOv5目标检测模型
│   └── crnn/         # CRNN文字识别模型
├── src/              # 源代码目录
│   └── main.py       # 主程序入口
└── utils/            # 工具函数目录
```

## 功能特点

- 使用YOLOv5进行门牌号区域检测
- 使用CRNN进行文字识别
- 支持批量处理图像
- 提供详细的处理日志和可视化结果
- 支持Windows和Linux系统

## 环境要求

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Pandas

## 安装步骤

1. 克隆仓库：
```bash
git clone [repository_url]
cd street_number_recognition
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 下载预训练模型：
- YOLOv5权重文件
- CRNN模型文件

## 使用方法

1. 准备输入图像：
将需要识别的图像放入指定目录

2. 运行识别程序：
```bash
python src/main.py
```

## 处理流程

1. 图像预处理
2. YOLOv5目标检测
3. 区域裁剪
4. CRNN文字识别
5. 结果输出

## 输出结果

- 检测结果保存在 `output/detections` 目录
- 包含检测框坐标、置信度和识别文字
- 提供可视化结果

## 注意事项

- 确保输入图像清晰度
- 建议使用标准格式的门牌号图像
- 处理大量图像时注意内存使用

## 开发进度

- [x] 基础框架搭建
- [x] YOLOv5检测模块
- [x] CRNN识别模块
- [x] 图像预处理
- [x] 结果可视化
- [ ] 批量处理优化
- [ ] 模型训练脚本
- [ ] 性能优化
目前进度是能够集成yolo，但是crnn因为参数对接问题（识别类别数、图像大小）未集成完毕，需要修改，其主要使用的crnn相关代码位于models/cenn/code/test.py，改文件已实现封装，但是调用有问题，与同目录下model.py的协调也因为修改而不是很对
由于文件大小原因，删掉了需要检测的图片数据，在本地运行需要将data_loader运行处理后的数据加入data/mchar_test_a_processed，因为模型读取的是该目录下文件，原始图片可以不管。

## 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## 许可证

[待定]

## 联系方式

[待补充] 