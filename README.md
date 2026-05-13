# 天气识别项目

基于深度学习的天气图像分类系统，支持多种预训练模型，提供完整的训练、评估和预测流程。

## 项目简介

本项目是一个功能完善的天气图像分类系统，能够识别多种天气状况。项目采用模块化设计，支持多种主流深度学习模型，并提供数据增强、模型评估、批量预测等完整功能。

## 核心特性

- 多模型支持：MobileNetV3、ResNet50、DenseNet121、YOLO26
- 模块化架构：配置、模型、数据、工具分离，易于扩展
- 数据增强：支持随机翻转、旋转、颜色抖动等增强方法
- 气象专用增强：模拟雾霾、雨雪效果，提升模型鲁棒性
- 完整评估流程：准确率、F1分数、分类报告等多维度评估
- 批量预测：支持单张图片、目录批量、验证集评估
- 模型对比：可视化对比多个模型性能
- 检查点管理：自动保存最佳模型和训练检查点

## 项目结构

```
Weather-Classification/
├── config.py              # 统一配置管理
├── models/                # 模型定义模块
│   ├── __init__.py       # ModelFactory 模型工厂
│   └── mobilenetv3.py    # MobileNetV3 模型示例
├── utils/                 # 工具模块
│   ├── __init__.py       # 工具模块初始化
│   ├── data_aug.py       # 数据增强（含气象专用增强）
│   ├── preprocess.py     # 图像预处理
│   └── metrics.py        # 评估指标计算
├── data/                  # 数据加载模块
│   └── __init__.py       # WeatherDataset 和 DataLoader
├── train.py              # 模型训练脚本
├── evaluate.py           # 模型评估脚本
├── predict.py            # 模型预测脚本
├── split_dataset.py      # 数据集划分脚本
├── requirements.txt      # 依赖包列表
├── README.md             # 项目文档
├── datasets/             # 原始数据集
├── train/                # 训练集（自动生成）
├── val/                  # 验证集（自动生成）
├── checkpoints/          # 模型检查点
├── predictions/          # 预测结果和评估报告
└── weights/              # 预训练权重
```

## 环境要求

- Python 3.8+
- PyTorch 2.8.0+
- CUDA 12.9（可选，用于GPU加速）

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- torch, torchvision：深度学习框架
- opencv-python, opencv-contrib-python：图像处理
- scikit-learn：评估指标计算
- pandas：数据处理
- matplotlib：可视化
- ultralytics：YOLO模型支持
- tqdm：进度条显示

## 支持的模型

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| MobileNetV3 | 轻量级，速度快 | 移动端部署、实时预测 |
| ResNet50 | 经典架构，平衡性好 | 通用场景 |
| DenseNet121 | 特征复用，参数少 | 小数据集 |
| YOLO26 | 现代架构，性能强 | 高精度要求 |

## 天气类别（4类）

Cloudy（多云）、Rain（雨）、Shine（晴）、sunrise（日出）

## 快速开始

### 1. 准备数据集

将数据集按类别组织到 `datasets/` 目录：

```
datasets/
├── Cloudy/
│   ├── image1.jpg
│   └── ...
├── Rain/
│   ├── image1.jpg
│   └── ...
├── Shine/
│   ├── image1.jpg
│   └── ...
└── sunrise/
    ├── image1.jpg
    └── ...
```

### 2. 划分数据集

首次使用需要将原始数据集划分为训练集和验证集：

```bash
python split_dataset.py
```

默认按 8:2 比例划分，可在 `split_dataset.py` 中修改 `TRAIN_RATIO`。

### 3. 训练模型

训练单个模型：

```bash
python train.py --model mobilenetv3 --batch_size 32 --lr 0.001 --epochs 50
```

支持的模型参数：mobilenetv3, resnet50, densenet121, yolo26

训练参数说明：
- `--model`: 模型名称
- `--batch_size`: 批次大小（默认32）
- `--lr`: 学习率（默认0.001）
- `--epochs`: 训练轮数（默认50）

### 4. 模型评估

对比所有模型的性能：

```bash
python evaluate.py --mode compare
```

评估单个模型：

```bash
python evaluate.py --mode single --model mobilenetv3
```

单张图片预测：

```bash
python evaluate.py --single_img test.jpg --model mobilenetv3
```

### 5. 模型预测

预测单张图片：

```bash
python predict.py --image test.jpg --model mobilenetv3
```

批量预测目录中的所有图片：

```bash
python predict.py --directory ./test_images --output results.csv
```

预测整个验证集并生成详细报告：

```bash
python predict.py --val ./val --model mobilenetv3
```

## 配置说明

所有配置项都在 `config.py` 中的 `Config` 类中统一管理：

```python
class Config:
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据配置
    NUM_CLASSES = 4                       # 分类数量
    CLASS_NAMES = ['Cloudy', 'Rain', 'Shine', 'sunrise']  # 类别名称列表
    IMAGE_SIZE = (224, 224)               # 图像尺寸
    MEAN = [0.485, 0.456, 0.406]          # 归一化均值
    STD = [0.229, 0.224, 0.225]           # 归一化标准差
    
    # 模型配置
    SUPPORTED_MODELS = ["mobilenetv3", "resnet50", "densenet121", "yolo26"]  # 支持的模型列表
    
    # 路径配置
    DATA_DIR = "./datasets"               # 原始数据目录
    TRAIN_DIR = "./train"                 # 训练集目录
    VAL_DIR = "./val"                     # 验证集目录
    CHECKPOINT_DIR = "./checkpoints"      # 检查点目录
    OUTPUT_DIR = "./predictions"          # 输出目录
    
    # 训练配置
    TRAIN_RATIO = 0.8                     # 训练集比例
    BATCH_SIZE = 32                       # 批次大小
    NUM_EPOCHS = 50                       # 训练轮数
    LEARNING_RATE = 0.001                 # 学习率
```

## 数据增强

项目提供了丰富的数据增强方法：

### 基础增强
- 随机水平翻转（p=0.5）
- 随机旋转（±15度）
- 颜色抖动（亮度、对比度、饱和度）
- 标准化（ImageNet均值和标准差）

### 气象专用增强
- **雾霾效果模拟**：`add_fog_effect()` - 提升模型对低能见度天气的识别能力
- **雨雪噪声模拟**：`add_weather_noise()` - 增强模型对恶劣天气的鲁棒性

使用示例：

```python
from utils import add_fog_effect, add_weather_noise
import cv2

# 添加雾霾效果
img = cv2.imread('image.jpg')
foggy_img = add_fog_effect(img, intensity_range=(0.2, 0.5))

# 添加雨雪噪声
rainy_img = add_weather_noise(img, noise_type="rain", intensity=0.02)
```

## 输出文件

### 训练输出
- `checkpoints/checkpoint_{model}.pth`: 训练检查点（包含优化器状态、训练历史等）
- `best_{model}.pth`: 最佳模型权重（仅模型参数）

### 评估输出
- `predictions/all_models_comparison.png`: 模型对比柱状图
- `predictions/all_models_comparison.csv`: 模型对比表格

### 预测输出
- `predictions/classification_report.txt`: 详细分类报告
- `predictions/val_predictions.csv`: 验证集预测结果
- `predictions.csv`: 批量预测结果（自定义输出路径）

## 🔄 更换数据集

### 步骤 1：准备新数据集

将你的新数据集按类别组织到 `datasets/` 目录下，结构如下：

```
datasets/
├── 类别1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── 类别2/
│   ├── image1.jpg
│   └── ...
└── ...
```

### 步骤 2：修改配置文件

在 `config.py` 中修改以下参数：

```python
# 修改类别数量
NUM_CLASSES = 你的类别数量

# 修改类别名称列表
CLASS_NAMES = [
    '类别1', '类别2', '类别3', ...  # 替换为你的类别名称
]

# 根据需要调整图像大小
IMAGE_SIZE = (224, 224)  # 或其他尺寸

# 如果新数据集的图像分布不同，可能需要重新计算均值和标准差
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
```

### 步骤 3：重新划分数据集

运行数据集划分脚本：

```bash
python split_dataset.py
```

这会将 `datasets/` 中的数据按 8:2 的比例划分到 `train/` 和 `val/` 目录。

### 步骤 4：清理旧模型（可选）

如果需要从头开始训练，删除旧的模型文件：

```bash
del best_*.pth
del checkpoints\checkpoint_*.pth
```

### 步骤 5：重新训练模型

```bash
python train.py --model mobilenetv3
```

### 可选调整

如果需要调整训练参数，在 `config.py` 中修改：
- `BATCH_SIZE`：批次大小
- `NUM_EPOCHS`：训练轮数
- `LEARNING_RATE`：学习率
- `TRAIN_RATIO`：训练集比例（在 split_dataset.py 中也要修改）

## ➕ 添加新模型

### 步骤 1：在 config.py 中添加模型名称

在 `config.py` 的 `SUPPORTED_MODELS` 列表中添加新模型：

```python
SUPPORTED_MODELS = ["mobilenetv3", "resnet50", "densenet121", "yolo26", "efficientnet", "vit"]
```

### 步骤 2：在 models/__init__.py 中添加模型创建逻辑

在 `models/__init__.py` 的 `create_model` 方法中添加新的 `elif` 分支。

#### 常见模型添加示例

**EfficientNet-B0:**
```python
elif model_name == "efficientnet":
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, Config.NUM_CLASSES)
```

**ViT (Vision Transformer):**
```python
elif model_name == "vit":
    from torchvision.models import vit_b_16
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, Config.NUM_CLASSES)
```

**ResNet101:**
```python
elif model_name == "resnet101":
    from torchvision.models import resnet101
    model = resnet101(weights=None)
    model.fc = nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
```

**VGG16:**
```python
elif model_name == "vgg16":
    from torchvision.models import vgg16
    model = vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, Config.NUM_CLASSES)
```

**ShuffleNetV2:**
```python
elif model_name == "shufflenet":
    from torchvision.models import shufflenet_v2_x1_0
    model = shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, Config.NUM_CLASSES)
```

### 步骤 3：训练新模型

添加完成后，可以直接使用新模型进行训练：

```bash
python train.py --model efficientnet
python train.py --model vit
```

### 查找分类层位置

不同模型的分类层位置不同，以下是常见模型的分类层位置：

| 模型 | 分类层位置 | 替换方式 |
|------|-----------|---------|
| ResNet 系列 | `model.fc` | `model.fc = nn.Linear(in_features, num_classes)` |
| DenseNet | `model.classifier` | `model.classifier = nn.Linear(in_features, num_classes)` |
| MobileNetV3 | `model.classifier[3]` | `model.classifier[3] = nn.Linear(in_features, num_classes)` |
| EfficientNet | `model.classifier[1]` | `model.classifier[1] = nn.Linear(in_features, num_classes)` |
| VGG | `model.classifier[6]` | `model.classifier[6] = nn.Linear(in_features, num_classes)` |
| ShuffleNet | `model.fc` | `model.fc = nn.Linear(in_features, num_classes)` |
| ViT | `model.heads.head` | `model.heads.head = nn.Linear(in_features, num_classes)` |

## 项目特色

### 1. 模块化设计
- 将配置、模型、工具、数据加载等功能分离到独立模块
- 提高代码复用性和可维护性
- 便于团队协作和功能扩展

### 2. 统一配置管理
- 所有配置集中在 `config.py`
- 避免配置分散和重复定义
- 便于统一修改和管理

### 3. 模型工厂模式
- `ModelFactory` 统一管理模型的创建和加载
- 简化模型切换和扩展
- 支持动态添加新模型

### 4. 气象专用数据增强
- 模拟雾霾、雨雪等恶劣天气效果
- 提升模型在复杂天气条件下的鲁棒性
- 针对天气识别任务优化

### 5. 完善的评估体系
- 多维度评估指标（准确率、F1分数等）
- 详细的分类报告
- 可视化模型对比

### 6. 灵活的预测接口
- 支持单张图片预测
- 支持批量预测
- 支持验证集整体评估

## 性能优化建议

1. **使用 GPU 训练**：项目自动检测并使用 GPU，可大幅提升训练速度
2. **调整批次大小**：根据显存大小调整 `batch_size`，充分利用硬件资源
3. **学习率调度**：已集成 ReduceLROnPlateau 调度器，自动调整学习率
4. **数据增强**：启用数据增强可提升模型泛化能力
5. **模型选择**：根据实际需求选择合适的模型（速度 vs 精度）
6. **混合精度训练**：可考虑使用 torch.cuda.amp 加速训练

## 常见问题

### Q1: 训练时显存不足怎么办？
A: 减小 `batch_size`，或使用更小的模型（如 MobileNetV3）。

### Q2: 如何提高模型准确率？
A: 
- 增加训练数据量
- 使用数据增强
- 调整学习率和训练轮数
- 尝试不同的模型架构

### Q3: 如何使用预训练权重？
A: 在 `models/__init__.py` 中，将 `weights=None` 改为 `weights='DEFAULT'`。

### Q4: 预测时如何设置置信度阈值？
A: 在 `predict.py` 的 `predict` 方法中添加阈值判断逻辑。

### Q5: 如何导出模型用于部署？
A: 使用 `torch.jit.trace()` 或 `torch.onnx.export()` 导出模型。

## 注意事项

1. 确保数据集目录结构正确：`datasets/{类别名}/图片`
2. 首次使用前运行 `split_dataset.py` 划分数据集
3. 训练前确保有足够的 GPU 内存
4. 模型权重文件会自动保存到项目根目录
5. 更换数据集后记得修改 `config.py` 中的类别配置
