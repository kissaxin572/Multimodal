# 基于多模态的容器异常行为检测方法

## 目录
1. [项目概述](#1-项目概述)
2. [数据集说明](#2-数据集说明)
3. [项目结构](#3-项目结构)
4. [实现方法](#4-实现方法)
5. [评估指标](#5-评估指标)
6. [环境配置](#6-环境配置)
7. [项目运行](#7-项目运行)
8. [未来改进](#8-未来改进)

## 1. 项目概述
此为毕设论文第三章实验。何为多模态？多模态就是指我们用多种不同的方式来获取、理解和处理信息。在人工智能中，系统通过同时处理文本、图像、声音等多种不同类型的信息，来做出更准确的判断或生成更丰富的输出。这种技术让机器能够像人类一样，结合多个感知方式来理解世界。
第一章是基于时序hpc数值序列的异常检测，第二章是基于快照二进制文件转图像的异常检测，第三章融合了前两章获得的两种模态，更加精准鲁棒地对异常行为进行检测

## 2. 数据集说明
   
### 2.1 采样时间
共有三种采样时间，`10s/20s/30s`，每种采样时间下分别得到csv文件和图像。

### 2.2 执行流程
以10s为例：
1. 运行`Scripts/10s.sh`，输出路径为：`/home/ubuntu20/Workspace/Datasets/Multimodal/10s/8Events/B_1.txt && M_1.txt` && `/home/ubuntu20/Workspace/Datasets/Multimodal/10s/Snapshots/B/B_1/ck1`；
2. 运行`Preprocess/10s.py`，输入路径为：`input_dirs = "Datasets/Original/10s/8Events"` && `"Datasets/Original/10s/Snapshots"`，输出路径为`"Dataset/Processed/10s/10s_100ms.csv"` && `"Datasets/Processed/10s/SFC/Gray/B{1..500}.png"`。
3. 运行`main.py`，输入路径为：`"Dataset/Processed/10s/10s_100ms.csv"` && `"Datasets/Processed/10s/SFC/Gray/B{1..500}.png"`，输出路径为`"Dataset/Results/10s/"`。

### 2.3 数据对齐
在步骤2中，每一步都要进行数据对齐，以确保csv文件和图像的顺序一致。
1. `Scripts/10s.sh`中，每运行一个良性程序或恶意程序，经过10s后，先收集时序hpc数值数据，再收集快照二进制文件，注意输出路径，便于预处理。
2. `Preprocess/10s.py`中，`hpc()`函数中，利用第一章预处理代码，最后的csv文件，保存成`sample_id` `timestamp_id` `label` `features_value`列，便于后续处理。`b2image()`函数中，利用第二章预处理代码，将`combined.img`转换为`sfc`图像，并保存到`Datasets/Processed/10s/SFC/Gray/B{1..500}.png`。
3. `main.py`中，`load_aligned_data()`将数值数据和图像数据进行对齐，送入到多模态模型中。

## 3. 项目结构
```
F:\WORKSPACE\CODE\MULTIMODAL
├─Datasets
│  ├─Original
│  │  ├─10s
│  │  │  ├─8Events
│  │  │  └─Snapshots
│  │  ├─20s
│  │  │  ├─8Events
│  │  │  └─Snapshots
│  │  └─30s
│  │      ├─8Events
│  │      └─Snapshots
│  └─Processed
│      ├─10s
│      │  ├─Byteplot
│      │  ├─Markov
│      │  ├─SFC
│      │  │  ├─Gray
│      │  │  ├─Hilbert
│      │  │  └─Zorder
│      │  └─Simhash
│      ├─20s
│      │  ├─Byteplot
│      │  ├─Markov
│      │  ├─SFC
│      │  │  ├─Gray
│      │  │  ├─Hilbert
│      │  │  └─Zorder
│      │  └─Simhash
│      └─30s
│          ├─Byteplot
│          ├─Markov
│          ├─SFC
│          │  ├─Gray
│          │  ├─Hilbert
│          │  └─Zorder
│          └─Simhash
├─Preprocess
├─Reference
├─Scripts
└─SFC
    ├─scurve
    └─test
```

## 4. 实现方法

### 4.1 深度学习方法

#### 4.1.1 主要特点
BiLSTM-Attention + ResNet50

#### 4.1.2 处理流程
1. 数据预处理
   - 图像转换
2. 模型训练与优化
   - 训练集划分
   - 模型训练
3. 模型评估与可视化


## 5. 实验评估

### 5.1 评估指标
- Accuracy: 分类准确率
- Precision: 精确率
- Recall: 召回率
- F1-Score: F1分数
- AUC: ROC曲线下面积

### 5.2 实验结果

- loss.csv && loss.png
- metrics.png && metrics histogram
- fpr_tpr.csv && roc.png

## 6. 环境配置

### 6.1 基础环境
- Python >= 3.8
- CUDA >= 11.0 (GPU加速)
- cuDNN >= 8.0

### 6.2 依赖安装

```bash
pip install -r requirements.txt
```

### 6.3 环境检查

```python
# 运行以下代码检查环境配置
import torch
import numpy
import pandas

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"NumPy版本: {numpy.__version__}")
print(f"Pandas版本: {pandas.__version__}")
```

### 6.4 注意事项
- PyTorch版本需要与CUDA版本匹配
- 建议使用虚拟环境管理依赖

## 7. 项目运行

### 7.1 数据收集
```bash
Scripts/10s.sh
```

### 7.2 数据预处理
```bash
python Preprocess/10s.py
```

### 7.3 模型训练和评估
```bash
python main.py
```

## 8. 未来改进
- 数据集扩充：探索多种图像算法和hpc时序数据的组合
- 模型改进：探索更多模型结构，如Transformer、ViT等
