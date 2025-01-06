import os
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import pandas as pd
from PIL import Image
from torchvision import transforms

# 定义加载并对齐图像与时序数据的函数
def load_aligned_data(image_dir, sequence_file, img_size, batch_size):
    # 加载时序数据CSV文件
    sequence_data = pd.read_csv(sequence_file)

    # 定义图像数据的预处理转换，包括调整大小、转换为RGB格式、归一化等
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # 调整图像大小
        transforms.Lambda(lambda x: x.convert("RGB")),  # 将图像转换为RGB
        transforms.ToTensor(),  # 转换为Tensor格式
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 归一化图像
    ])

    aligned_data = []  # 用于存储对齐后的数据

    # 遍历每个标签（良性和恶性）进行数据加载
    for label in ["B", "M"]:  # "B" 表示良性，"M" 表示恶性
        image_path = os.path.join(image_dir, label)  # 构造图像文件夹路径
        for img_file in os.listdir(image_path):  # 遍历图像文件夹中的每个文件
            # 提取样本ID（图像文件名作为样本ID）
            sample_id = f"{label}_{os.path.splitext(img_file)[0]}"
            # 查找与当前图像对应的时序数据行
            sequence_rows = sequence_data[sequence_data["sample_id"] == sample_id]

            # 如果找到了对应的时序数据，进行数据对齐
            if not sequence_rows.empty:
                # 使用真实图像替代随机图像
                img = transform(Image.open(os.path.join(image_path, img_file)))  # 加载并转换图像
                # 提取时序数据的特征部分
                features = sequence_rows.iloc[:, 3:].values.astype(float)  # 提取特征列
                # 设置标签值，良性为0，恶性为1
                label_value = 0 if label == "B" else 1
                # 将图像、时序数据和标签作为元组添加到对齐数据列表中
                aligned_data.append((img, torch.tensor(features), label_value))

    # 划分数据集：训练集70%，验证集10%，测试集20%
    total_size = len(aligned_data)
    train_size = int(0.7 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    # 使用 random_split 将对齐数据划分为训练集、验证集和测试集
    train_data, val_data, test_data = random_split(aligned_data, [train_size, val_size, test_size])

    # 创建 DataLoader，用于批处理加载数据
    def create_loader(data):
        images, sequences, labels = zip(*data)  # 解包对齐数据
        images = torch.stack(images)  # 堆叠图像张量
        sequences = torch.stack(sequences)  # 堆叠时序数据张量
        labels = torch.tensor(labels, dtype=torch.float32)  # 将标签转换为浮点张量
        dataset = TensorDataset(images, sequences, labels)  # 创建TensorDataset
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 返回DataLoader

    return create_loader(train_data), create_loader(val_data), create_loader(test_data)
