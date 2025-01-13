from data_loader import load_aligned_data  # 导入数据加载模块
from models import MultimodalModel  # 导入模型模块
from train import Trainer  # 导入训练模块
import torch
import os

# 主程序入口
if __name__ == "__main__":
    sample_time = "10s"

    image_dir = f"Datasets/Processed/{sample_time}/SFC/Gray"
    # 检查并创建图像目录
    if not os.path.exists(image_dir):
        print(f"创建图像目录: {image_dir}")
        os.makedirs(image_dir, exist_ok=True)

    sample_interval = "100ms" if sample_time == "10s" else "200ms" if sample_time == "20s" else "300ms"
    time_series_file = f"Datasets/Processed/{sample_time}/{sample_time}_{sample_interval}.csv"
    # 检查时序数据文件是否存在
    if not os.path.exists(time_series_file):
        raise FileNotFoundError(f"时序数据文件不存在: {time_series_file}")

    img_size = 256  # 图像尺寸
    batch_size = 32  # 批次大小

    # 加载对齐的数据
    train_loader, val_loader, test_loader = load_aligned_data(image_dir, time_series_file, img_size, batch_size)

    # 创建多模态模型
    hidden_size = 64  # LSTM 隐藏层大小
    input_size = 8  # 时序数据的特征维度（根据你的数据调整）
    model = MultimodalModel(input_size, hidden_size)

    # 设置设备（使用 GPU 或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建Trainer实例
    log_path = f"Results/{sample_time}/train.log"
    trainer = Trainer(log_path)

    # 定义损失函数、优化器和学习率调度器
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练模型
    history = trainer.train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, sample_time)

    # 评估模型
    metrics = trainer.evaluate_model(model, test_loader, device)

    # 保存结果
    trainer.save_results(metrics, history, sample_time)
