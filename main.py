from data_loader import load_aligned_data
from models import MultimodalModel
from train import Trainer
import torch
import os
import plot_utils

if __name__ == "__main__":
    sample_time = "30s"
    # 图像类型和路径映射字典
    image_type_to_path = {
        "Byteplot": f"Datasets/Processed/{sample_time}/Byteplot",
        "Markov": f"Datasets/Processed/{sample_time}/Markov",
        "Simhash": f"Datasets/Processed/{sample_time}/Simhash",
        "SFC_Gray": f"Datasets/Processed/{sample_time}/SFC/Gray",
        "SFC_Hilbert": f"Datasets/Processed/{sample_time}/SFC/Hilbert",
        "SFC_Zorder": f"Datasets/Processed/{sample_time}/SFC/Zorder"
    }

    # 根据采样时间设置对应的采样间隔
    sample_interval_map = {
        "10s": "100ms",
        "20s": "200ms", 
        "30s": "300ms"
    }

    results = {}

    for image_type, image_dir in image_type_to_path.items():
        print(f"Processing image type: {image_type}")
        
        # 检查图像目录
        if not os.path.exists(image_dir):
            print(f"Image directory not found: {image_dir}")
            continue
        
        sample_interval = sample_interval_map[sample_time]
        time_series_file = f"Datasets/Processed/{sample_time}/{sample_time}_{sample_interval}.csv"
        
        if not os.path.exists(time_series_file):
            print(f"Time series file not found: {time_series_file}")
            continue
        
        train_loader, val_loader, test_loader = load_aligned_data(image_dir, time_series_file, img_size=256, batch_size=32)
        
        # 初始化模型
        model = MultimodalModel(input_size=8, hidden_size=64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        log_path = f"Results/{sample_time}/{image_type}/train.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        trainer = Trainer(log_path)
        
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        # 训练模型
        history = trainer.train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, f"{sample_time}/{image_type}")
        
        # 评估模型
        metrics = trainer.evaluate_model(model, test_loader, device)
        
        # 保存结果
        trainer.save_results(metrics, history, f"{sample_time}/{image_type}")
        results[image_type] = {"metrics": metrics, "history": history}

    # 对比绘图
    plot_utils.compare_results(results, f"Results/{sample_time}/comparison")
