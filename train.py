import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import logging
import os
import sys
from tqdm import tqdm
from plot_utils import save_and_plot_loss, save_and_plot_metrics, save_and_plot_roc

class LoggerStreamHandler:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

class Trainer:
    def __init__(self, log_path):
        # 设置日志记录到文件和屏幕
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(log_path, mode="w")
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # 重定向标准输出和标准错误流到日志系统
        sys.stdout = LoggerStreamHandler(self.logger, logging.INFO)
        sys.stderr = LoggerStreamHandler(self.logger, logging.INFO)

    def train_model(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, sample_time, epochs=10):
        model.to(device)

        # 记录模型信息
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model Architecture: {model.__class__.__name__}")
        self.logger.info(f"Total Trainable Parameters: {total_params}")

        history = {"train_loss": [], "val_loss": []}

        best_val_loss = float("inf")
        best_model_weights = None

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for img_batch, seq_batch, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                img_batch, seq_batch, labels = img_batch.to(device), seq_batch.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(img_batch, seq_batch).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img_batch, seq_batch, labels in val_loader:
                    img_batch, seq_batch, labels = img_batch.to(device), seq_batch.to(device), labels.to(device)
                    outputs = model(img_batch, seq_batch).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict()
                self.logger.info(f"Epoch {epoch+1}/{epochs} - New best model found. Saving weights.")

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Learning Rate: {current_lr:.6f}")

        # 保存最佳模型
        if best_model_weights is not None:
            best_model_path = f"Results/{sample_time}/best_model.pth"
            torch.save(best_model_weights, best_model_path)
            self.logger.info(f"Best model weights saved as {best_model_path}")

        return history

    def evaluate_model(self, model, test_loader, device):
        model.eval()
        y_true = []
        y_pred = []
        y_scores = []

        with torch.no_grad():
            for img_batch, seq_batch, labels in test_loader:
                img_batch, seq_batch, labels = img_batch.to(device), seq_batch.to(device), labels.to(device)
                outputs = model(img_batch, seq_batch).squeeze()
                scores = torch.sigmoid(outputs)
                preds = (scores > 0.5).float()

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_scores.extend(scores.cpu().numpy())

        # 计算评估指标
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)

        self.logger.info(f"Test Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc_score:.4f}")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc_score
        }

    def save_results(self, metrics, history, sample_time):
        result_dir = f"Results/{sample_time}"
        os.makedirs(result_dir, exist_ok=True)

        # 保存训练历史和绘制损失曲线
        save_and_plot_loss(history, result_dir)

        # 保存评估指标和绘制Metrics柱状图
        save_and_plot_metrics(metrics, result_dir)

        # 绘制并保存ROC曲线
        save_and_plot_roc(metrics, result_dir)
