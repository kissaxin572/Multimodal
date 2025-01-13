import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def save_and_plot_loss(history, result_dir):
    # 保存损失数据到CSV
    loss_csv_file = f"{result_dir}/loss.csv"
    with open(loss_csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
        for epoch, (train_loss, val_loss) in enumerate(zip(history["train_loss"], history["val_loss"])):
            writer.writerow([epoch+1, train_loss, val_loss])

    # 绘制并保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Train Loss", linestyle='-', marker='o')
    plt.plot(history["val_loss"], label="Validation Loss", linestyle='--', marker='x')
    plt.xlabel("Epoch", fontdict={"family": "Times New Roman"})
    plt.ylabel("Loss", fontdict={"family": "Times New Roman"})
    plt.title("Training and Validation Loss", fontdict={"family": "Times New Roman"})
    plt.legend(prop={"family": "Times New Roman"})
    plt.grid(True)
    plt.savefig(f"{result_dir}/loss.png")
    plt.close()

def save_and_plot_metrics(metrics, result_dir):
    # 保存Metrics数据到CSV
    metrics_csv_file = f"{result_dir}/metrics.csv"
    with open(metrics_csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for label in ["accuracy", "precision", "recall", "f1"]:
            writer.writerow([label.capitalize(), metrics[label]])

    # 绘制并保存Metrics柱状图
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    values = [metrics[label.lower()] for label in ["accuracy", "precision", "recall", "f1"]]
    x = np.arange(len(labels))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, values, width=0.4, color='b', alpha=0.7)
    plt.xlabel("Metrics", fontdict={"family": "Times New Roman"})
    plt.ylabel("Scores", fontdict={"family": "Times New Roman"})
    plt.title("Metrics", fontdict={"family": "Times New Roman"})
    plt.xticks(x, labels, fontdict={"family": "Times New Roman"})
    plt.ylim(0, 1)
    plt.grid(axis='y')

    # 在柱状图顶部显示数值
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontdict={"family": "Times New Roman"})

    plt.savefig(f"{result_dir}/metrics.png")
    plt.close()

def save_and_plot_roc(metrics, result_dir):
    # 保存fpr和tpr数据到CSV
    fpr_tpr_csv_file = f"{result_dir}/fpr_tpr.csv"
    with open(fpr_tpr_csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FPR", "TPR"])
        for fpr, tpr in zip(metrics["fpr"], metrics["tpr"]):
            writer.writerow([fpr, tpr])

    # 绘制并保存ROC曲线
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["fpr"], metrics["tpr"], label=f'ROC (AUC = {metrics["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate", fontdict={"family": "Times New Roman"})
    plt.ylabel("True Positive Rate", fontdict={"family": "Times New Roman"})
    plt.title("ROC Curve", fontdict={"family": "Times New Roman"})
    plt.legend(prop={"family": "Times New Roman"})
    plt.grid(True)
    plt.savefig(f"{result_dir}/roc.png")
    plt.close() 