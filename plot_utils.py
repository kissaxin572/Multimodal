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
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(history["train_loss"], label="Train Loss", linestyle='-', marker='o')
    plt.plot(history["val_loss"], label="Validation Loss", linestyle='--', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
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
    plt.rcParams['font.family'] = 'Times New Roman'
    bars = plt.bar(x, values, width=0.4, color='b', alpha=0.7)
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.title("Metrics")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.grid(axis='y')

    # 在柱状图顶部显示数值
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.2f}', ha='center', va='bottom', fontsize=10)

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
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.plot(metrics["fpr"], metrics["tpr"], label=f'ROC (AUC = {metrics["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{result_dir}/roc.png")
    plt.close()

def compare_results(results, result_dir):
    os.makedirs(result_dir, exist_ok=True)
    plt.rcParams['font.family'] = 'Times New Roman'
    font_dict = {"size": 14}
    title_font_dict = {"size": 16}

    # 使用论文风格的调色板
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf"
    ]

    # 1. Loss Comparison (CSV + PNG)
    loss_csv_path = os.path.join(result_dir, "loss.csv")
    with open(loss_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Model", "Train Loss", "Validation Loss"])
        for image_type, data in results.items():
            for epoch in range(len(data["history"]["train_loss"])):
                train_loss = data["history"]["train_loss"][epoch]
                val_loss = data["history"]["val_loss"][epoch]
                writer.writerow([epoch + 1, image_type, train_loss, val_loss])

    # 绘制 Loss 图
    plt.figure(figsize=(10, 6))
    for i, (image_type, data) in enumerate(results.items()):
        plt.plot(data["history"]["train_loss"], label=f"{image_type} - Train",
                 linestyle='-', color=colors[i % len(colors)], linewidth=2)
        plt.plot(data["history"]["val_loss"], label=f"{image_type} - Validation",
                 linestyle='--', color=colors[i % len(colors)], linewidth=2, alpha=0.8)
    plt.xlabel("Epoch", fontdict=font_dict)
    plt.ylabel("Loss", fontdict=font_dict)
    plt.title("Training and Validation Loss", fontdict=title_font_dict)
    plt.legend(loc='upper right', prop={"size": 12}, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Metrics Comparison (CSV + PNG)
    metrics_csv_path = os.path.join(result_dir, "metrics.csv")
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "F1"])
        for image_type, data in results.items():
            writer.writerow([image_type, data["metrics"]["accuracy"], data["metrics"]["precision"],
                             data["metrics"]["recall"], data["metrics"]["f1"]])

    plt.figure(figsize=(10, 6))
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metrics))
    bar_width = 0.15
    for i, (image_type, data) in enumerate(results.items()):
        values = [data["metrics"][metric.lower()] for metric in metrics]
        plt.bar(x + i * bar_width, values, bar_width, label=image_type, color=colors[i % len(colors)], alpha=0.9)
    plt.xlabel("Metrics", fontdict=font_dict)
    plt.ylabel("Scores", fontdict=font_dict)
    plt.title("Metrics Comparison", fontdict=title_font_dict)
    plt.xticks(x + bar_width * (len(results) - 1) / 2, ["Accuracy", "Precision", "Recall", "F1-score"], fontsize=12)
    plt.ylim(0, 1)
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), prop={"size": 12}, frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. ROC Comparison (CSV + PNG)
    roc_csv_path = os.path.join(result_dir, "fpr_tpr.csv")
    with open(roc_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "FPR", "TPR"])
        for image_type, data in results.items():
            for fpr, tpr in zip(data["metrics"]["fpr"], data["metrics"]["tpr"]):
                writer.writerow([image_type, fpr, tpr])

    plt.figure(figsize=(10, 6))
    for i, (image_type, data) in enumerate(results.items()):
        plt.plot(data["metrics"]["fpr"], data["metrics"]["tpr"],
                 label=f"{image_type} (AUC={data['metrics']['auc']:.2f})",
                 color=colors[i % len(colors)], linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random", linewidth=1.5)
    plt.xlabel("False Positive Rate", fontdict=font_dict)
    plt.ylabel("True Positive Rate", fontdict=font_dict)
    plt.title("ROC Curve Comparison", fontdict=title_font_dict)
    plt.legend(loc='lower right', prop={"size": 12}, frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "roc.png"), dpi=300, bbox_inches='tight')
    plt.close()


