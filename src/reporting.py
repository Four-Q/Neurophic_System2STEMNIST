"""保存实验原始数据，并生成论文复核所需图表。"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


def write_dict_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_history_csv(history, path):
    keys = list(history)
    lengths = {key: len(history[key]) for key in keys}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"训练历史各列长度不一致：{lengths}")

    row_count = lengths[keys[0]]
    has_epoch = "epoch" in history
    fieldnames = keys if has_epoch else ["epoch", *keys]
    rows = []
    for index in range(row_count):
        row = {key: history[key][index] for key in keys}
        if not has_epoch:
            row = {"epoch": index + 1, **row}
        rows.append(row)
    write_dict_rows(path, fieldnames, rows)


def history_epochs(history):
    if "epoch" in history:
        return history["epoch"]
    return list(range(1, len(history["train_loss"]) + 1))


def save_run_config(config, path):
    rows = [{"name": key, "value": value} for key, value in config.items()]
    write_dict_rows(path, ["name", "value"], rows)


def build_confusion_matrix(targets, predictions, class_count):
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(
        confusion,
        (np.asarray(targets, dtype=np.int64), np.asarray(predictions, dtype=np.int64)),
        1,
    )
    return confusion


def save_test_data(
    test_metrics,
    dataset,
    classes,
    data_dir,
    checkpoint_epoch,
    best_val_accuracy,
):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    targets = test_metrics["targets"]
    predictions = test_metrics["predictions"]
    confidences = test_metrics["confidences"]
    if not (len(targets) == len(predictions) == len(confidences) == len(dataset)):
        raise RuntimeError("测试预测数量与测试集不一致。")

    prediction_rows = []
    for index, (target, prediction, confidence) in enumerate(
        zip(targets, predictions, confidences)
    ):
        sample = dataset.sample_info(index)
        if int(sample["label_index"]) != target:
            raise RuntimeError(f"第 {index} 个测试样本标签顺序不一致。")
        prediction_rows.append(
            {
                "row_index": index,
                "sample_id": sample["sample_id"],
                "true_index": target,
                "true_label": classes[target],
                "predicted_index": prediction,
                "predicted_label": classes[prediction],
                "correct": int(target == prediction),
                "confidence": confidence,
            }
        )

    write_dict_rows(
        data_dir / "test_predictions.csv",
        [
            "row_index",
            "sample_id",
            "true_index",
            "true_label",
            "predicted_index",
            "predicted_label",
            "correct",
            "confidence",
        ],
        prediction_rows,
    )

    confusion = build_confusion_matrix(targets, predictions, len(classes))
    confusion_rows = []
    for index, label in enumerate(classes):
        row = {"true_label": label}
        row.update(
            {
                f"pred_{predicted_label}": int(confusion[index, column])
                for column, predicted_label in enumerate(classes)
            }
        )
        confusion_rows.append(row)
    write_dict_rows(
        data_dir / "confusion_matrix.csv",
        ["true_label", *[f"pred_{label}" for label in classes]],
        confusion_rows,
    )

    per_class_rows = []
    for index, label in enumerate(classes):
        sample_count = int(confusion[index].sum())
        correct = int(confusion[index, index])
        accuracy = correct / sample_count if sample_count else 0.0
        per_class_rows.append(
            {
                "label_index": index,
                "label": label,
                "samples": sample_count,
                "correct": correct,
                "accuracy": accuracy,
            }
        )
    write_dict_rows(
        data_dir / "per_class_accuracy.csv",
        ["label_index", "label", "samples", "correct", "accuracy"],
        per_class_rows,
    )

    metrics_row = {
        "split": "test",
        "checkpoint_epoch": checkpoint_epoch,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "correct": test_metrics["correct"],
        "samples": test_metrics["samples"],
        "samples_per_second": test_metrics["samples_per_second"],
        "lif1_firing_rate": test_metrics["firing_rates"]["lif1"],
        "lif2_firing_rate": test_metrics["firing_rates"]["lif2"],
        "output_lif_firing_rate": test_metrics["firing_rates"]["output_lif"],
    }
    write_dict_rows(
        data_dir / "test_metrics.csv",
        list(metrics_row),
        [metrics_row],
    )
    return confusion, per_class_rows, metrics_row


def finish_figure(figure, path, dpi=300):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def plot_training_history(history, path):
    epochs = history_epochs(history)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train", color="tab:red")
    axes[0].plot(epochs, history["val_loss"], label="Validation", color="tab:orange")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        history["train_accuracy"],
        label="Train",
        color="tab:blue",
    )
    axes[1].plot(
        epochs,
        history["val_accuracy"],
        label="Validation",
        color="tab:green",
    )
    best_index = int(np.argmax(history["val_accuracy"]))
    axes[1].scatter(
        epochs[best_index],
        history["val_accuracy"][best_index],
        color="tab:green",
        zorder=5,
    )
    axes[1].annotate(
        f"Best: {history['val_accuracy'][best_index]:.2%}\nEpoch {epochs[best_index]}",
        (epochs[best_index], history["val_accuracy"][best_index]),
        xytext=(-90, -35),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "tab:green"},
    )
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    figure.suptitle("STEMNIST_CSNN Training History")
    figure.tight_layout()
    finish_figure(figure, path)


def plot_confusion_matrix(confusion, classes, path):
    row_totals = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion,
        row_totals,
        out=np.zeros_like(confusion, dtype=np.float64),
        where=row_totals != 0,
    )

    figure, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    ticks = np.arange(len(classes))
    axis.set_xticks(ticks, classes, rotation=90)
    axis.set_yticks(ticks, classes)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Test Confusion Matrix (Row-normalized)")
    figure.colorbar(image, ax=axis, label="Proportion")
    figure.tight_layout()
    finish_figure(figure, path)


def plot_per_class_accuracy(per_class_rows, path):
    labels = [row["label"] for row in per_class_rows]
    accuracies = [row["accuracy"] for row in per_class_rows]

    figure, axis = plt.subplots(figsize=(16, 6))
    bars = axis.bar(labels, accuracies, color="tab:blue")
    axis.bar_label(bars, labels=[f"{value:.0%}" for value in accuracies], fontsize=7)
    axis.set(title="Test Accuracy by Class", xlabel="Class", ylabel="Accuracy")
    axis.set_ylim(0, 1.08)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    figure.tight_layout()
    finish_figure(figure, path)


def plot_training_and_test_summary(history, test_metrics, path):
    epochs = history_epochs(history)
    best_index = int(np.argmax(history["val_accuracy"]))
    best_val_accuracy = history["val_accuracy"][best_index]
    test_accuracy = test_metrics["test_accuracy"]

    figure, axes = plt.subplots(1, 3, figsize=(19, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train", color="tab:red")
    axes[0].plot(epochs, history["val_loss"], label="Validation", color="tab:orange")
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs,
        history["train_accuracy"],
        label="Train",
        color="tab:blue",
    )
    axes[1].plot(
        epochs,
        history["val_accuracy"],
        label="Validation",
        color="tab:green",
    )
    axes[1].axvline(
        epochs[best_index],
        color="tab:green",
        linestyle="--",
        alpha=0.6,
        label=f"Best epoch {epochs[best_index]}",
    )
    axes[1].set(title="Training Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    bars = axes[2].bar(
        ["Best validation", "Final test"],
        [best_val_accuracy, test_accuracy],
        color=["tab:green", "tab:purple"],
        width=0.6,
    )
    axes[2].bar_label(
        bars,
        labels=[f"{best_val_accuracy:.2%}", f"{test_accuracy:.2%}"],
        padding=4,
    )
    axes[2].set(title="Best Model Evaluation", ylabel="Accuracy", ylim=(0, 1.08))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].grid(axis="y", linestyle="--", alpha=0.3)

    figure.suptitle("STEMNIST_CSNN Training and Final Test")
    figure.tight_layout()
    finish_figure(figure, path)
