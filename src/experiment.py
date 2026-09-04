"""以固定配置运行 Pressure 或 Spike 的完整训练与最终测试。"""

import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
from torch import nn

from src.data.loader import LoaderConfig, create_loaders
from src.data.transform import build_pressure_transform
from src.models import STEMNIST_CSNN
from src.reporting import (
    plot_confusion_matrix,
    plot_per_class_accuracy,
    plot_training_and_test_summary,
    plot_training_history,
    save_history_csv,
    save_run_config,
    save_test_data,
)
from src.training import evaluate_epoch, train_model


REFERENCE_VAL_ACCURACY = {
    "pressure": 0.6432900432900432,
    "spike": 0.941991341991342,
}


def find_project_root(start=None):
    override = os.environ.get("STEMNIST_PROJECT_ROOT")
    current = Path(override).expanduser() if override else Path(start or Path.cwd())
    current = current.resolve()

    for candidate in (current, *current.parents):
        if (
            (candidate / "data" / "STEMNIST Dataset.zip").is_file()
            and (candidate / "data" / "spike.zip").is_file()
            and (candidate / "src" / "models" / "stemnist_csnn.py").is_file()
        ):
            return candidate

    raise FileNotFoundError(
        "未找到 STEMNIST_Ready 项目根目录。请从项目内运行 Notebook，"
        "或设置 STEMNIST_PROJECT_ROOT。"
    )


def configure_randomness(seed, reproducible):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 保留产生当前 v4 结果的快速配置；不同 GPU/软件环境可能有轻微浮动。
    torch.use_deterministic_algorithms(reproducible)
    torch.backends.cudnn.deterministic = reproducible
    torch.backends.cudnn.benchmark = not reproducible
    torch.set_float32_matmul_precision("highest" if reproducible else "high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = not reproducible
        torch.backends.cudnn.allow_tf32 = not reproducible


def prepare_output_directory(project_root, experiment_name, overwrite):
    output_root = (Path(project_root) / "outputs").resolve()
    experiment_dir = (output_root / experiment_name).resolve()

    # 删除只允许命中 outputs 下的 pressure 或 spike，避免配置错误扩大范围。
    if experiment_dir.parent != output_root or experiment_dir.name not in {
        "pressure",
        "spike",
    }:
        raise ValueError(f"实验输出路径不安全：{experiment_dir}")

    if experiment_dir.exists() and any(experiment_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"实验目录已有内容：{experiment_dir}。"
                "确认需要重跑时，在 Notebook 中设置 OVERWRITE_OUTPUT=True。"
            )
        shutil.rmtree(experiment_dir)

    data_dir = experiment_dir / "data"
    figure_dir = experiment_dir / "figure"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir, data_dir, figure_dir


def build_run_config(data_kind, device, backend, amp_enabled, num_workers):
    return {
        "experiment_name": data_kind,
        "data_kind": data_kind,
        "input_transform": "/255" if data_kind == "pressure" else "binary spike",
        "model_class": "STEMNIST_CSNN",
        "parameter_count": STEMNIST_CSNN.EXPECTED_PARAMETER_COUNT,
        "num_classes": 35,
        "time_steps": 240,
        "batch_size": 64,
        "num_workers": num_workers,
        "prefetch_factor": 4,
        "dropout": 0.1,
        "tau": 10.0,
        "temporal_bins": 4,
        "readout_features": 56,
        "learning_rate": 0.005,
        "minimum_learning_rate": 1e-5,
        "warmup_epochs": 5,
        "epochs": 100,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "loss": "CrossEntropyLoss",
        "seed": 42,
        "reproducible": False,
        "device": str(device),
        "snn_backend": backend,
        "amp_enabled": amp_enabled,
        "amp_dtype": "torch.float16",
        "amp_init_scale": 1024.0,
        "reference_best_val_accuracy": REFERENCE_VAL_ACCURACY[data_kind],
    }


def run_experiment(data_kind, project_root=None, overwrite_output=False):
    if data_kind not in {"pressure", "spike"}:
        raise ValueError("data_kind 只能是 'pressure' 或 'spike'。")

    project_root = Path(project_root or find_project_root()).resolve()
    prepared_root = project_root / "data" / data_kind
    if not prepared_root.is_dir():
        notebook_name = (
            "prepare_pressure_data.ipynb"
            if data_kind == "pressure"
            else "prepare_spike_data.ipynb"
        )
        raise FileNotFoundError(
            f"缺少 {prepared_root}。请先运行 src/data/{notebook_name}。"
        )

    seed = 42
    reproducible = False
    configure_randomness(seed, reproducible)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.float16
    backend = "cupy" if device.type == "cuda" else "torch"
    num_workers = min(8, os.cpu_count() or 1)

    experiment_dir, data_dir, figure_dir = prepare_output_directory(
        project_root,
        data_kind,
        overwrite_output,
    )
    best_model_path = experiment_dir / "best_model.pt"

    loader_config = LoaderConfig(
        batch_size=64,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        seed=seed,
        prefetch_factor=4,
        in_memory=device.type == "cuda",
    )
    pressure_transform = build_pressure_transform() if data_kind == "pressure" else None
    loaders = create_loaders(
        data_root=project_root / "data",
        data_kind=data_kind,
        train_transform=pressure_transform,
        eval_transform=pressure_transform,
        config=loader_config,
    )

    classes = loaders["test"].dataset.classes
    if classes != loaders["train"].dataset.classes:
        raise RuntimeError("训练集与测试集类别顺序不一致。")

    model = STEMNIST_CSNN(
        num_classes=35,
        dropout=0.1,
        tau=10.0,
        temporal_bins=4,
        readout_features=56,
        backend=backend,
    ).to(device)
    if model.parameter_count() != model.EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"模型参数量应为 {model.EXPECTED_PARAMETER_COUNT}，"
            f"实际为 {model.parameter_count()}。"
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.005,
        weight_decay=1e-4,
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=5,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=95,
        eta_min=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[5],
    )

    run_config = build_run_config(
        data_kind,
        device,
        backend,
        amp_enabled,
        num_workers,
    )
    save_run_config(run_config, data_dir / "run_config.csv")

    print(f"项目目录：{project_root}")
    print(f"实验：{data_kind}")
    print(f"设备：{device}，SNN backend：{backend}，AMP：{amp_enabled}")
    print(f"输出目录：{experiment_dir}")

    train_result = train_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=100,
        save_path=best_model_path,
        checkpoint_metadata=run_config,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        amp_init_scale=1024.0,
        progress_update_interval=20,
    )
    history = train_result["history"]
    save_history_csv(history, data_dir / "history.csv")

    # train_model 已重新载入最佳验证模型；测试集不会参与训练或模型选择。
    test_metrics = evaluate_epoch(
        model=model,
        loader=loaders["test"],
        criterion=criterion,
        device=device,
        split_name="test",
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        collect_predictions=True,
        progress_update_interval=20,
    )
    confusion, per_class_rows, metrics_row = save_test_data(
        test_metrics,
        loaders["test"].dataset,
        classes,
        data_dir,
        train_result["best_epoch"],
        train_result["best_val_accuracy"],
    )

    plot_training_history(history, figure_dir / "training_history.png")
    plot_confusion_matrix(confusion, classes, figure_dir / "confusion_matrix.png")
    plot_per_class_accuracy(
        per_class_rows,
        figure_dir / "per_class_accuracy.png",
    )
    plot_training_and_test_summary(
        history,
        metrics_row,
        figure_dir / "training_and_test_summary.png",
    )

    reference = REFERENCE_VAL_ACCURACY[data_kind]
    difference = train_result["best_val_accuracy"] - reference
    print("\n训练与最终测试完成。")
    print(f"最佳验证准确率：{train_result['best_val_accuracy']:.2%}")
    print(f"当前仓库参考值：{reference:.2%}，差值：{difference:+.2%}")
    print(f"最终测试准确率：{metrics_row['test_accuracy']:.2%}")
    print(f"结果目录：{experiment_dir}")

    return {
        "experiment_dir": experiment_dir,
        "best_model_path": best_model_path,
        "best_epoch": train_result["best_epoch"],
        "best_val_accuracy": train_result["best_val_accuracy"],
        "test_accuracy": metrics_row["test_accuracy"],
        "data_dir": data_dir,
        "figure_dir": figure_dir,
    }

