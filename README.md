# STEMNIST_Ready

本项目只保留论文所需的两组 STEMNIST 对照实验：

- 原始压力帧经 `/255` 缩放后直接输入 `STEMNIST_CSNN`；
- 仿真神经形态系统生成的二值脉冲直接输入同一个 `STEMNIST_CSNN`。

两组实验使用相同的数据划分、模型结构和训练超参数。Pressure 的历史最佳验证准确率参考值为 `64.33%`，Spike 为 `94.20%`。原结果使用快速 GPU 配置而非严格确定性配置，因此更换 GPU、CUDA 或 CuPy 后允许存在小幅数值波动。

## 项目结构

```text
STEMNIST_Ready/
├── data/
│   ├── STEMNIST Dataset.zip
│   └── spike.zip
├── outputs/                         # 初始为空，训练时自动生成
├── src/
│   ├── data/
│   │   ├── prepare_pressure_data.ipynb
│   │   ├── prepare_spike_data.ipynb
│   │   ├── dataset.py
│   │   ├── loader.py
│   │   └── transform.py
│   ├── models/
│   │   └── stemnist_csnn.py         # 唯一模型：STEMNIST_CSNN
│   ├── train/
│   │   ├── pressure_train.ipynb
│   │   └── spike_train.ipynb
│   ├── experiment.py
│   ├── reporting.py
│   └── training.py
└── tests/
    └── smoke_test.py
```

`data/pressure/`、`data/spike/` 和全部 `outputs/` 均为本地派生产物，已由 `.gitignore` 排除。GitHub 中只需提交两个 ZIP 数据文件。

## 推荐环境

用于复现现有结果的参考环境为 Linux、Python 3.12、PyTorch 2.8 和 CUDA 12.8。RTX 5090 环境可先安装 CUDA 版 PyTorch，再安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
jupyter lab
```

如果镜像已预装匹配的 CUDA 版 PyTorch，可直接安装 `requirements.txt`，但应先确认 `torch.cuda.is_available()` 为 `True`。

## 运行顺序

1. 运行 `src/data/prepare_pressure_data.ipynb`，从原始 ZIP 重建固定的 train/val/test 压力数据。
2. 运行 `src/data/prepare_spike_data.ipynb`，安全解压并验证完整脉冲数据。
3. 运行 `src/train/pressure_train.ipynb`。
4. 运行 `src/train/spike_train.ipynb`。

两份训练 Notebook 均固定训练 100 epochs：每轮只使用训练集更新参数，使用验证集选择最佳 checkpoint。全部训练结束后重新载入最佳验证模型，仅对测试集评估一次。测试集不会参与训练、调参或最佳 epoch 选择。

首次运行时保持 `OVERWRITE_OUTPUT = False`。如需覆盖某一已有实验的全部输出，明确将对应 Notebook 中的值改为 `True`。

## 输出内容

每组训练自动生成：

```text
outputs/<pressure|spike>/
├── best_model.pt
├── data/
│   ├── run_config.csv
│   ├── history.csv
│   ├── test_metrics.csv
│   ├── test_predictions.csv
│   ├── confusion_matrix.csv
│   └── per_class_accuracy.csv
└── figure/
    ├── training_history.png
    ├── confusion_matrix.png
    ├── per_class_accuracy.png
    └── training_and_test_summary.png
```

其中 `test_metrics.csv` 保存最终测试准确率和损失；`test_predictions.csv` 保存全部测试样本的真实标签、预测标签、正确性与置信度；混淆矩阵 CSV 保存原始计数，图中采用逐行归一化显示。

## 快速检查

不运行完整训练时，可以执行：

```bash
python tests/smoke_test.py
```

该检查会验证两个 ZIP 的摘要、Notebook 语法、模型类名、参数量和 CPU 前向输出形状。

