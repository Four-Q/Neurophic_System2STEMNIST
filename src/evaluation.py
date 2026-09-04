"""在最佳验证模型上执行一次最终测试，并保留逐样本预测。"""

import time

import torch
from spikingjelly.activation_based import functional
from tqdm.auto import tqdm

from src.function_utils import _autocast_context, _synchronize_cuda


def evaluate_test(
    model,
    test_loader,
    criterion,
    device,
    amp_enabled=False,
    amp_dtype=torch.float16,
    progress_update_interval=20,
):
    device = torch.device(device)
    amp_enabled = bool(amp_enabled and device.type == "cuda")
    was_training = model.training
    model.eval()

    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_samples = 0
    firing_rate_sums = {}
    targets = []
    predictions = []
    confidences = []

    functional.reset_net(model)
    progress_bar = tqdm(
        test_loader,
        desc="Final Test",
        dynamic_ncols=True,
        leave=True,
    )
    _synchronize_cuda(device)
    start_time = time.perf_counter()

    try:
        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(
                progress_bar,
                start=1,
            ):
                inputs = inputs.to(
                    device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                labels = labels.to(
                    device,
                    dtype=torch.long,
                    non_blocking=True,
                )
                # 与原验证函数保持一致，将批次输入转换为多步模式布局。
                inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                batch_size = labels.size(0)
                functional.reset_net(model)

                with _autocast_context(device, amp_enabled, amp_dtype):
                    logits, firing_rates = model(
                        inputs,
                        return_firing_rates=True,
                    )
                    loss = criterion(logits.float(), labels)

                probabilities = logits.float().softmax(dim=1)
                batch_confidences, batch_predictions = probabilities.max(dim=1)

                total_loss += loss.detach().float() * batch_size
                total_correct += (batch_predictions == labels).sum()
                total_samples += batch_size

                targets.extend(labels.detach().cpu().tolist())
                predictions.extend(batch_predictions.detach().cpu().tolist())
                confidences.extend(batch_confidences.detach().cpu().tolist())

                for name, rate in firing_rates.items():
                    if name not in firing_rate_sums:
                        firing_rate_sums[name] = torch.zeros((), device=device)
                    firing_rate_sums[name] += rate.detach().float() * batch_size

                if (
                    batch_index % progress_update_interval == 0
                    or batch_index == len(test_loader)
                ):
                    progress_bar.set_postfix(
                        loss=f"{total_loss.item() / total_samples:.4f}",
                        accuracy=f"{total_correct.item() / total_samples:.4f}",
                    )
    finally:
        functional.reset_net(model)
        model.train(was_training)

    if total_samples == 0:
        raise RuntimeError("测试集没有样本。")

    _synchronize_cuda(device)
    elapsed_seconds = time.perf_counter() - start_time
    return {
        "loss": total_loss.item() / total_samples,
        "accuracy": total_correct.item() / total_samples,
        "correct": total_correct.item(),
        "samples": total_samples,
        "samples_per_second": total_samples / elapsed_seconds,
        "firing_rates": {
            name: value.item() / total_samples
            for name, value in firing_rate_sums.items()
        },
        "targets": targets,
        "predictions": predictions,
        "confidences": confidences,
    }

