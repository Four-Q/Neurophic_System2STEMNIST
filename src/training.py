"""STEMNIST_CSNN 的训练、验证和测试逻辑。"""

from contextlib import nullcontext
import time

import torch
from spikingjelly.activation_based import functional
from tqdm.auto import tqdm


def autocast_context(device, amp_enabled, amp_dtype):
    if amp_enabled and torch.device(device).type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def synchronize_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def prepare_batch(inputs, labels, device):
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
    # DataLoader 为 [N,T,C,H,W]，SpikingJelly 多步模式要求时间维在最前。
    inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
    return inputs, labels


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    scaler,
    amp_enabled,
    amp_dtype,
    progress_update_interval=20,
):
    model.train()
    functional.reset_net(model)

    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_samples = 0

    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    progress = tqdm(
        loader,
        desc=f"Train Epoch {epoch}",
        dynamic_ncols=True,
        leave=True,
    )
    synchronize_cuda(device)
    start_time = time.perf_counter()

    for batch_index, (inputs, labels) in enumerate(progress, start=1):
        inputs, labels = prepare_batch(inputs, labels, device)
        batch_size = labels.size(0)

        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled, amp_dtype):
            logits = model(inputs)
            loss = criterion(logits.float(), labels)

        use_scaler = scaler is not None and scaler.is_enabled()
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # 限制偶发的大梯度；AMP 溢出由 GradScaler 负责跳过和缩放。
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=not use_scaler,
        )

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        predictions = logits.detach().argmax(dim=1)
        total_loss += loss.detach().float() * batch_size
        total_correct += (predictions == labels).sum()
        total_samples += batch_size

        if (
            batch_index % progress_update_interval == 0
            or batch_index == len(loader)
        ):
            progress.set_postfix(
                loss=f"{total_loss.item() / total_samples:.4f}",
                accuracy=f"{total_correct.item() / total_samples:.4f}",
            )

    if total_samples == 0:
        raise RuntimeError("训练集没有样本。")

    synchronize_cuda(device)
    elapsed_seconds = time.perf_counter() - start_time
    peak_memory_gb = 0.0
    if torch.device(device).type == "cuda":
        peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    return {
        "loss": total_loss.item() / total_samples,
        "accuracy": total_correct.item() / total_samples,
        "samples": total_samples,
        "samples_per_second": total_samples / elapsed_seconds,
        "peak_memory_gb": peak_memory_gb,
    }


def evaluate_epoch(
    model,
    loader,
    criterion,
    device,
    split_name,
    amp_enabled,
    amp_dtype,
    epoch=None,
    collect_predictions=False,
    progress_update_interval=20,
):
    was_training = model.training
    model.eval()
    functional.reset_net(model)

    total_loss = torch.zeros((), device=device)
    total_correct = torch.zeros((), device=device, dtype=torch.long)
    total_samples = 0
    firing_rate_sums = {}
    all_targets = []
    all_predictions = []
    all_confidences = []

    description = split_name.title()
    if epoch is not None:
        description = f"{description} Epoch {epoch}"

    progress = tqdm(loader, desc=description, dynamic_ncols=True, leave=True)
    synchronize_cuda(device)
    start_time = time.perf_counter()

    try:
        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(progress, start=1):
                inputs, labels = prepare_batch(inputs, labels, device)
                batch_size = labels.size(0)
                functional.reset_net(model)

                with autocast_context(device, amp_enabled, amp_dtype):
                    logits, firing_rates = model(
                        inputs,
                        return_firing_rates=True,
                    )
                    loss = criterion(logits.float(), labels)

                probabilities = logits.float().softmax(dim=1)
                confidences, predictions = probabilities.max(dim=1)

                total_loss += loss.detach().float() * batch_size
                total_correct += (predictions == labels).sum()
                total_samples += batch_size

                for name, rate in firing_rates.items():
                    if name not in firing_rate_sums:
                        firing_rate_sums[name] = torch.zeros((), device=device)
                    firing_rate_sums[name] += rate.detach().float() * batch_size

                if collect_predictions:
                    all_targets.extend(labels.detach().cpu().tolist())
                    all_predictions.extend(predictions.detach().cpu().tolist())
                    all_confidences.extend(confidences.detach().cpu().tolist())

                if (
                    batch_index % progress_update_interval == 0
                    or batch_index == len(loader)
                ):
                    progress.set_postfix(
                        loss=f"{total_loss.item() / total_samples:.4f}",
                        accuracy=f"{total_correct.item() / total_samples:.4f}",
                    )
    finally:
        functional.reset_net(model)
        model.train(was_training)

    if total_samples == 0:
        raise RuntimeError(f"{split_name} 集没有样本。")

    synchronize_cuda(device)
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
        "targets": all_targets,
        "predictions": all_predictions,
        "confidences": all_confidences,
    }


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs,
    save_path,
    checkpoint_metadata,
    amp_enabled,
    amp_dtype,
    amp_init_scale=1024.0,
    progress_update_interval=20,
):
    device = torch.device(device)
    amp_enabled = bool(amp_enabled and device.type == "cuda")
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=amp_init_scale,
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
        "train_samples_per_second": [],
        "val_samples_per_second": [],
        "peak_memory_gb": [],
        "val_lif1_firing_rate": [],
        "val_lif2_firing_rate": [],
        "val_output_lif_firing_rate": [],
    }

    best_val_accuracy = -1.0
    best_epoch = 0
    functional.reset_net(model)

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            scaler,
            amp_enabled,
            amp_dtype,
            progress_update_interval,
        )
        val_metrics = evaluate_epoch(
            model,
            val_loader,
            criterion,
            device,
            "validation",
            amp_enabled,
            amp_dtype,
            epoch=epoch,
            progress_update_interval=progress_update_interval,
        )
        rates = val_metrics["firing_rates"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["learning_rate"].append(current_lr)
        history["train_samples_per_second"].append(
            train_metrics["samples_per_second"]
        )
        history["val_samples_per_second"].append(
            val_metrics["samples_per_second"]
        )
        history["peak_memory_gb"].append(train_metrics["peak_memory_gb"])
        history["val_lif1_firing_rate"].append(rates["lif1"])
        history["val_lif2_firing_rate"].append(rates["lif2"])
        history["val_output_lif_firing_rate"].append(rates["output_lif"])

        print(
            f"Epoch {epoch:03d}/{num_epochs:03d} | "
            f"Train loss={train_metrics['loss']:.4f} | "
            f"Train acc={train_metrics['accuracy']:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} | "
            f"Val acc={val_metrics['accuracy']:.4f} | LR={current_lr:.6g}"
        )
        print(
            "Validation firing rates | "
            f"lif1={rates['lif1']:.6f} | "
            f"lif2={rates['lif2']:.6f} | "
            f"output={rates['output_lif']:.6f}"
        )

        # 与原实验保持一致：先推进调度器，再保存最佳 checkpoint。
        if scheduler is not None:
            scheduler.step()

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "scaler_state_dict": (
                    scaler.state_dict() if scaler.is_enabled() else None
                ),
                "amp_enabled": amp_enabled,
                "amp_dtype": str(amp_dtype),
                "metadata": dict(checkpoint_metadata),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "history": history,
            }
            torch.save(checkpoint, save_path)
            print(
                f"保存最佳模型：epoch={epoch}, "
                f"val_accuracy={best_val_accuracy:.4f}, path={save_path}"
            )

    checkpoint = torch.load(save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    functional.reset_net(model)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
    }

