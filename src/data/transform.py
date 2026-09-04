# src/data/transform.py

import numpy as np
import torch


class PressureToTensor:
    """将原始 uint8 压力帧转换成模型可以使用的 Tensor。"""

    def __call__(self, frames):
        expected_shape = (240, 16, 16)
        if tuple(frames.shape) != expected_shape:
            raise ValueError(
                f"压力帧形状应为 {expected_shape}，实际为 {tuple(frames.shape)}"
            )

        # memmap 通常只读，复制后再交给 PyTorch 可避免未定义写入行为。
        array = np.array(frames, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(array).to(torch.float32)
        tensor.div_(255.0)
        return tensor.unsqueeze(1)


def build_pressure_transform():
    """仅执行当前基线实验使用的 /255 缩放，不做标准化。"""

    return PressureToTensor()
