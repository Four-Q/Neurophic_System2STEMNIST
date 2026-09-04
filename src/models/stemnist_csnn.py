"""用于 STEMNIST 的无归一化多步卷积脉冲神经网络。"""

from numbers import Real

import numpy as np
import torch
from torch import nn

# SpikingJelly 0.0.0.0.14 的 CuPy 内核仍检查 np.int；该别名在
# NumPy 2 中被删除。仅补回等价的 Python int，避免改动第三方包。
if "int" not in np.__dict__:
    setattr(np, "int", int)

from spikingjelly.activation_based import functional, layer, neuron, surrogate


class STEMNIST_CSNN(nn.Module):
    """联合空间发放图与分段时间发放率完成 35 类识别。"""

    EXPECTED_PARAMETER_COUNT = 97_027

    def __init__(
        self,
        num_classes=35,
        dropout=0.1,
        tau=10.0,
        logit_scale=1.0,
        temporal_bins=4,
        readout_features=56,
        backend="torch",
    ):
        super().__init__()

        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes 必须是大于 0 的整数。")
        if not isinstance(dropout, Real) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout 必须位于 [0, 1) 区间。")
        if not isinstance(tau, Real) or isinstance(tau, bool) or tau <= 1.0:
            raise ValueError("tau 必须是大于 1.0 的数值。")
        if not isinstance(logit_scale, Real) or logit_scale <= 0.0:
            raise ValueError("logit_scale 必须大于 0。")
        if not isinstance(temporal_bins, int) or temporal_bins <= 0:
            raise ValueError("temporal_bins 必须是大于 0 的整数。")
        if not isinstance(readout_features, int) or readout_features <= 0:
            raise ValueError("readout_features 必须是大于 0 的整数。")
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend 必须是非空字符串。")

        self.num_classes = num_classes
        self.dropout_probability = float(dropout)
        self.tau = float(tau)
        self.logit_scale = float(logit_scale)
        self.temporal_bins = temporal_bins
        self.readout_feature_count = readout_features
        self.backend = backend

        self.conv1 = layer.Conv2d(
            1,
            16,
            kernel_size=3,
            padding=1,
            bias=True,
            step_mode="m",
        )
        self.lif1 = self._make_lif(v_threshold=0.5)

        self.conv2 = layer.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1,
            bias=True,
            step_mode="m",
        )
        self.lif2 = self._make_lif(v_threshold=0.75)
        self.pool1 = layer.MaxPool2d(kernel_size=2, stride=2, step_mode="m")

        self.conv3 = layer.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1,
            bias=True,
            step_mode="m",
        )
        self.lif3 = self._make_lif(v_threshold=1.0)
        self.pool2 = layer.MaxPool2d(kernel_size=2, stride=2, step_mode="m")

        # 完整 4x4 空间发放图保留字符位置，分段统计保留粗粒度时序。
        summary_features = 64 * 4 * 4 + temporal_bins * 64
        self.readout_hidden = nn.Linear(summary_features, readout_features)
        self.readout_activation = nn.ReLU(inplace=True)
        self.readout_dropout = nn.Dropout(p=self.dropout_probability)
        self.readout_classifier = nn.Linear(readout_features, num_classes)

        self._initialize_weights()
        functional.set_step_mode(self, step_mode="m")
        functional.set_backend(self, backend=backend, instance=neuron.LIFNode)

    def _make_lif(self, v_threshold):
        return neuron.LIFNode(
            tau=self.tau,
            decay_input=False,
            v_threshold=v_threshold,
            surrogate_function=surrogate.ATan(),
            detach_reset=False,
            step_mode="m",
            backend="torch",
        )

    def _initialize_weights(self):
        for convolution in (self.conv1, self.conv2, self.conv3):
            nn.init.kaiming_normal_(
                convolution.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
            nn.init.zeros_(convolution.bias)

        nn.init.kaiming_uniform_(self.readout_hidden.weight, nonlinearity="relu")
        nn.init.zeros_(self.readout_hidden.bias)

        # 小尺度分类层让初始交叉熵接近 log(35)，避免 logits 过早膨胀。
        nn.init.normal_(self.readout_classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.readout_classifier.bias)

    def _validate_inputs(self, inputs):
        if inputs.ndim != 5:
            raise ValueError("输入必须是 [T,N,C,H,W] 五维张量。")
        if inputs.shape[0] < self.temporal_bins:
            raise ValueError(
                f"时间步数 {inputs.shape[0]} 小于 temporal_bins={self.temporal_bins}。"
            )
        if inputs.shape[1] <= 0:
            raise ValueError("batch 维不能为空。")
        if tuple(inputs.shape[2:]) != (1, 16, 16):
            raise ValueError("输入的通道和空间形状必须是 [1,16,16]。")

    def _encode(self, inputs):
        self._validate_inputs(inputs)

        hidden1_spikes = self.lif1(self.conv1(inputs))
        hidden2_spikes = self.lif2(self.conv2(hidden1_spikes))
        hidden2_pooled = self.pool1(hidden2_spikes)
        hidden3_spikes = self.lif3(self.conv3(hidden2_pooled))
        feature_spikes = self.pool2(hidden3_spikes)

        firing_rates = {
            "lif1": hidden1_spikes.detach().float().mean(),
            "lif2": hidden2_spikes.detach().float().mean(),
            "output_lif": hidden3_spikes.detach().float().mean(),
        }
        return feature_spikes, firing_rates

    def _spatiotemporal_summary(self, feature_spikes):
        float_spikes = feature_spikes.float()
        spatial_rates = float_spikes.mean(dim=0).flatten(start_dim=1)
        temporal_rates = torch.cat(
            [
                chunk.mean(dim=(0, 3, 4))
                for chunk in torch.tensor_split(
                    float_spikes,
                    self.temporal_bins,
                    dim=0,
                )
            ],
            dim=1,
        )
        return torch.cat((spatial_rates, temporal_rates), dim=1)

    def _sequence_summary(self, feature_spikes):
        spatial_features = feature_spikes.float().flatten(start_dim=2)
        global_features = feature_spikes.float().mean(dim=(-2, -1))
        repeated_temporal_features = global_features.repeat(
            1,
            1,
            self.temporal_bins,
        )
        return torch.cat((spatial_features, repeated_temporal_features), dim=2)

    def _classify(self, summary):
        hidden = self.readout_hidden(summary)
        hidden = self.readout_activation(hidden)
        hidden = self.readout_dropout(hidden)
        return self.readout_classifier(hidden) * self.logit_scale

    def forward_sequence(self, inputs):
        feature_spikes, firing_rates = self._encode(inputs)
        currents = self._classify(self._sequence_summary(feature_spikes))
        return currents, firing_rates

    def forward(self, inputs, return_firing_rates=False):
        feature_spikes, firing_rates = self._encode(inputs)
        logits = self._classify(self._spatiotemporal_summary(feature_spikes))
        if return_firing_rates:
            return logits, firing_rates
        return logits

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def extra_repr(self):
        return (
            f"num_classes={self.num_classes}, "
            f"dropout={self.dropout_probability}, "
            f"tau={self.tau}, logit_scale={self.logit_scale}, "
            f"temporal_bins={self.temporal_bins}, "
            f"readout_features={self.readout_feature_count}, "
            f"backend={self.backend!r}, normalization='none', "
            "readout='spatiotemporal_rate'"
        )
