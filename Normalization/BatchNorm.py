import torch
import torch.nn as nn

class Batch:
    def __init__(self, channels: int,
                 eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        self.channels = channels
        self.esp = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))

        if self.track_running_stats:
            self.register_buffer('exp_mean', torch.zeros(channels))
            self.register_buffer('exp_var', torch.ones(channels))

    def forward(self, x: torch.Tensor):
        x_shape = x.shape
        batch_size = x.shape[0]
        assert self.channels == x.shape[1]
        x = x.view(batch_size, self.channels, -1)

        if self.training or not self.track_running_stats:
            mean = x.mean(dim=[0, 2])
            mean_x2 = (x ** 2).mean(dim=[0, 2])
            var = mean_x2 - mean ** 2
            if self.training and self.track_running_stats:
                # Momentum动量，"指数移动平均"（Exponential Moving Average，EMA）
                # 平滑，减少噪声
                self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
                self.exp_var = (1 - self.momentum) * self.var + self.momentum * var
        else:
            mean = self.exp_mean
            var = self.exp_var

        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var + self.eps).view(1, -1, 1)
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        return x_norm.view(x_shape)