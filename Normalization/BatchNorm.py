import torch
import torch.nn as nn

class Batch(nn.Module):
    def __init__(self, channels: int, training: bool = True,
                 eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super(Batch, self).__init__()
        self.channels = channels
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.training = training

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
                self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
                self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var  # 修正了变量名 self.var -> self.exp_var
        else:
            mean = self.exp_mean
            var = self.exp_var

        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var + self.eps).view(1, -1, 1)
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        return x_norm.view(x_shape)

def main():
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    
    # 定义输入参数
    batch_size = 4
    channels = 3
    height = 8
    width = 8
    
    # 创建 Batch 实例
    bn = Batch(channels=channels)
    
    # 创建随机输入张量
    x = torch.randn(batch_size, channels, height, width)
    
    print("输入张量形状:", x.shape)
    print("输入张量均值:", x.mean().item())
    print("输入张量方差:", x.var().item())
    
    # 训练模式测试
    bn.training = True
    output_train = bn.forward(x)
    
    print("\n训练模式输出:")
    print("输出张量形状:", output_train.shape)
    print("输出张量均值:", output_train.mean().item())
    print("输出张量方差:", output_train.var().item())
    
    # 评估模式测试
    bn.training = False
    output_eval = bn.forward(x)
    
    print("\n评估模式输出:")
    print("输出张量形状:", output_eval.shape)
    print("输出张量均值:", output_eval.mean().item())
    print("输出张量方差:", output_eval.var().item())
    
    # 检查运行时统计信息
    if bn.track_running_stats:
        print("\n运行时统计信息:")
        print("指数移动平均均值:", bn.exp_mean)
        print("指数移动平均方差:", bn.exp_var)

if __name__ == "__main__":
    main()
