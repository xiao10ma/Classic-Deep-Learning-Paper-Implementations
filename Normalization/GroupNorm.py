import torch
import torch.nn as nn

class GroupNorm:
    def __init__(self, groups: int, channels: int,
                 eps: float = 1e-5, affine: bool = True):
        assert channels % groups == 0, "Number of channels should be evenly divisible by the number of groups"
        self.groups = groups
        self.channels = channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor):
        x_shape = x.shape
        batch_size = x_shape[0]
        assert self.channels == x_shape[1]

        # [batch_size, group, (channels // group) * H * W]
        x = x.view(batch_size, self.groups, -1)
        group_mean = x.mean(dim=-1, keepdim=True)
        group_mean2 = (x ** 2).mean(-1, keepdim=True)
        group_var = group_mean2 - group_mean ** 2

        x_norm = (x - group_mean) / torch.sqrt(self.eps + group_var)

        if self.affine:
            x_norm = x_norm.view(batch_size, self.channels, -1)
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)
        
        return x_norm.view(x_shape)

def main():
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    
    # 定义输入参数
    batch_size = 2
    channels = 6
    height = 4
    width = 4
    groups = 2
    
    # 创建 GroupNorm 实例
    gn = GroupNorm(groups=groups, channels=channels)
    
    # 创建随机输入张量
    x = torch.randn(batch_size, channels, height, width)
    
    print("输入张量形状:", x.shape)
    print("输入张量:")
    print(x)
    
    # 应用组归一化
    output = gn.forward(x)
    
    print("\n输出张量形状:", output.shape)
    print("输出张量:")
    print(output)
    
    # 验证输出的均值和方差
    output_reshaped = output.view(batch_size, groups, -1)
    mean = output_reshaped.mean(dim=-1)
    var = output_reshaped.var(dim=-1)
    
    print("\n每个组的均值:")
    print(mean)
    print("\n每个组的方差:")
    print(var)

if __name__ == "__main__":
    main()
