from diffusers import UNet2DModel
import torch.nn as nn
from lib.config import Config


class DFUNet(nn.Module):
    def __init__(self, config):
        super(DFUNet, self).__init__()
        self.config = config
        # 参考
        # https://huggingface.co/docs/diffusers/tutorials/basic_training
        self.model = UNet2DModel(
            sample_size=config.img_size,  # 图像大小
            in_channels=config.img_channels,  # 输入通道数
            out_channels=config.img_channels,  # 输出通道数
            layers_per_block=config.layers,
            block_out_channels=(64, 64, 128, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D",
                              "DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D",  "UpBlock2D", "UpBlock2D",
                            "UpBlock2D", "UpBlock2D", "UpBlock2D")
        )

    def forward(self, x, ts):
        return self.model(x, ts)[0]