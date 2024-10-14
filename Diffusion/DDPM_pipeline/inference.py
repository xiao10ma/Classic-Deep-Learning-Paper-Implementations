import os
import torch
from lib.config import Config
from lib.networks import DFUNet, DDPMScheduler
from torchvision.utils import save_image

if __name__ == "__main__":
    config = Config('configs/ddpm_config.yaml')
    model = DFUNet(config).to(config.device)
    model.eval()

    scheduler = DDPMScheduler(config)
    timesteps = scheduler.set_timesteps()  # 获取时间步

    output_dir = 'output'
    chkpnt_path = os.path.join(output_dir, 'final_model.pth')

    # 读取模型
    checkpoint = torch.load(chkpnt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    pure_noise = torch.randn((1, config.img_channels, config.img_size, config.img_size)).to(config.device)
    
    # 遍历时间步
    for t in timesteps:
        with torch.no_grad():
            noise_pred = model(pure_noise, t)
        
        # 使用scheduler.step来更新图像
        pure_noise = scheduler.step(noise_pred, t, pure_noise)

    # 保存最终生成的图像
    generated_image = pure_noise.clamp(-1, 1)  # 确保像素值在 [-1, 1] 范围内
    generated_image = (generated_image + 1) / 2  # 将范围从 [-1, 1] 转换为 [0, 1]
    
    save_path = os.path.join(output_dir, 'generated_image.png')
    save_image(generated_image, save_path)
    
    print(f"生成的图像已保存到: {save_path}")
