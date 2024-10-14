import torch
from lib.datasets import MNISTData
from lib.config import Config
from lib.networks import DFUNet, DDPMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

config = Config('configs/ddpm_config.yaml')

# 创建输出目录
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# 初始化 TensorBoard writer
writer = SummaryWriter(os.path.join(output_dir, 'tensorboard_logs'))

training_data = MNISTData(config, r"data", return_label=True)
train_dataloader = DataLoader(training_data, batch_size=config.batch_size, shuffle=True)
model = DFUNet(config).to(config.device)
scheduler = DDPMScheduler(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

log_interval = 100  # 每100步记录一次损失

for ep in range(config.epochs):
    progress_bar = tqdm(total=len(train_dataloader))
    model.train()
    epoch_loss = 0.0
    for step, (image, _) in enumerate(train_dataloader):
        image = image.to(config.device)
        batch = image.shape[0]
        timesteps = scheduler.sample_timesteps(batch)
        noise = torch.randn(image.shape).to(config.device)
        noisy_image = scheduler.add_noise(image, noise, timesteps)

        pred = model(noisy_image, timesteps)
        loss = torch.nn.functional.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

        # 每 log_interval 步记录一次损失
        if (step + 1) % log_interval == 0:
            avg_loss = epoch_loss / (step + 1)
            writer.add_scalar('训练损失', avg_loss, ep * len(train_dataloader) + step)

        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "ep": ep + 1}
        progress_bar.set_postfix(**logs)

    # 每个 epoch 结束后记录平均 loss
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    writer.add_scalar('每轮平均损失', avg_epoch_loss, ep)

    print(f"Epoch {ep+1}/{config.epochs}, 平均损失: {avg_epoch_loss:.4f}")

    # 根据 config.interval 保存模型
    if (ep + 1) % config.save_interval == 0:
        checkpoint = {
            'epoch': ep + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }
        torch.save(checkpoint, os.path.join(output_dir, f'model_epoch_{ep+1}.pth'))
        print(f"模型已保存：epoch_{ep+1}")

# 关闭 TensorBoard writer
writer.close()

# 保存最终模型
final_checkpoint = {
    'epoch': config.epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_epoch_loss,
}
torch.save(final_checkpoint, os.path.join(output_dir, 'final_model.pth'))
print("最终模型已保存")
