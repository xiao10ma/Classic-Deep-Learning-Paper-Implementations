import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from UNet import UNet  # 假设UNet模型定义在UNet.py文件中

# 定义一个虚构的数据集
class DummyDataset(Dataset):
    def __init__(self, size=100, image_size=256):
        self.size = size
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 生成有意义的图像和掩码
        image = torch.rand(3, self.image_size, self.image_size)
        mask = torch.zeros(1, self.image_size, self.image_size)
        # 在图像中添加一些简单的形状
        x, y = torch.randint(0, self.image_size//2, (2,))
        mask[:, x:x+self.image_size//2, y:y+self.image_size//2] = 1
        image[:, x:x+self.image_size//2, y:y+self.image_size//2] += 0.5
        image = torch.clamp(image, 0, 1)
        return image, mask

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据加载器
train_dataset = DummyDataset()
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# 初始化模型
model = UNet(in_channels=3, out_channels=1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 打印一些调试信息
        # print(f"Batch loss: {loss.item():.4f}")
        # print(f"Output mean: {outputs.mean().item():.4f}, std: {outputs.std().item():.4f}")
        # print(f"Mask mean: {masks.mean().item():.4f}, std: {masks.std().item():.4f}")

    return total_loss / len(loader)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    loss = train(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "unet_model.pth")
print("训练完成，模型已保存。")