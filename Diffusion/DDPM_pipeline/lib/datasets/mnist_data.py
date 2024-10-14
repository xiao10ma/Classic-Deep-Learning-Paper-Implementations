import torch
from torchvision import datasets
from torchvision.transforms import v2, InterpolationMode
from torch.utils.data import Dataset
from lib.config import Config
from torchvision.datasets import MNIST

class MNISTData(Dataset):
    def __init__(self,
                 config: 'Config', dataset_dir: str, return_label=False) -> None:
        '''
        ### Args:
            - dataset_dir: 数据及所在的位置，或数据集希望保存的位置
            - return_label: 获取数据是是否返回 label
        '''
        self.config = config
        self.target_size = config.img_size
        self.return_label = return_label
        self.transforms = v2.Compose(
            [v2.Resize((self.target_size, self.target_size), InterpolationMode.BILINEAR),
             v2.ToImageTensor(), 
             v2.ConvertImageDtype(torch.float32),
             v2.Normalize([0.5], [0.5])
             ])
        self.dataset = MNIST(dataset_dir, train=True, transform=self.transforms, download=True)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx: int):
        image, label = self.dataset.__getitem__(index=idx)
        if self.return_label:
            return image.to(self.config.device), torch.tensor(label).to(self.config.device)
        else:
            return image.to(self.config.device)