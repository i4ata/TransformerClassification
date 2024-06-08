from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from typing import Tuple
import os

# https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification/data
# All images are [3, 500, 500]
class LeavesData:

    def __init__(self) -> None:
        super().__init__()
        dataset_name = 'bean-leaf-lesions-classification'
        username = 'marquis03'
        if os.path.exists('data/'):
            print('Dataset already present')
        else:
            print(f'Downloading dataset from Kaggle: https://www.kaggle.com/datasets/{username}/{dataset_name}')
            os.system(f'kaggle datasets download -d {username}/{dataset_name}')
            os.system(f'unzip -q {dataset_name}.zip')
            os.system(f'rm {dataset_name}.zip')
            os.system(f'mv {dataset_name}/ data/')

        self.training_dataset = ImageFolder(root='data/train/')
        self.validation_dataset = ImageFolder(root='data/val/')
        
    def get_dataloaders(self, batch_size: int = 10) -> Tuple[DataLoader, DataLoader]:
        training_dataloader = DataLoader(dataset=self.training_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(dataset=self.validation_dataset, batch_size=batch_size)
        return training_dataloader, validation_dataloader
