from torch.utils.data import DataLoader, Subset

from torchvision.datasets import ImageFolder
from torchvision import transforms

import lightning as L

import os
from typing import Optional
import random as rd

# https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification/data
# All images are [3, 500, 500]
class ImageClassificationDataModule(L.LightningDataModule):

    def __init__(self,
                 train_transform: Optional[transforms.Compose] = None,
                 val_transform: Optional[transforms.Compose] = None, 
                 data_dir: str = 'data/', 
                 batch_size: int = 10, 
                 num_workers: int = os.cpu_count()) -> None:
        
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = train_transform if train_transform is not None else transforms.Compose([
            transforms.TrivialAugmentWide(),
            transforms.ToTensor()
        ])

        self.val_transform = val_transform if val_transform is not None else transforms.Compose([
            transforms.ToTensor()
        ])

    def prepare_data(self) -> None:
        
        dataset_name = 'bean-leaf-lesions-classification'
        username = 'marquis03'
        if os.path.exists(self.data_dir):
            print('Dataset already present')
        else:
            print(f'Downloading dataset from Kaggle: https://www.kaggle.com/datasets/{username}/{dataset_name}')
            os.system(f'kaggle datasets download -d {username}/{dataset_name}')
            os.system(f'unzip -q {dataset_name}.zip')
            os.system(f'rm {dataset_name}.zip')
            os.system(f'mv {dataset_name}/ {self.data_dir}/')

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.training_dataset = ImageFolder(root=self.data_dir + '/train/', transform=self.train_transform)
        self.validation_dataset = ImageFolder(root=self.data_dir + '/val/', transform=self.val_transform)    

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.training_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.validation_dataset, num_workers=self.num_workers, batch_size=self.batch_size)
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.validation_dataset, num_workers=self.num_workers, batch_size=self.batch_size)