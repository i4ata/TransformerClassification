import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import models, transforms
from torchmetrics import F1Score
from torchinfo import summary
from PIL import Image
from typing import Tuple
import argparse

from src.early_stopper import EarlyStopper
from src.dataset import LeavesData
from src.transforms import model_transforms
from src.custom_transformer.vit import ViT

class Trainer:

    def __init__(self, 
                 model: nn.Module, 
                 name: str = 'default_name', 
                 device: str = 'cpu', 
                 transform: transforms.Compose = transforms.ToTensor()) -> None:
        self.model = model.to(device)
        self.name = name
        self.device = device
        self.val_transform = transform
        with open('data/classname.txt') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
    def _train_step(self, train_dataloader: DataLoader) -> Tuple[float, float]:
        train_loss, train_f1 = 0, 0
        self.model.train()
        for images, labels in train_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            predictions = self.model(images)
            loss = F.cross_entropy(input=predictions, target=labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss
            train_f1 += self.f1_score(predictions, labels)
        return train_loss / len(train_dataloader), train_f1 / len(train_dataloader)
    
    def _val_step(self, val_dataloader: DataLoader) -> Tuple[float, float]:
        val_loss, val_f1 = 0, 0
        self.model.eval()
        for images, labels in val_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.inference_mode():
                predictions = self.model(images)
            loss = F.cross_entropy(input=predictions, target=labels)
            val_loss += loss
            val_f1 += self.f1_score(predictions, labels)
        return val_loss / len(val_dataloader), val_f1 / len(val_dataloader)

    def fit(self, 
            train_transform: transforms.Compose = transforms.ToTensor(),
            batch_size: int = 10,
            epochs: int = 10, 
            learning_rate: float = .001) -> None:
        data = LeavesData()
        data.training_dataset.transforms = train_transform
        data.validation_dataset.transforms = self.val_transform
        train_dataloader, val_dataloader = data.get_dataloaders(batch_size=batch_size)
        self.optimizer = Adam(params=self.model.parameters(), lr=learning_rate)
        self.f1_score = F1Score(task='multiclass', num_classes=len(data.training_dataset.classes))
        early_stopper = EarlyStopper()
        for epoch in range(epochs):
            train_loss, train_f1 = self._train_step(train_dataloader)
            val_loss, val_f1 = self._val_step(val_dataloader)
            print(f'{epoch} | Train loss: {train_loss} | Train F1: {train_f1} | Val loss: {val_loss} | Val F1: {val_f1}', flush=True)
            if early_stopper.check():
                print('Training stops early due to overfitting suspicion')
                break
            if early_stopper.save_model: torch.save(self.model.state_dict(), 'models/' + self.name + '.pt')
        else:
            print('Model might not have converged')

    def print_summary(self):
        print(summary(self.model, input_size=(16, 3, 224, 224)))

    def predict(self, image_path: str) -> Tuple[str, float]:
        image_tensor = self.val_transform(img=Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            prediction: torch.Tensor = torch.softmax(self.model(image_tensor).detach().cpu()[0], dim=0)
        return self.classes[prediction.argmax()], prediction.max().item() 
        

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--use_custom',     type=bool,  default=False,           help='Whether to use the custom implementation or the pretrained one')
    parser.add_argument('--epochs',         type=int,   default=10,              help='Number of epochs to train the model for')
    parser.add_argument('--learning_rate',  type=float, default=1e-3,            help='Model learning rate')
    parser.add_argument('--name',           type=str,   default='default_name',  help='Experiment name')
    parser.add_argument('--batch_size',     type=int,   default=10,              help='Dataloader batch size')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    if args.use_custom:
        model = ViT()
        augmentations = model_transforms['custom']
    else:
        model = models.vit_b_16(weights='DEFAULT')
        for parameter in model.parameters(): parameter.requires_grad = False
        model.heads = nn.Linear(in_features=768, out_features=3)
        augmentations = model_transforms['pretrained']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model=model, name=args.name, device=device, transform=augmentations['val'])
    print('Training starts', flush=True)
    trainer.fit(
        train_transform=augmentations['train'],
        batch_size=args.batch_size,
        epochs=args.epochs, 
        learning_rate=args.learning_rate
    )
    