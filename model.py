import lightning as L
from lightning.pytorch.utilities.model_summary import ModelSummary

import torch
import torch.nn.functional as F
import torch.nn as nn

import torchmetrics

class ClassifierModel(L.LightningModule):
    
    def __init__(self, model: nn.Module, image_size: int = 500, learning_rate: float = 1e-3, num_classes: int = 3) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.example_input_array = torch.Tensor(5, 3, image_size, image_size)
        self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def print_summary(self) -> None:
        print(ModelSummary(self, max_depth=-1))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx) -> float:
        X, y = batch
        y_pred = self(X)
        loss = F.cross_entropy(y_pred, y)
        self.log_dict({'Train loss': loss, f'Train F1 score': self.f1_score(y_pred, y)}, 
                      on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx) -> float:
        X, y = batch
        y_pred = self(X)
        loss = F.cross_entropy(y_pred, y)
        self.log_dict({'Validation loss': loss, f'Validation F1 score': self.f1_score(y_pred, y)}, 
                      on_step=False, on_epoch=True)
        return loss
