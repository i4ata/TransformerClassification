import torch
import lightning as L
from model import ClassifierModel
from dataset import ImageClassificationDataModule

if __name__ == '__main__':
    
    pretrained_vit = ClassifierModel.load_from_checkpoint('models/pretrained_vit.ckpt')
    pretrained_vit_data_module = ImageClassificationDataModule(val_transform=pretrained_vit.val_transform)    
    
    # custom_vit = ClassifierModel.load_from_checkpoint('models/pretrained_vit.ckpt')
    # custom_vit_data_module = ImageClassificationDataModule(val_transform=custom_vit.val_transform)

    trainer = L.Trainer(accelerator='auto', devices='auto')
    
    pretrained_vit_predictions = trainer.predict(pretrained_vit, datamodule=pretrained_vit_data_module)
    # custom_vit_predictions = trainer.predict(custom_vit, datamodule=custom_vit_data_module)

    print(pretrained_vit_predictions[0])
    
    