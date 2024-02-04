import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from model import ClassifierModel
from dataset import ImageClassificationDataModule
from custom_transformer.vit import ViT

from torchvision import models, transforms
import torch.nn as nn
import torch

import argparse

class ModelSaver(L.Callback):
    def __init__(self, monitor='Validation loss', save_dir: str = 'models', name: str = 'base_name'):
        super().__init__()
        self.monitor = monitor
        self.best_loss = float('inf')
        self.save_dir = save_dir + '/' + name + '.pth'

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        val_loss = trainer.callback_metrics.get(self.monitor)
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(pl_module, self.save_dir)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--use_custom',             type=bool,  default=False,      help='Whether to use the custom implementation or the pretrained one')
    parser.add_argument('--image_size',             type=int,   default=500,        help='Input image dimensions')
    parser.add_argument('--patch_size',             type=int,   default=50,         help='Dimensions of image patches')
    parser.add_argument('--epochs',                 type=int,   default=10,         help='Number of epochs to train the model for')
    parser.add_argument('--learning_rate',          type=float, default=1e-3,       help='Model learning rate')
    parser.add_argument('--patience',               type=int,   default=3,          help='Early stopper validation loss patience')
    parser.add_argument('--name',                   type=str,   default='my_vit',   help='Experiment name')
    parser.add_argument('--batch_size',             type=int,   default=10,         help='Dataloader batch size')

    # Custom vit params
    parser.add_argument('--num_transformer_layers', type=int,   default=12,         help='Number of transformer encoder layers')
    parser.add_argument('--embedding_dim',          type=int,   default=768,        help='Embedding dimension of the model')
    parser.add_argument('--num_msa_heads',          type=int,   default=12,         help='Number of selfattention heads in each encoder layer')
    parser.add_argument('--mlp_size',               type=int,   default=3072,       help='Number of hidden units in the MLP block')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    logger = TensorBoardLogger(save_dir='runs', name=args.name)
    num_classes = 3
    
    if args.use_custom:
        vit = ViT(image_size=args.image_size,
                  in_channels=3, 
                  patch_size=args.patch_size, 
                  num_transformer_layers=args.num_transformer_layers,
                  embedding_dim=args.embedding_dim,
                  mlp_size=args.mlp_size,
                  num_heads=args.num_msa_heads,
                  num_classes=num_classes)
        
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor()
        ])
        val_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])

    else:
        if args.image_size != 224:
            print(f'Image size {args.image_size} passed but pretrained vit uses 224. Defaulting to 224')
            args.image_size = 224

        pretrained_vit_weights = models.ViT_B_16_Weights.DEFAULT
        vit = models.vit_b_16(weights=pretrained_vit_weights)

        for parameter in vit.parameters():
            parameter.requires_grad = False
        
        vit.heads = nn.Linear(in_features=768, out_features=num_classes)
        train_transform, val_transform = pretrained_vit_weights.transforms(), pretrained_vit_weights.transforms()

    model = ClassifierModel(vit, image_size=args.image_size, num_classes=num_classes, train_transform=train_transform, val_transform=val_transform)
    # model.print_summary()
    
    early_stopper = EarlyStopping('Validation loss', patience=args.patience)
    model_saver = ModelSaver(name=args.name)

    trainer = L.Trainer(accelerator='auto', devices='auto', 
                        max_epochs=args.epochs, callbacks=[early_stopper, model_saver], 
                        logger=logger, enable_checkpointing=False)
    
    data_module = ImageClassificationDataModule(num_workers=7, batch_size=args.batch_size, train_transform=train_transform, val_transform=val_transform)
    trainer.fit(model=model, datamodule=data_module)
    