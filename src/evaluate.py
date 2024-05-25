import torch
import torch.nn as nn
from torchvision import models
from glob import glob
import random as rd
from PIL import Image
import matplotlib.pyplot as plt

from src.train import Trainer
from src.custom_transformer.vit import ViT
from src.transforms import model_transforms

if __name__ == '__main__':

    val_image_filenames = glob('data/val/*/*.jpg')
    samples = rd.sample(val_image_filenames, k=9)
    val_images = list(map(Image.open, samples))
    classes = [line.strip() for line in open('data/classname.txt').readlines()]
    true_classes = [filename.split('/')[2] for filename in samples]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    custom_vit: ViT = ViT()
    custom_vit.load_state_dict(torch.load('models/my_vit.pt', map_location=device))
    custom_vit_trainer = Trainer(model=custom_vit, device=device, transform=model_transforms['custom']['val'])

    pretrained_vit: models.VisionTransformer = models.vit_b_16()
    pretrained_vit.heads = nn.Linear(768, 3)
    pretrained_vit.load_state_dict(torch.load('models/pretrained_vit.pt', map_location=device))
    pretrained_vit_trainer = Trainer(model=pretrained_vit, device=device, transform=model_transforms['pretrained']['val'])

    fig, ax = plt.subplots(3, 3, figsize=(20,20))
    ax = ax.flatten()
    for i in range(9):
        
        ax[i].imshow(val_images[i])
        ax[i].axis('off')

        true_label = f'True label: {true_classes[i]}'

        custom_label, custom_confidence = custom_vit_trainer.predict(samples[i])
        pretrained_label, pretrained_confidence = pretrained_vit_trainer.predict(samples[i])
        
        custom_vit_str = f'Custom ViT: {custom_label}, Confidence: {custom_confidence:.3f}'
        pretrained_vit_str = f'Pretrained ViT: {pretrained_label}, Confidence: {pretrained_confidence:.3f}'
        ax[i].set_title(true_label + '\n' + custom_vit_str + '\n' + pretrained_vit_str)
    
    plt.tight_layout()
    plt.savefig('example_predictions.png', dpi=200)

    
    