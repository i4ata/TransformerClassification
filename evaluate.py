import torch

from model import ClassifierModel

from glob import glob
import random as rd
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':

    val_image_filenames = glob('data/val/*/*.jpg')
    samples = rd.sample(val_image_filenames, k=9)
    val_images = list(map(Image.open, samples))
    classes = [line.strip() for line in open('data/classname.txt').readlines()]
    
    true_classes = [filename.split('/')[2] for filename in samples]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    custom_vit: ClassifierModel = torch.load('models/my_vit.pth', map_location=device)
    custom_vit.eval()

    pretrained_vit: ClassifierModel = torch.load('models/pretrained_vit.pth', map_location=device)
    pretrained_vit.eval()

    with torch.inference_mode():
        pretrained_vit_transformed_images = torch.stack(list(map(pretrained_vit.val_transform, val_images))).to(device)
        pretrained_vit_preds = torch.softmax(pretrained_vit(pretrained_vit_transformed_images), dim=1).cpu()
        
        custom_vit_transformed_images = torch.stack(list(map(custom_vit.val_transform, val_images))).to(device)
        custom_vit_preds = torch.softmax(custom_vit(custom_vit_transformed_images), dim=1).cpu()

    fig, ax = plt.subplots(3, 3, figsize=(20,20))
    ax = ax.flatten()
    for i in range(9):
        
        ax[i].imshow(val_images[i])
        ax[i].set_axis_off()

        true_label = f'True label: {true_classes[i]}'
        custom_vit_str = f'Custom ViT: {classes[torch.argmax(custom_vit_preds[i])]}, Confidence: {torch.max(custom_vit_preds[i]):.3f}'
        pretrained_vit_str = f'Pretrained ViT: {classes[torch.argmax(pretrained_vit_preds[i])]}, Confidence: {torch.max(pretrained_vit_preds[i]):.3f}'
        ax[i].set_title(true_label + '\n' + custom_vit_str + '\n' + pretrained_vit_str)
    
    plt.tight_layout()
    plt.savefig('example_predictions.png', dpi=200)

    
    