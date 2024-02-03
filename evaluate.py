import torch
from model import ClassifierModel

from glob import glob
import random as rd
from PIL import Image

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    custom_vit: ClassifierModel = torch.load('models/custom_vit.pth', map_location=device)
    custom_vit.eval()

    pretrained_vit: ClassifierModel = torch.load('models/pretrained_vit.pth', map_location=device)
    pretrained_vit.eval()

    val_image_filenames = glob('data/val/*/*.jpg')
    samples = rd.sample(val_image_filenames, k=9)
    val_images = list(map(Image.open, samples))

    with torch.inference_mode():
        pretrained_vit_transformed_images = torch.stack(list(map(pretrained_vit.val_transform, val_images))).to(device)
        pretrained_vit_preds = torch.softmax(pretrained_vit(pretrained_vit_transformed_images), dim=1).cpu()
        
        custom_vit_transformed_images = torch.stack(list(map(custom_vit.val_transform, val_images))).to(device)
        custom_vit_preds = torch.softmax(custom_vit(custom_vit_transformed_images), dim=1).cpu()

    print(pretrained_vit_preds, custom_vit_preds)
    
    # custom_vit = torch.load('models/my_vit.pth', map_location=torch.device('cpu'))
    # custom_vit_data_module = ImageClassificationDataModule(val_transform=custom_vit.val_transform)

    # trainer = L.Trainer(logger=False, accelerator='cpu')

    # print(custom_vit_data_module)

    # pretrained_vit_predictions = trainer.predict(pretrained_vit, datamodule=pretrained_vit_data_module)
    # custom_vit_predictions = trainer.predict(custom_vit, datamodule=custom_vit_data_module)

    # print(custom_vit_predictions[0])
    
    