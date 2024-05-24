from torchvision import transforms, models
from typing import Literal, Dict

_weights = models.ViT_B_16_Weights.DEFAULT

model_transforms: Dict[Literal['custom', 'pretrained'], Dict[Literal['train', 'val'], transforms.Compose]] = {
    'custom': {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    },
    'pretrained': {
        'train': _weights.transforms(),
        'val': _weights.transforms()
    }
}