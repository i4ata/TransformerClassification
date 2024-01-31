from torchvision.datasets import ImageFolder
from torchvision import transforms

# https://www.kaggle.com/datasets/marquis03/bean-leaf-lesions-classification/data
# All images are [3, 500, 500]
print('imports done')
d = ImageFolder(root='data/train', transform=transforms.ToTensor())

print(len(d))