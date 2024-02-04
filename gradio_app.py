import gradio as gr
from glob import glob
import random as rd
import argparse
from PIL import Image
from typing import List, Dict
from functools import partial

import torch

from model import ClassifierModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='pretrained_vit', help='Name of the model to do prediction with')
    return parser.parse_args()

def predict(img_file: str, model: ClassifierModel, classes: List[str]) -> Dict[str, float]:
    img = model.val_transform(Image.open(img_file)).unsqueeze(0)
    with torch.inference_mode():
        preds = torch.softmax(model(img), dim=1)[0].cpu().numpy()
    return {classes[i] : preds[i] for i in range(len(classes))}
    
if __name__ == '__main__':
    
    args = parse_args()
    model: ClassifierModel = torch.load(f'models/{args.name}.pth', map_location='cpu')
    examples = rd.sample(glob('data/val/*/*.jpg'), k=3)
    classes = [line.strip() for line in open('data/classname.txt').readlines()]

    demo = gr.Interface(
        fn=partial(predict, model=model, classes=classes),
        inputs=gr.Image(type='filepath'),
        outputs=gr.Label(num_top_classes=3),
        examples=examples,
        title='Plants disease classification based on leaves',
        description='Computer vision classifier based on the Vistion Transformer architecture',
        article=''
    )

    demo.launch(share=True)