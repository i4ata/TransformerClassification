import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from model import ClassifierModel
from dataset import ImageClassificationDataModule
from custom_transformer.vit import ViT

import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('use_custom', type=bool, default=True, help='Whether to use the custom implementation or the pretrained one')
    parser.add_argument('image_size', type=int, default=500, help='Input image dimensions')
    parser.add_argument('patch_size', type=int, default=50, help='Dimensions of image patches')
    parser.add_argument('epochs', type=int, default=10, help='Number of epochs to train the model for')

if __name__ == '__main__':

    logger = TensorBoardLogger(save_dir='runs', name='my_vit')
    vit = ViT(image_size=500, patch_size=50, num_classes=3)
    model = ClassifierModel(vit, image_size=500, num_classes=3)
    # model.print_summary()
    early_stopper = EarlyStopping('Validation loss', patience=3)
    trainer = L.Trainer(accelerator='auto', devices='auto', max_epochs=10, callbacks=[early_stopper], logger=logger, default_root_dir='models/')
    data_module = ImageClassificationDataModule(num_workers=7)
    trainer.fit(model=model, datamodule=data_module)
    preds = trainer.predict(model=model, datamodule=data_module)[0]
    print(preds['y_pred'], preds['y_true'])
    