import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from model import ClassifierModel
from dataset import ImageClassificationDataModule
from custom_transformer.vit import ViT

if __name__ == '__main__':

    logger = TensorBoardLogger(save_dir='runs', name='my_vit')
    vit = ViT(image_size=500, patch_size=50, num_classes=3)
    model = ClassifierModel(vit, image_size=500, num_classes=3)
    # model.print_summary()
    early_stopper = EarlyStopping('Validation loss', patience=5)
    trainer = L.Trainer(accelerator='gpu', max_epochs=20, callbacks=[early_stopper], logger=logger)
    data_module = ImageClassificationDataModule(num_workers=7)
    trainer.fit(model=model, datamodule=data_module)
