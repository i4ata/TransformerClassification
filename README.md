# TransformerClassification

My own implementation of the Vision Transformer compared to a pretrained implementation from `torchvision.models`.
Unfortunately, the models are too big so they are not in github. The results can be replicated by running the `train.py` script.

To train:

`python -m src.train`

To visualize training:

`tesnorboard --logdir runs`

To visually see the predictions on some examples:

`python -m src.evaluate`

A Gradio application can be seen on HuggingFace Spaces [here](https://huggingface.co/spaces/i4ata/CustomTransformerClassification)
