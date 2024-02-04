# TransformerClassification

My own implementation of the Vision Transformer compared to a pretrained implementation from `torchvision.models`.
Unfortunately, the models are too big so they are not in github. The results can be replicated by running the `train.py` script.

To train:

`python train.py`

To visualize training:

`tesnorboard --logdir runs`

To visually see the predictions on some examples:

`python evaluate.py`

To launch a Gradio application:

`python gradio_app.py`