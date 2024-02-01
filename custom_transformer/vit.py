import torch
import torch.nn as nn

import sys
sys.path.append('..')
from custom_transformer.embedding import Embedding
from custom_transformer.encoder import TransformerEncoderBlock

class ViT(nn.Module):

    def __init__(self, 
                 image_size: int = 224,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 num_transformer_layers: int = 12,
                 embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 num_classes: int = 3) -> None:
        
        super().__init__()

        self.embedding = Embedding(image_size=image_size, in_channels=in_channels, embedding_dim=embedding_dim, patch_size=patch_size)
        self.transformer_encoders = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim=embedding_dim, hidden_size=mlp_size, num_heads=num_heads)
              for _ in range(num_transformer_layers)]
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoders(x)
        x = self.classifier(x[:, 0])
        return x
    
if __name__ == '__main__':
    sample_image_batch = torch.rand(5,3,500,500)
    vit = ViT(image_size=500, patch_size=50)
    print(vit(sample_image_batch).shape)