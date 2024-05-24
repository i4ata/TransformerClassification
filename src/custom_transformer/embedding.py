import torch
import torch.nn as nn
import math

DEBUG = False

class PatchEmbedding(nn.Module):

    def __init__(self, in_channels: int = 3, embedding_dim: int = 768, patch_size: int = 16) -> None:
        
        super(PatchEmbedding, self).__init__()
        self.linear_projection = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Input: [batch_size, in_channels, H, W]
        if DEBUG: print(f'Patch embedding input shape: {x.shape} [batch_size, in_channels, image_height, image_width]')

        # Linear Projection: [batch_size, embedding_dim, sqrt(n_patches), sqrt(n_patches)]
        x = self.linear_projection(x)
        if DEBUG: print(f'Linearly projected input: {x.shape} [batch_size, embedding_dim, sqrt(n_patches), sqrt(n_patches)]')

        # Flattening: [batch_size, embedding_dim, n_patches]
        x = x.flatten(start_dim=2)
        if DEBUG: print(f'Flattening of last 2 dimensions of linear projection: {x.shape} [batch_size, embedding_dim, n_patches]')

        # Transpose last 2 dimensions: [batch_size, n_patches, embedding_dim]
        x = x.mT
        if DEBUG: print(f'Transpose last 2 dimensions: {x.shape} [batch_size, n_patches, embedding_dim]')

        return x
    
class Embedding(nn.Module):

    def __init__(self, image_size: int = 224, in_channels: int = 3, embedding_dim: int = 768, patch_size: int = 16) -> None:
        
        super(Embedding, self).__init__()

        assert image_size % patch_size == 0

        self.n_patches = (image_size * image_size) // (patch_size * patch_size)
        if DEBUG: print(f'Total number of patches: {self.n_patches}, i.e. {int(math.sqrt(self.n_patches))} x {int(math.sqrt(self.n_patches))}')

        # Patch embedding defined above
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, embedding_dim=embedding_dim, patch_size=patch_size)
        
        # The class token x0, 1 for each embedding dim
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # The positional embedding, `n_patches` many for each embedding dim
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches + 1, embedding_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if DEBUG: print(f'Embedding input shape: {x.shape}: [batch_size, in_channels, height, width]')

        x = self.patch_embedding(x)
        if DEBUG: print(f'Patch embedding output: {x.shape}: [batch_size, n_patches, embedding_dim]')

        x = torch.cat((self.class_token.expand(len(x), -1, -1), x), dim=1)
        if DEBUG: print(f'Class token prepended: {x.shape}: [batch_size, n_patches + 1, embedding_dim]')

        x = x + self.position_embedding
        if DEBUG: print(f'Positional embedding added: {x.shape}: [batch_size, n_patches + 1, embedding_dim]')

        return x
    
if __name__ == '__main__':
    DEBUG = True
    sample_image_batch = torch.rand(5,3,224,224)
    embedding = Embedding()
    out = embedding(sample_image_batch)
    print(out)
