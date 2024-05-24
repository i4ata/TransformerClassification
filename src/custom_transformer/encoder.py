import torch
import torch.nn as nn 
import math

DEBUG = False

class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, embedding_dim: int = 768, num_heads: int = 12) -> None:
        
        super(MultiHeadSelfAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.q_w, self.k_w, self.v_w, self.out_w = (nn.Linear(embedding_dim, embedding_dim) for _ in range(4))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

        if DEBUG: print(f'MSA Input shape (Q, K, V): {q.shape}: [batch_size, n_patches, embedding_dim]')

        # Linear projections for Q, K, V
        if DEBUG: print(f'Linear projection for Q, K, V: {q.shape} [batch_size, n_patches, embedding_dim]')
        q = self.q_w(q).view(*q.shape[:-1], self.num_heads, self.head_dim)
        k = self.k_w(k).view(*k.shape[:-1], self.num_heads, self.head_dim)
        v = self.q_w(v).view(*v.shape[:-1], self.num_heads, self.head_dim)
        if DEBUG: print(f'Splitting the last dimension once for each head: {q.shape} [batch_size, n_patches, num_heads, head_dim]')

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if DEBUG: print(f'Swap patches and head to have the head come first: {q.shape} [batch_size, num_heads, n_patches, head_dim]')

        attention_scores = torch.matmul(q, k.mT) / math.sqrt(self.head_dim)
        if DEBUG: print(f'Compute attention scores for each head (scaled dot product): {attention_scores.shape} [batch_size, num_heads, n_patches, n_patches]')

        attention_weights = torch.softmax(attention_scores, dim=-1)
        if DEBUG: print(f'Softmax of attention scores: {attention_weights.shape} [batch_size, num_batches, n_patches, n_patches]')

        weighted_sum = torch.matmul(attention_weights, v)
        if DEBUG: print(f'Weighted sum of Values: {weighted_sum.shape} [batch_size, num_heads, n_patches, head_dim]')

        weighted_sum = weighted_sum.transpose(1, 2).contiguous()
        if DEBUG: print(f'Swap again the patches and the heads: {weighted_sum.shape} [batch_size, n_patches, num_heads, head_dim]')

        weighted_sum = weighted_sum.view(*weighted_sum.shape[:-2], -1)
        if DEBUG: print(f'Recover the original dimensions by merging the last 2: {weighted_sum.shape} [batch_size, n_patches, embedding_dim]')

        output = self.out_w(weighted_sum)
        if DEBUG: print(f'(Output) Linear projection of the weighted sum: {output.shape} [batch_size, num_heads, n_patches, embedding_dim]')
        
        return output
        

class MSABlock(nn.Module):

    def __init__(self, embedding_dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.msa = MultiHeadSelfAttention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        return self.msa(x, x, x)

class MLPBlock(nn.Module):

    def __init__(self, embedding_dim: int = 768, hidden_size: int = 3072) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=hidden_size),
            nn.GELU(),
            nn.Linear(in_features=hidden_size, out_features=embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.layer_norm(x))


class TransformerEncoderBlock(nn.Module):
    
    def __init__(self, embedding_dim: int = 768, hidden_size: int = 3072, num_heads: int = 12) -> None:
        super().__init__()
        self.msa = MSABlock(embedding_dim=embedding_dim, num_heads=num_heads)
        self.mlp = MLPBlock(embedding_dim=embedding_dim, hidden_size=hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.msa(x) + x
        x = self.mlp(x) + x
        return x

if __name__ == '__main__':

    DEBUG = True
    x = torch.rand(5, 197, 768)
    msa = MultiHeadSelfAttention()
    out = msa(x,x,x)
    print(out.shape)