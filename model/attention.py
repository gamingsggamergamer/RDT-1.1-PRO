import torch
import torch.nn as nn

class MultiHeadCausalAttention(nn.Module):
    def __init__(self, n_embd, n_heads, dropout=0.1, block_size=64):
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_size = n_embd // n_heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(y))
