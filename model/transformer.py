import torch.nn as nn
from .attention import MultiHeadCausalAttention

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads, dropout=0.1, block_size=64):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadCausalAttention(n_embd, n_heads, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
