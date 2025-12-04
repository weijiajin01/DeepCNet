import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


def clone_layers(module, n):
    """Create N identical copies of a layer."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MHAttentionLayer(nn.Module):
    """Multi-head self-attention encoder block with LayerNorm."""

    def __init__(self, d_model=128, nhead=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, padding_mask=None, attn_mask=None):
        h = self.norm1(x)
        h, w = self.attn(h, h, h, key_padding_mask=padding_mask, attn_mask=attn_mask)
        x = x + h

        h = self.norm2(x)
        x = x + self.ff(h)
        return x, w


class MHAttentionLayerNoLN(nn.Module):
    """Multi-head attention encoder block without LayerNorm."""

    def __init__(self, d_model=128, nhead=8, ff_dim=256, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None, attn_mask=None):
        h, w = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)
        x = x + h

        h = self.fc2(self.dropout(F.relu(self.fc1(x))))
        x = x + self.dropout(h)

        return x, w
