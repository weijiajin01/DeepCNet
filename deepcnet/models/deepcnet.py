import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from deepcnet.models.layers import (
    clone_layers,
    MHAttentionLayer,
    MHAttentionLayerNoLN,
)


class DeepCNet(nn.Module):
    """
    Single-condition DeepCNet,
    Produces expression predictions and attention matrix.
    """
    def __init__(
        self,
        n_encoder=3,
        out_dim=64,
        head=4,
        embedding_dim=32,
        epi_dim=2002,
        n_enhancer=60,
        device="cuda",
        useLN=True,
        ablation="None"
    ):
        super().__init__()

        self.n_encoder = n_encoder
        self.n_head = head
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim
        self.n_enhancer = n_enhancer
        self.device = device
        self.useLN = useLN
        self.ablation = ablation
        self.n_extraFeat = 2

        # epi_conv: (B, 61, epi_dim) → (B, 61, out_dim)
        self.epi_conv = nn.Sequential(
            nn.Conv1d(epi_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ELU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )

        # Attention encoder
        if useLN:
            enc_layer = MHAttentionLayer(d_model=out_dim, nhead=head)
        else:
            enc_layer = MHAttentionLayerNoLN(d_model=out_dim, nhead=head)
        self.attn_stack = clone_layers(enc_layer, n_encoder)

        # attention mask: allow promoter -> enhancers, but no self-attention on diagonals
        attn_mask = (~np.identity(self.n_enhancer+1).astype(bool))
        attn_mask[:, 0] = False
        attn_mask[0, :] = False
        attn_mask = torch.from_numpy(attn_mask)
        attn_mask.masked_fill(attn_mask, float('-inf'))
        self.attn_mask = attn_mask

        # Distance / activity feature projection
        self.pos_conv = nn.Sequential(
            nn.Conv1d(out_dim + self.n_extraFeat, out_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=1),
            nn.ReLU(),
        )

        # gene-level projection
        proj_in_dim = out_dim + embedding_dim
        if ablation == "gene_embedding":
            proj_in_dim = out_dim

        self.to_expr = nn.Sequential(
            nn.Linear(proj_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.name = (
            f"DeepCNet."
            f"enh{n_enhancer}."
            f"enc{n_encoder}."
            f"out{out_dim}."
            f"head{head}."
            f"emb{embedding_dim}."
            f"useLN{useLN}"
        )
    def forward(self, dist_feat=None, epi_feat=None, gene_embedding=None):
        # epi_feat: (B, 61, epi_dim)
        padding_mask = (epi_feat.abs().sum(-1) == 0)

        # (B, 61, epi_dim) → (B, out_dim, 61) → (B, 61, out_dim)
        x = self.epi_conv(epi_feat.permute(0, 2, 1)).permute(0, 2, 1)

        # add position/distance features
        if dist_feat is not None and self.ablation != "Dist":
            x_in = torch.cat([x, dist_feat], axis=-1)       # (B, 61, out_dim + n_extraFeat)
            x = self.pos_conv(x_in.permute(0, 2, 1)).permute(0, 2, 1)

        # transformer encoder stack
        attn_list = []
        for layer in self.attn_stack:
            x, attn = layer(
                x,
                padding_mask=padding_mask,
                attn_mask=self.attn_mask.to(self.device),
            )
            attn_list.append(attn.unsqueeze(0))

        # promoter token at position 0
        p_embed = torch.flatten(x[:,0,:], start_dim=1)

        # concat gene embedding
        if gene_embedding is not None and self.ablation != "gene_embedding":
            p_embed = torch.cat([p_embed, gene_embedding], dim=-1)

        expr = self.to_expr(p_embed)
        return expr, attn_list
