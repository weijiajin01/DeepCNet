import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deepcnet.models.layers import (
    clone_layers,
    MHAttentionLayer,
    MHAttentionLayerNoLN,
)


class DeepCNetD(nn.Module):
    """
    Two-condition DeepCNet with condition-specific attention biases.
    Produces expression predictions and attention matrix for Control and Disease conditions separately.
    """
    def __init__(
        self,
        n_encoder=3,
        out_dim=128,
        head=4,
        embedding_dim=32,
        epi_dim=2002,
        n_enhancer=60,
        device="cuda",
        useLN=True
    ):
        super().__init__()

        self.n_encoder = n_encoder
        self.out_dim = out_dim
        self.epi_dim = epi_dim
        self.n_enhancer = n_enhancer
        self.n_head = head
        self.embedding_dim = embedding_dim
        self.device = device
        self.useLN = useLN

        # -----------------------
        # Epigenetic 1×1 Conv block
        # -----------------------
        self.epi_conv = nn.Sequential(
            nn.Conv1d(epi_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ELU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )

        # -----------------------
        # Attention encoder layers
        # -----------------------
        if useLN:
            layer = MHAttentionLayer(d_model=out_dim, nhead=head)
        else:
            layer = MHAttentionLayerNoLN(d_model=out_dim, nhead=head)

        self.attn_blocks = clone_layers(layer, n_encoder)

        # -----------------------
        # Base attention mask (disallow enhancer-enhancer self attn except promoter index 0)
        # -----------------------
        mask = (~np.identity(self.n_enhancer + 1).astype(bool))
        mask[:, 0] = False
        mask[0, :] = False
        mask = torch.from_numpy(mask).float()
        mask.masked_fill_(mask.bool(), float("-inf"))
        self.register_buffer("attn_mask_base", mask)

        # -----------------------
        # Condition-specific biases
        # -----------------------
        self.attn_bias_control = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, n_enhancer + 1)) for _ in range(n_encoder)]
        )
        self.attn_bias_disease = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, n_enhancer + 1)) for _ in range(n_encoder)]
        )

        # -----------------------
        # Distance / activity feature projection
        # -----------------------
        self.add_pos_conv = nn.Sequential(
            nn.Conv1d(out_dim + 2, out_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=1),
            nn.ReLU(),
        )

        # -----------------------
        # Final promoter → expression MLP
        # -----------------------
        self.pToExpr = nn.Sequential(
            nn.Linear(out_dim + embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.name = (
            f"DeepCNet-D."
            f"enh{n_enhancer}."
            f"enc{n_encoder}."
            f"out{out_dim}."
            f"head{head}."
            f"emb{embedding_dim}."
            f"useLN{useLN}"
        )


    # =====================================================================
    # Forward pass for two conditions
    # =====================================================================
    def forward(
        self,
        distfeat_Control=None,
        distfeat_Disease=None,
        epi_feat=None,
        embedding_Control=None,
        embedding_Disease=None,
    ):

        # -----------------------
        # Padding mask (where epi = 0)
        # -----------------------
        padding_mask = (epi_feat.abs().sum(-1) == 0).float()  
        padding_mask = padding_mask.masked_fill(padding_mask == 1, float('-inf'))


        # -----------------------
        # Epigenetic projection
        # -----------------------
        x = self.epi_conv(epi_feat.permute(0, 2, 1)).permute(0, 2, 1)

        # -----------------------
        # Build condition-specific enhancer representations
        # -----------------------
        if distfeat_Control is not None:
            x_control = self.add_pos_conv(
                torch.cat([x, distfeat_Control], dim=-1).permute(0, 2, 1)
            ).permute(0, 2, 1)
        else:
            x_control = x

        if distfeat_Disease is not None:
            x_disease = self.add_pos_conv(
                torch.cat([x, distfeat_Disease], dim=-1).permute(0, 2, 1)
            ).permute(0, 2, 1)
        else:
            x_disease = x

        attn_list_control = []
        attn_list_disease = []

        # -----------------------
        # Run through encoder layers
        # -----------------------
        for i in range(self.n_encoder):

            # Control mask
            mask_control = self.attn_mask_base.clone()
            mask_control[0, :] += self.attn_bias_control[i][0]

            x_disease, w_control = self.attn_blocks[i](
                x_control,
                padding_mask,
                attn_mask=mask_control.to(self.device),
            )
            attn_list_control.append(w_control.unsqueeze(0))

            # disease mask
            mask_disease = self.attn_mask_base.clone()
            mask_disease[0, :] += self.attn_bias_disease[i][0]

            x_disease, w_disease = self.attn_blocks[i](
                x_disease,
                padding_mask,
                attn_mask=mask_disease.to(self.device),
            )
            attn_list_disease.append(w_disease.unsqueeze(0))

        # promoter token at position 0
        p_control = torch.flatten(x_control[:, 0, :], start_dim=1)
        p_disease = torch.flatten(x_disease[:, 0, :], start_dim=1)

        # concat gene embedding
        if embedding_Control is not None:
            p_control = torch.cat([p_control, embedding_Control], dim=-1)
        if embedding_Disease is not None:
            p_disease = torch.cat([p_disease, embedding_Disease], dim=-1)

        # -----------------------
        # Predict expression
        # -----------------------
        expr_control = self.pToExpr(p_control)
        expr_disease = self.pToExpr(p_disease)

        # -----------------------
        # Return outputs
        # -----------------------
        return (
            expr_control,
            expr_disease,
            attn_list_control,
            attn_list_disease,
            self.attn_bias_control,
            self.attn_bias_disease,
        )
