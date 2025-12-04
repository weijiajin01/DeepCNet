import math
import time
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiCelltypeGeneAE(nn.Module):
    def __init__(
        self,
        num_cells: int,
        celltype_ids: torch.Tensor,
        num_celltypes: int,
        ct_latent_dim: int = 8,
        final_latent_dim: int = 32,
        encoder_hidden: int = 64,
        projection_hidden: int = 128,
        decoder_hidden: int = 128,
        dropout: float = 0.2,
        use_ct_decoders: bool = False,
    ) -> None:
        super().__init__()

        self.num_cells = num_cells
        self.celltype_ids = celltype_ids.to(torch.long)
        self.num_celltypes = num_celltypes

        self.ct_indices: List[torch.Tensor] = []
        self.ct_counts: List[int] = []
        for ct in range(num_celltypes):
            inds = (self.celltype_ids == ct).nonzero(as_tuple=True)[0]
            self.ct_indices.append(inds)
            self.ct_counts.append(int(inds.numel()))

        self.encoders = nn.ModuleList()
        for ct in range(num_celltypes):
            in_dim = max(1, self.ct_counts[ct])
            self.encoders.append(
                nn.Sequential(
                    nn.Linear(in_dim, encoder_hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(encoder_hidden, ct_latent_dim),
                )
            )

        self.use_ct_decoders = use_ct_decoders
        if use_ct_decoders:
            self.ct_decoders = nn.ModuleList()
            for ct in range(num_celltypes):
                out_dim = max(1, self.ct_counts[ct])
                self.ct_decoders.append(
                    nn.Sequential(
                        nn.Linear(ct_latent_dim, encoder_hidden),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(encoder_hidden, out_dim),
                    )
                )

        self.final_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=ct_latent_dim,
                out_channels=final_latent_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.final_fc = nn.Linear(final_latent_dim, final_latent_dim)

        self.global_decoder = nn.Sequential(
            nn.Linear(final_latent_dim, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, num_cells),
        )

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        ct_embs: List[torch.Tensor] = []
        ct_recons: Optional[List[torch.Tensor]] = [] if self.use_ct_decoders else None

        for ct in range(self.num_celltypes):
            inds = self.ct_indices[ct]
            if inds.numel() == 0:
                emb = x.new_zeros((batch, self.encoders[ct][-1].out_features))
                ct_embs.append(emb)
                if self.use_ct_decoders:
                    ct_recons.append(x.new_zeros((batch, 0)))
                continue

            x_ct = x[:, inds]
            emb = self.encoders[ct](x_ct)
            ct_embs.append(emb)

            if self.use_ct_decoders:
                recon_ct = self.ct_decoders[ct](emb)
                ct_recons.append(recon_ct)

        emb_stack = torch.stack(ct_embs, dim=2)
        conv_out = self.final_conv(emb_stack)
        conv_out = conv_out.squeeze(2)
        final_emb = self.final_fc(conv_out)
        global_recon = self.global_decoder(final_emb)

        return global_recon, final_emb, ct_recons, ct_embs


def compute_anchor_similarity_loss(
    ct_embs: Sequence[torch.Tensor],
    anchor_ct_id: int,
    weights_row: torch.Tensor,
) -> torch.Tensor:
    loss = 0.0
    anchor_emb = ct_embs[anchor_ct_id]
    for i, emb in enumerate(ct_embs):
        if i == anchor_ct_id:
            continue
        w = weights_row[i]
        if w == 0:
            continue
        diff = anchor_emb - emb
        dist = (diff.pow(2).sum(dim=1)).mean()
        loss = loss + w * dist
    return loss


def make_dataloaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train).float()),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val).float()),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = None
    if X_test is not None:
        test_loader = DataLoader(
            TensorDataset(torch.tensor(X_test).float()),
            batch_size=batch_size,
            shuffle=False,
        )
    print("Data loaded successfully")
    return train_loader, val_loader, test_loader


def train_gene_autoencoder(
    model: MultiCelltypeGeneAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    celltype_anchor: Optional[str] = None,
    celltype_to_id: Optional[Dict[str, int]] = None,
    weights_df=None,
    lambda_anchor: float = 1.0,
    beta_ct: float = 1.0,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    num_epochs: int = 100,
    patience: int = 5,
    device: Optional[str] = None,
    model_path: Optional[str] = None,
    verbose: bool = True,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = math.inf
    epochs_no_improve = 0

    hist = {
        "train_loss": [],
        "val_loss": [],
        "train_recon_loss": [],
        "train_anchor_loss": [],
        "train_ct_loss": [],
    }

    if celltype_anchor is not None:
        assert celltype_to_id is not None and weights_df is not None
        anchor_ct_id = celltype_to_id[celltype_anchor]
        weights_row = torch.tensor(
            weights_df.loc[celltype_anchor].values,
            device=device,
            dtype=torch.float,
        )
        weights_row[anchor_ct_id] = 0.0
    else:
        anchor_ct_id = None
        weights_row = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_anchor = 0.0
        train_ct_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)
            opt.zero_grad()
            global_recon, final_emb, ct_recons, ct_embs = model(x)

            recon_loss = nn.functional.mse_loss(global_recon, x, reduction="mean")

            if anchor_ct_id is not None:
                anchor_loss = compute_anchor_similarity_loss(
                    ct_embs, anchor_ct_id, weights_row
                )

                ct_loss = 0.0
                if model.use_ct_decoders:
                    for ct_idx, inds in enumerate(model.ct_indices):
                        if inds.numel() == 0:
                            continue
                        x_ct = x[:, inds]
                        recon_ct = ct_recons[ct_idx]
                        ct_loss = ct_loss + nn.functional.mse_loss(
                            recon_ct, x_ct, reduction="mean"
                        )
                else:
                    ct_loss = torch.tensor(0.0, device=device)

                loss = recon_loss + lambda_anchor * anchor_loss + beta_ct * ct_loss
            else:
                loss = recon_loss
                anchor_loss = torch.tensor(0.0, device=device)
                ct_loss = torch.tensor(0.0, device=device)

            loss.backward()
            opt.step()

            bs = x.shape[0]
            train_loss += loss.item() * bs
            train_recon += recon_loss.item() * bs
            train_anchor += lambda_anchor * anchor_loss.item() * bs
            train_ct_loss += beta_ct * ct_loss.item() * bs

        n_train = len(train_loader.dataset)
        train_loss /= n_train
        train_recon /= n_train
        train_anchor /= n_train
        train_ct_loss /= n_train

        hist["train_loss"].append(train_loss)
        hist["train_recon_loss"].append(train_recon)
        hist["train_anchor_loss"].append(train_anchor)
        hist["train_ct_loss"].append(train_ct_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                global_recon, final_emb, ct_recons, ct_embs = model(x)
                recon_loss = nn.functional.mse_loss(
                    global_recon, x, reduction="mean"
                )
                if anchor_ct_id is not None:
                    anchor_loss = compute_anchor_similarity_loss(
                        ct_embs, anchor_ct_id, weights_row
                    )

                    ct_loss = 0.0
                    if model.use_ct_decoders:
                        for ct_idx, inds in enumerate(model.ct_indices):
                            if inds.numel() == 0:
                                continue
                            x_ct = x[:, inds]
                            recon_ct = ct_recons[ct_idx]
                            ct_loss = ct_loss + nn.functional.mse_loss(
                                recon_ct, x_ct, reduction="mean"
                            )
                    else:
                        ct_loss = torch.tensor(0.0, device=device)

                    loss = recon_loss + lambda_anchor * anchor_loss + beta_ct * ct_loss
                else:
                    loss = recon_loss

                val_loss += loss.item() * x.shape[0]

        val_loss /= len(val_loader.dataset)
        hist["val_loss"].append(val_loss)

        if verbose:
            dt = time.time() - t0
            print(
                f"Epoch {epoch} "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"time={dt:.1f}s"
            )

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            if model_path:
                torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    return model, hist


def extract_embeddings(
    model: MultiCelltypeGeneAE,
    X: np.ndarray,
    device: Optional[str] = None,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X).to(device).float()
        _, emb, _, _ = model(X_tensor)
    return emb.cpu().numpy()
