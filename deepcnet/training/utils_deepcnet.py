import os
import sys
import h5py
import torch
torch._dynamo.disable()
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats
from tqdm import tqdm

from deepcnet.utils.utils_loggers import get_lr, Logger, EarlyStopping


def train(
    model,
    dataset,
    fold_idx,
    save_dir="../models",
    lr=5e-4,
    patience = 3,
    logger=None,
    valid_dataset=None,
    model_name=None,
    batch_size=64,
    device="cuda",
    EPOCHS=20,
    valid_size=1000,
    stratify=None
):
    """Train DeepCNet model."""

    os.makedirs(save_dir, exist_ok=True)

    if model_name is None:
        model_name = model.name

    # train/valid split (if no external validation set provided)
    if valid_dataset is None:
        idx_train, idx_val = train_test_split(
            list(range(len(dataset))),
            test_size=valid_size,
            shuffle=True,
            random_state=42,
            stratify=stratify,
        )
        train_ds = Subset(dataset, idx_train)
        valid_ds = Subset(dataset, idx_val)
    else:
        train_ds = dataset
        valid_ds = valid_dataset

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True
    )

    ckpt_path = os.path.join(
        save_dir, f"fold_{fold_idx}_best_{model_name}_checkpoint.pt"
    )
    stopper = EarlyStopping(patience=patience, verbose=True, path=ckpt_path)

    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)

    print(f"Fold {fold_idx}: train={len(train_ds)}  valid={len(valid_ds)}  total={len(dataset)}")
    print("Model:", model.name)

    model.to(device)
    for epoch in range(EPOCHS):
        model.train()
        print("LR:", get_lr(optimizer))

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            try:
                dist_feat, y_expr, epi_feat, ct_embed, eid = batch
                ct_embed = ct_embed.float().to(device)
            except ValueError:
                dist_feat, y_expr, epi_feat, eid = batch
                ct_embed = None

            dist_feat = dist_feat.float().to(device)
            epi_feat = epi_feat.float().to(device)
            y_expr = y_expr.float().to(device)

            pred, _ = model(
                dist_feat=dist_feat,
                epi_feat=epi_feat,
                gene_embedding=ct_embed,
            )

            loss = loss_fn(pred, y_expr)
            loss.backward()
            optimizer.step()

        # validation metrics
        mse, r2, pearson = validate(
            model, valid_ds, device=device
        )
        print("Validation R2:", r2)

        stopper(-r2, model, epoch)
        if logger is not None:
            logger.add([fold_idx, epoch, mse, r2, pearson, stopper.counter])

        if stopper.early_stop:
            print("Early stopping.")
            break

    return


def validate(model, valid_ds, batch_size=32, device="cuda"):
    """Evaluate MSE / R2 / Pearson on validation set."""

    loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True
    )
    loss_fn = torch.nn.MSELoss()
    preds, actual = [], []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            try:
                dist_feat, y_expr, epi_feat, ct_embed, eid = batch
                ct_embed = ct_embed.float().to(device)
            except ValueError:
                dist_feat, y_expr, epi_feat, eid = batch
                ct_embed = None

            dist_feat = dist_feat.float().to(device)
            epi_feat = epi_feat.float().to(device)
            y_expr = y_expr.float().to(device)

            pred, _ = model(
                dist_feat=dist_feat,
                epi_feat=epi_feat,
                gene_embedding=ct_embed,
            )

            preds.extend(pred.flatten().cpu().numpy())
            actual.extend(y_expr.flatten().cpu().numpy())

    slope, intercept, r_value, _, _ = stats.linregress(preds, actual)
    pearson_r, _ = stats.pearsonr(preds, actual)
    mse = mean_squared_error(preds, actual)

    print("Validation: MSE", mse, "R2", r_value**2, "Pearson", pearson_r)
    return mse, r_value**2, pearson_r


def test(model, test_ds, fold_idx, save_dir="../models", model_name=None, batch_size=32, device="cuda"):
    """Load checkpoint and evaluate on test set."""
    if model_name is None:
        model_name = model.name

    ckpt_file = os.path.join(save_dir, f"fold_{fold_idx}_best_{model_name}_checkpoint.pt")
    if os.path.exists(ckpt_file):
        print("Loading checkpoint:", ckpt_file)
        ckpt = torch.load(ckpt_file,weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    else:
        print("Checkpoint not found:", ckpt_file)

    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True
    )

    preds, actual, ids = [], [], []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(loader):
            try:
                dist_feat, y_expr, epi_feat, ct_embed, eid = batch
                ct_embed = ct_embed.float().to(device)
            except ValueError:
                dist_feat, y_expr, epi_feat, eid = batch
                ct_embed = None

            dist_feat = dist_feat.float().to(device)
            epi_feat = epi_feat.float().to(device)
            y_expr = y_expr.float().to(device)

            pred, _ = model(
                dist_feat=dist_feat,
                epi_feat=epi_feat,
                gene_embedding=ct_embed,
            )

            preds.extend(pred.flatten().cpu().numpy())
            actual.extend(y_expr.flatten().cpu().numpy())
            ids.extend(eid)

    slope, intercept, r_value, p_val, _ = stats.linregress(preds, actual)
    pearson_r, _ = stats.pearsonr(preds, actual)
    mse = mean_squared_error(preds, actual)

    print("\nTest metrics:")
    print("  MSE    :", mse)
    print("  R2     :", r_value**2)
    print("  Pearson:", pearson_r)

    df = pd.DataFrame(
        {"Pred": preds, "Actual": actual, "Fold": fold_idx},
        index=np.array(ids).flatten(),
    )

    out_file = os.path.join(
        save_dir, f"fold_{fold_idx}_{model_name}_predictions.csv"
    )
    df.to_csv(out_file)
    print("Saved predictions:", out_file)

    return df


class DeepCNetDataset(Dataset):
    """
    Dataset for DeepCNet.
    Returns:
        - distance & activity tensor (pe_feat_tensor)
        - expression tensor (expr_tensor)
        - epigenetic embedding tensor (epi_tensor)
        - optional gene embedding
        - gene id
    """

    def __init__(
        self,
        activity_folder,
        expression_folder,
        enhancer_info_path,
        epi_path,
        n_enhancers=60,
        gene_embedding=None,
    ):
        self.activity_h5 = h5py.File(os.path.join(activity_folder, "enhancer_activity.h5"), "r")
        self.epi_h5 = h5py.File(epi_path, "r")
        self.enhancer_info_h5 = h5py.File(enhancer_info_path, "r")

        self.expr_df = pd.read_csv(
            os.path.join(expression_folder, "gene_expr.csv"),
            index_col="ENSID"
        )

        self.n_enhancers = n_enhancers
        self.gene_embedding = gene_embedding

    def __len__(self):
        return len(self.enhancer_info_h5["ensid"])

    def __getitem__(self, idx):
        ensid = self.enhancer_info_h5["ensid"][idx].decode()

        # enhancer distance (61 values: 1 promoter + 60 enhancers)
        enhancer_distance = self.enhancer_info_h5["distance"][idx, 1:]
        pe_distance = np.concatenate([[0], enhancer_distance / 1000])

        # enhancer activity (61 values: 1 promoter + 60 enhancers)
        activity = self.activity_h5["enhancer_activity"][:, idx].T
        enhancer_activity = activity[1:]                    # skip promoter
        pe_activity = np.concatenate([[0], enhancer_activity])

        # distance + activity → final extraFeat
        # shape: (61, 2)
        pe_feat = np.stack([pe_distance, pe_activity], axis=-1)
        pe_feat_tensor = torch.from_numpy(pe_feat[: self.n_enhancers + 1]).float()

        # epigenetic tensor (61 × epi_dim)
        enhancer_epi = self.epi_h5["epigenetics"][idx]
        epi_tensor = torch.from_numpy(enhancer_epi).float()

        # expression
        expr_value = float(self.expr_df.loc[ensid]["expression"])
        expr_tensor = torch.tensor([expr_value]).float()

        # optional gene embedding
        if self.gene_embedding is not None:
            gene_emb = self.gene_embedding[idx]
            return pe_feat_tensor, expr_tensor, epi_tensor, gene_emb, ensid

        return pe_feat_tensor, expr_tensor, epi_tensor, ensid

