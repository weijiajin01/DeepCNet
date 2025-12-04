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

def train_two_cond(
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
    os.makedirs(save_dir, exist_ok=True)

    if model_name is None:
        model_name = model.name

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

    print(
        f"[Fold {fold_idx}] train={len(train_ds)}  valid={len(valid_ds)}"
    )

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
            
            (
                feat_C, expr_C, epi_C, emb_C,
                feat_D, expr_D, epi_D, emb_D,
                ensid,
            ) = batch

            feat_C = feat_C.float().to(device)
            expr_C = expr_C.float().to(device)
            epi_C = epi_C.float().to(device)

            feat_D = feat_D.float().to(device)
            expr_D = expr_D.float().to(device)
            epi_D = epi_D.float().to(device)

            if isinstance(emb_C, torch.Tensor):
                emb_C = emb_C.float().to(device)
            else:
                emb_C = None

            if isinstance(emb_D, torch.Tensor):
                emb_D = emb_D.float().to(device)
            else:
                emb_D = None

            pred_C, pred_D, _, _, _, _ = model(
                distfeat_Control=feat_C,
                distfeat_Disease=feat_D,
                epi_feat=epi_C,
                embedding_Control=emb_C,
                embedding_Disease=emb_D,
            )

            loss_C = loss_fn(pred_C, expr_C)
            loss_D = loss_fn(pred_D, expr_D)
            loss = 0.5 * (loss_C + loss_D)

            loss.backward()
            optimizer.step()

        (
            mse_C, r2_C, pr_C,
            mse_D, r2_D, pr_D,
        ) = validate_two_cond(
            model,
            valid_ds,
            batch_size=batch_size,
            device=device,
        )

        r2_avg = 0.5 * (r2_C + r2_D)
        mse_avg = 0.5 * (mse_C + mse_D)
        pr_avg = 0.5 * (pr_C + pr_D)

        print(
            f"Validation R² Control: {r2_C:.4f}, "
            f"Disease: {r2_D:.4f}, Avg: {r2_avg:.4f}"
        )

        stopper(-r2_avg, model, epoch)

        if logger is not None:
            label_type = model.name.split(".")[-1]
            logger.add([
                fold_idx, epoch, train_loss,
                mse_C, pr_C, r2_C,
                mse_D, pr_D, r2_D,
                mse_avg, pr_avg, r2_avg,
                stopper.counter, label_type,
            ])

        if stopper.early_stop:
            print("Early stopping triggered.")
            break

    return


def validate_two_cond(
    model,
    valid_ds,
    batch_size=16,
    device="cuda",
):
    loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True
    )

    model.to(device)
    model.eval()
    loss_fn = torch.nn.MSELoss()

    preds_C, preds_D = [], []
    actual_C, actual_D = [], []
    loss_C_sum, loss_D_sum = 0.0, 0.0

    with torch.no_grad():
        for batch in loader:
            (
                feat_C, expr_C, epi_C, emb_C,
                feat_D, expr_D, epi_D, emb_D,
                ensid,
            ) = batch

            feat_C = feat_C.float().to(device)
            expr_C = expr_C.float().to(device)
            epi_C = epi_C.float().to(device)

            feat_D = feat_D.float().to(device)
            expr_D = expr_D.float().to(device)
            epi_D = epi_D.float().to(device)

            if isinstance(emb_C, torch.Tensor):
                emb_C = emb_C.float().to(device)
            else:
                emb_C = None

            if isinstance(emb_D, torch.Tensor):
                emb_D = emb_D.float().to(device)
            else:
                emb_D = None

            pred_C, pred_D, _, _, _, _ = model(
                distfeat_Control=feat_C,
                distfeat_Disease=feat_D,
                epi_feat=epi_C,
                embedding_Control=emb_C,
                embedding_Disease=emb_D,
            )

            loss_C = loss_fn(pred_C, expr_C)
            loss_D = loss_fn(pred_D, expr_D)
            loss_C_sum += loss_C.item()
            loss_D_sum += loss_D.item()

            preds_C.extend(pred_C.flatten().cpu().numpy().tolist())
            actual_C.extend(expr_C.flatten().cpu().numpy().tolist())
            preds_D.extend(pred_D.flatten().cpu().numpy().tolist())
            actual_D.extend(expr_D.flatten().cpu().numpy().tolist())

    def _metrics(preds, actual):
        slope, intercept, r_value, _, _ = stats.linregress(preds, actual)
        pearson_r, _ = stats.pearsonr(preds, actual)
        mse = mean_squared_error(preds, actual)
        return mse, r_value**2, pearson_r

    mse_C, r2_C, pr_C = _metrics(preds_C, actual_C)
    mse_D, r2_D, pr_D = _metrics(preds_D, actual_D)

    print(f"Validation loss Control: {loss_C_sum/len(loader):.6f}")
    print(f"Validation loss Disease: {loss_D_sum/len(loader):.6f}")
    print(f"Control  - MSE {mse_C:.4f}, R² {r2_C:.4f}, Pearson {pr_C:.4f}")
    print(f"Disease  - MSE {mse_D:.4f}, R² {r2_D:.4f}, Pearson {pr_D:.4f}")

    return mse_C, r2_C, pr_C, mse_D, r2_D, pr_D


def test_two_cond(
    model,
    test_ds,
    fold_idx,
    save_dir="../models",
    model_name=None,
    batch_size=16,
    device="cuda",
):
    if model_name is None:
        model_name = model.name

    ckpt_file = os.path.join(save_dir, f"fold_{fold_idx}_best_{model_name}_checkpoint.pt")
    if os.path.exists(ckpt_file):
        print("Loading checkpoint:", ckpt_file)
        ckpt = torch.load(ckpt_file,weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    else:
        print("Checkpoint not found:", ckpt_file)

    testloader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True
    )

    model.to(device)
    model.eval()

    preds_C, preds_D = [], []
    actual_C, actual_D = [], []
    ensids = []

    with torch.no_grad():
        for batch in tqdm(testloader):
            (
                feat_C, expr_C, epi_C, emb_C,
                feat_D, expr_D, epi_D, emb_D,
                eid,
            ) = batch

            feat_C = feat_C.float().to(device)
            expr_C = expr_C.float().to(device)
            epi_C = epi_C.float().to(device)

            feat_D = feat_D.float().to(device)
            expr_D = expr_D.float().to(device)
            epi_D = epi_D.float().to(device)

            if isinstance(emb_C, torch.Tensor):
                emb_C = emb_C.float().to(device)
            else:
                emb_C = None

            if isinstance(emb_D, torch.Tensor):
                emb_D = emb_D.float().to(device)
            else:
                emb_D = None

            pred_C, pred_D, _, _, _, _ = model(
                distfeat_Control=feat_C,
                distfeat_Disease=feat_D,
                epi_feat=epi_C,
                embedding_Control=emb_C,
                embedding_Disease=emb_D,
            )

            preds_C.extend(pred_C.flatten().cpu().numpy().tolist())
            actual_C.extend(expr_C.flatten().cpu().numpy().tolist())
            preds_D.extend(pred_D.flatten().cpu().numpy().tolist())
            actual_D.extend(expr_D.flatten().cpu().numpy().tolist())
            ensids.extend(eid)

    def _metrics(preds, actual):
        slope, intercept, r_value, _, _ = stats.linregress(preds, actual)
        pearson_r, _ = stats.pearsonr(preds, actual)
        mse = mean_squared_error(preds, actual)
        return r_value**2, pearson_r, mse

    r2_C, pr_C, mse_C = _metrics(preds_C, actual_C)
    r2_D, pr_D, mse_D = _metrics(preds_D, actual_D)

    print(f"Control  - R²: {r2_C:.4f}, Pearson: {pr_C:.4f}, MSE: {mse_C:.4f}")
    print(f"Disease  - R²: {r2_D:.4f}, Pearson: {pr_D:.4f}, MSE: {mse_D:.4f}")

    df = pd.DataFrame({
        "ENSG": np.array(ensids).flatten(),
        "Pred_Control": preds_C,
        "Actual_Control": actual_C,
        "Pred_Disease": preds_D,
        "Actual_Disease": actual_D,
        "fold_idx": fold_idx,
    })

    out_path = os.path.join(
        save_dir,
        f"fold_{fold_idx}_{model_name}_predictions.csv",
    )
    df.to_csv(out_path, index=False)
    print("Saved predictions to:", out_path)

    return df, (r2_C, r2_D), (pr_C, pr_D), (mse_C, mse_D)


class DeepCNetD_Dataset(Dataset):
    """
    Two-condition dataset for DeepCNet-D.
    Returns for each gene:
        - distance & activity tensor for both conditions(feat_C, feat_D)
        - expression tensor for both conditions (expr_C, expr_D)
        - epigenetic embedding tensor for both conditions (epi_C, epi_D)
        - gene embedding for each condition
        - gene id
    """

    def __init__(
        self,
        activity_folder_Control,
        activity_folder_Disease,
        expression_folder_Control,
        expression_folder_Disease,
        enhancer_info_path,
        epi_path,
        n_enhancers=60,
        gene_embedding_Control=None,
        gene_embedding_Disease=None,
    ):
        # peak activity H5 (two conditions)
        self.act_C = h5py.File(os.path.join(activity_folder_Control, "enhancer_activity.h5"), "r")
        self.act_D = h5py.File(os.path.join(activity_folder_Disease, "enhancer_activity.h5"), "r")

        # enhancer-level epigenetic features (same for both conditions)
        self.epi_h5 = h5py.File(epi_path, "r")

        # enhancer meta info (distance)
        self.info_h5 = h5py.File(enhancer_info_path, "r")

        # expression
        self.expr_C = pd.read_csv(
            os.path.join(expression_folder_Control, "gene_expr.csv"),
            index_col="ENSID"
        )
        self.expr_D = pd.read_csv(
            os.path.join(expression_folder_Disease, "gene_expr.csv"),
            index_col="ENSID"
        )

        # options
        self.n_enhancers = n_enhancers
        self.emb_C = gene_embedding_Control
        self.emb_D = gene_embedding_Disease

    def __len__(self):
        return len(self.info_h5["ensid"])

    # ------------------------------------------------------
    # Helper — process one condition (Control or AD)
    # ------------------------------------------------------
    def _process_condition(self, idx, activity_h5, expr_df, embedding):
        ensid = self.info_h5["ensid"][idx].decode()

        # ---------- distance (61 values) ----------
        dist = self.info_h5["distance"][idx, 1:]
        pe_dist = np.concatenate([[0], dist / 1000]).flatten()

        # ---------- enhancer activity (61 values) ----------
        activity = activity_h5["enhancer_activity"][:, idx].T
        enh_act = activity[1:]  # skip promoter
        pe_act = np.concatenate([[0], enh_act]).flatten()

        # final PE feature (distance + activity)
        pe_feat = np.concatenate([pe_dist[:, np.newaxis], pe_act[:, np.newaxis]], axis=-1)
        pe_feat_tensor = torch.from_numpy(pe_feat[: self.n_enhancers + 1]).float()

        # ---------- epigenetic tensor (61 × epi_dim) ----------
        epi = self.epi_h5["epigenetics"][idx]
        epi_tensor = torch.from_numpy(epi).float()

        # ---------- expression ----------
        expr_val = float(expr_df.loc[ensid]["expression"])
        expr_tensor = torch.tensor([expr_val]).float()

        # ---------- gene embedding ----------
        if embedding is None:
            gene_emb = None
        else:
            gene_emb = embedding[idx]

        return pe_feat_tensor, expr_tensor, epi_tensor, gene_emb

    def __getitem__(self, idx):
        ensid = self.info_h5["ensid"][idx].decode()

        # control condition
        feat_C, expr_C, epi_C, emb_C = self._process_condition(
            idx, self.act_C, self.expr_C, self.emb_C
        )

        # disease condition
        feat_D, expr_D, epi_D, emb_D = self._process_condition(
            idx, self.act_D, self.expr_D, self.emb_D
        )

        return (
            feat_C, expr_C, epi_C, emb_C,   # Control block
            feat_D, expr_D, epi_D, emb_D,   # Disease block
            ensid
        )
