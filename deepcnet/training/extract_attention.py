import os
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_attention_single_cond(
    model,
    dataset,
    save_dir,
    save_name = "attn_single.h5",
    checkpoint_path=None,
    batch_size=32,
    device="cuda",
):
    """
    Extract attention matrices for DeepCNet (single-condition).
    
    H5 contains:
        attn_weights        (N, 1, 61)
        ensid               (N,)
    """
    save_path = os.path.join(save_dir,save_name)

    print(f"[Extract] Saving attention to: {save_path}")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(save_dir, f"fold_1_best_{model.name}_checkpoint.pt")
        print(f"[Extract] No checkpoint_path provided. Using default: {checkpoint_path}")
    else:
        print(f"[Extract] Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)

    key = "model_state" if "model_state" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])

    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True)

    # ---- Inspect one sample to infer shapes ----
    sample = dataset[0]
    try:
        dist_feat, expr, epi, emb, ensid = sample
    except ValueError:
        dist_feat, expr, epi, ensid = sample
        emb = None

    n_tokens = dist_feat.shape[0]             # promoter + enhancers (e.g., 61)
    n_layers = model.n_encoder
    n_heads  = model.n_head

    # ---- Create HDF5 file ----
    
    h5f = h5py.File(save_path, "w")
    dset = h5f.create_dataset(
        "attn_weights",
        shape=(0, 1, n_tokens),
        maxshape=(None, 1, n_tokens),
        dtype="float32",
    )

    # ---- Loop through dataset ----
    row_cursor = 0
    all_ensids = []
    with torch.no_grad():
        for batch in tqdm(loader):
            if len(batch) == 5:
                dist_feat, y_expr, epi_feat, ct_embed, eid = batch
                ct_embed = ct_embed.float().to(device)
            else:
                dist_feat, y_expr, epi_feat, eid = batch
                ct_embed = None

            # forward
            pred, attn_list = model(
                dist_feat=dist_feat.to(device),
                epi_feat=epi_feat.to(device),
                gene_embedding=ct_embed,
            )

            # Extract attention shape per layer: attn_list: list of (1, B, L, S)
            attn_array = torch.stack([attn[0, :, 0, :] for attn in attn_list])
            # attn_array shape = (layers, batch, tokens)

            # mean across layers
            attn_mean = attn_array.mean(dim=0)
            
            # add dim to match new shape: (batch, 1, tokens)
            attn_mean = attn_mean.unsqueeze(1)

            attn_np = attn_mean.cpu().numpy().astype("float32")

            # Append to HDF5
            new_rows = row_cursor + attn_np.shape[0]
            dset.resize((new_rows, 1, n_tokens))
            dset[row_cursor:new_rows, :, :] = attn_np
            row_cursor = new_rows

            all_ensids.extend(eid)

        dt = h5py.string_dtype(encoding='utf-8')
        h5f.create_dataset("ensid", data=np.array(all_ensids, dtype=dt))

    h5f.close()
    print(f"[Extract] Done. Saved to: {save_path}")


def extract_attention_two_cond(
    model,
    dataset,
    save_dir,
    save_name = "attn_twocond.h5",
    checkpoint_path=None,
    batch_size=32,
    device="cuda",
):
    """
    Extract attention for DeepCNet-D (two-condition)

    H5 contains:
        attn_control        (N, 1, 61)
        attn_disease        (N, 1, 61)
        ensid               (N,)
    """

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir,save_name)

    print(f"[Extract-2cond] Saving attention to: {save_path}")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(save_dir, f"fold_1_best_{model.name}_checkpoint.pt")
        print(f"[Extract] No checkpoint_path provided. Using default: {checkpoint_path}")
    else:
        print(f"[Extract] Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)

    key = "model_state" if "model_state" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])

    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ==========================
    # Create H5 file
    # ==========================
    with h5py.File(save_path, "w") as h5f:

        # attention shape (B, 3, 1, 61)
        dset_control = h5f.create_dataset(
            "attn_control",
            shape=(0, 1, model.n_enhancer + 1),
            maxshape=(None, 1, model.n_enhancer + 1),
            dtype="float32",
        )
        dset_disease = h5f.create_dataset(
            "attn_disease",
            shape=(0, 1, model.n_enhancer + 1),
            maxshape=(None, 1, model.n_enhancer + 1),
            dtype="float32",
        )

        # ==========================
        # Loop and extract attention
        # ==========================
        all_ensids = []

        with torch.no_grad():
            for batch in tqdm(loader):

                (
                    feat_C,
                    expr_C,
                    epi_C,
                    emb_C,
                    feat_D,
                    expr_D,
                    epi_D,
                    emb_D,
                    eid,
                ) = batch

                feat_C = feat_C.float().to(device)
                feat_D = feat_D.float().to(device)
                epi_C = epi_C.float().to(device)

                if isinstance(emb_C, torch.Tensor):
                    emb_C = emb_C.float().to(device)
                if isinstance(emb_D, torch.Tensor):
                    emb_D = emb_D.float().to(device)

                # Forward pass
                pred_C, pred_D, attn_C_list, attn_D_list, _, _ = model(
                    distfeat_Control=feat_C,
                    distfeat_Disease=feat_D,
                    epi_feat=epi_C,
                    embedding_Control=emb_C,
                    embedding_Disease=emb_D,
                )

                # ========== convert attention to numpy ==========
                def process_attn(attn_list):
                    attn_array = torch.stack([attn[0, :, 0, :] for attn in attn_list])
                    attn_mean =  attn_array.mean(dim=0)
                    attn_mean = attn_mean.unsqueeze(1)               
                    return attn_mean.cpu().numpy().astype(np.float32)

                attn_control_np = process_attn(attn_C_list)
                attn_disease_np = process_attn(attn_D_list)

                B = attn_control_np.shape[0]

                # append to HDF5
                dset_control.resize((dset_control.shape[0] + B, 1, model.n_enhancer + 1))
                dset_disease.resize((dset_disease.shape[0] + B, 1, model.n_enhancer + 1))

                dset_control[-B:] = attn_control_np
                dset_disease[-B:] = attn_disease_np

                all_ensids.extend(eid)

        # store ENSIDs
        dt = h5py.string_dtype(encoding='utf-8')
        h5f.create_dataset("ensid", data=np.array(all_ensids, dtype=dt))

    print(f"[Extract-2cond] Finished. Saved to: {save_path}")
