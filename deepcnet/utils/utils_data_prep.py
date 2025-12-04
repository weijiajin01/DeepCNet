import h5py
import numpy as np
import pandas as pd
import torch


def load_gene_embeddings(path):
    arr = np.load(path)
    print(f"Loaded gene embeddings: {path}")
    return torch.tensor(arr, dtype=torch.float32)


def build_fold_indices(enhancer_info_path, split_df, fold_col):
    """
    Map ENSIDs in enhancer_info.h5 to row indices in split_df.
    Returns index arrays for train/valid/test subsets.

    Safety: intersection() ensures no KeyError if some ENSIDs are missing.
    """

    # Load ENSIDs in the exact dataset order
    with h5py.File(enhancer_info_path, "r") as h5:
        ensids = [eid.decode() for eid in h5["ensid"][:]]

    # Create ENSID → row index mapping
    idx_map = pd.Series(np.arange(len(ensids)), index=ensids)

    valid_ids = split_df.index.intersection(idx_map.index)

    # Extract per-split indices
    train_idx = idx_map.loc[valid_ids[split_df.loc[valid_ids, fold_col] == "train"]].to_numpy()
    valid_idx = idx_map.loc[valid_ids[split_df.loc[valid_ids, fold_col] == "valid"]].to_numpy()
    test_idx  = idx_map.loc[valid_ids[split_df.loc[valid_ids, fold_col] == "test"]].to_numpy()

    return train_idx, valid_idx, test_idx







