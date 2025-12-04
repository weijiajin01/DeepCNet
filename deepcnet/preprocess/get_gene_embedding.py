#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import torch

from deepcnet.preprocess.gene_embedding import (
    set_seed,
    MultiCelltypeGeneAE,
    make_dataloaders,
    train_gene_autoencoder,
    extract_embeddings,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train multi-celltype gene autoencoder.")

    parser.add_argument("--gene_activity", required=True, type=str,
                        help="Path to the GeneActivity.csv file. Rows are genes and columns are cells, generated using Signac::GeneActivity.")

    parser.add_argument("--cv_split", type=str, default="examples/example_data/split_example.csv",
    help="CSV file defining train/val/test split (default: toy example split)")

    parser.add_argument("--celltype_file", required=True, type=str,
                        help="Path to a text file mapping each cell (column in GeneActivity.csv) to a cell type label.")

    parser.add_argument("--weights_file", required=True, type=str,
                        help="Path to the cell-type similarity weight matrix (CSV). Used to guide the multi–cell-type embedding fusion.")

    parser.add_argument("--fold", required=True, type=str,
                        help="Fold name, e.g., fold_1")

    parser.add_argument("--output_dir", required=True, type=str,
                        help="Directory to save model and embeddings")

    parser.add_argument("--anchor_celltype", default=None, type=str,
                        help="Anchor cell type used for the similarity regularization term.")

    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)

    return parser.parse_args()


def load_celltypes(celltype_file):
    df = pd.read_csv(celltype_file, index_col=0)
    cell_types = df.iloc[:, 0].tolist()
    unique_types = sorted(set(cell_types))
    mapping = {ct: i for i, ct in enumerate(unique_types)}
    ids = torch.tensor([mapping[ct] for ct in cell_types], dtype=torch.long)
    return ids, mapping, unique_types


def load_split_data(gene_activity_path, cv_split_path):
    gene_activity = pd.read_csv(gene_activity_path, index_col=0)
    cv_split = pd.read_csv(cv_split_path).set_index("Gene_name")

    common = gene_activity.index.intersection(cv_split.index)
    gene_activity = gene_activity.loc[common]
    cv_split = cv_split.loc[common]

    return gene_activity, cv_split


def main():
    args = parse_arguments()
    set_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    gene_activity, cv_split = load_split_data(
        args.gene_activity,
        args.cv_split
    )

    celltype_ids, celltype_to_id, unique_celltypes = load_celltypes(
        args.celltype_file
    )

    weights_df = pd.read_csv(args.weights_file, index_col=0)

    fold = args.fold
    train_genes = cv_split[cv_split[fold] == "train"].index
    val_genes = cv_split[cv_split[fold] == "valid"].index
    test_genes = cv_split[cv_split[fold] == "test"].index

    X_train = gene_activity.loc[train_genes].values
    X_val = gene_activity.loc[val_genes].values
    X_test = gene_activity.loc[test_genes].values

    train_loader, val_loader, test_loader = make_dataloaders(
        X_train, X_val, X_test, batch_size=args.batch_size
    )

    num_cells = celltype_ids.numel()
    num_celltypes = len(unique_celltypes)

    model = MultiCelltypeGeneAE(
        num_cells=num_cells,
        celltype_ids=celltype_ids,
        num_celltypes=num_celltypes,
        ct_latent_dim=8,
        final_latent_dim=32,
        dropout=0.1,
        use_ct_decoders=False
    )

    model_path = os.path.join(args.output_dir, f"geneAE_{fold}.pth")

    trained_model, history = train_gene_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        celltype_anchor=args.anchor_celltype,
        celltype_to_id=celltype_to_id,
        weights_df=weights_df,
        lambda_anchor=1.0,
        beta_ct=0.25,
        lr=args.lr,
        num_epochs=args.epochs,
        patience=args.patience,
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=True
    )

    X_all = np.vstack([X_train, X_val, X_test])
    embeddings = extract_embeddings(trained_model, X_all)

    all_genes = np.concatenate([train_genes, val_genes, test_genes])
    embedding_df = pd.DataFrame(embeddings, index=all_genes)

    final_embeddings = pd.DataFrame(
        np.zeros((len(cv_split), embeddings.shape[1]), dtype=np.float32),
        index=cv_split.index
    )

    for gene in embedding_df.index:
        final_embeddings.loc[gene] = embedding_df.loc[gene]

    emb_out = os.path.join(args.output_dir, f"gene_embeddings_{fold}.npy")
    np.save(emb_out, final_embeddings.values.astype("float32"))


if __name__ == "__main__":
    main()
