import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Subset
from datetime import datetime

from deepcnet.models.deepcnet import DeepCNet
from deepcnet.training.utils_deepcnet import DeepCNetDataset, train, test
from deepcnet.utils.utils_data_prep import load_gene_embeddings, build_fold_indices


# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser("Train DeepCNet single-condition model")

    parser.add_argument("--cell", type=str, required=True)
    parser.add_argument("--fold_idx", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--n_enhancers", type=int, default=60)
    parser.add_argument('--n_enc_layer', type=int, help='Layers of encoder', default=3)

    parser.add_argument("--activity_folder", type=str, required=True)
    parser.add_argument("--expression_folder", type=str, required=True)
    parser.add_argument("--enhancer_info_path", default="examples/example_data/deepcnet/enhancer_info_toy.h5",
    help="Path to enhancer–gene mapping file (default: toy example)")
    parser.add_argument("--epi_path", type=str, default="examples/example_data/epi_embedding/epi_embedding_toy.h5",
    help="Path to epigenetic embedding file (default: toy example)")

    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--split_csv", type=str, default="examples/example_data/split_example.csv",
    help="CSV file defining train/val/test split (default: toy example split)")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--extract_attn", action="store_true",
                    help="If set, extract attention matrices after testing.")
    return parser.parse_args()


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------
def main():
    args = parse_arguments()
    extract_attn = args.extract_attn
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n### Training DeepCNet for cell: {args.cell} ###\n")

    # ---------------------------
    # Load gene embeddings
    # ---------------------------
    gene_embeddings = load_gene_embeddings(args.embeddings_path)

    # ---------------------------
    # Load cross-validation split
    # ---------------------------
    split_df = pd.read_csv(args.split_csv, index_col=0)

    # ---------------------------
    # Build fold indices
    # ---------------------------
    fold = f"fold_{args.fold_idx}"
    train_idx, valid_idx, test_idx = build_fold_indices(
        enhancer_info_path=args.enhancer_info_path,
        split_df=split_df,
        fold_col=fold,
    )

    # ---------------------------
    # Build dataset
    # ---------------------------
    full_dataset = DeepCNetDataset(
        activity_folder=args.activity_folder,
        expression_folder=args.expression_folder,
        enhancer_info_path=args.enhancer_info_path,
        epi_path=args.epi_path,
        n_enhancers=args.n_enhancers,
        gene_embedding=gene_embeddings,
    )

    train_ds = Subset(full_dataset, train_idx)
    valid_ds = Subset(full_dataset, valid_idx)
    test_ds  = Subset(full_dataset, test_idx)

    print(f"Train: {len(train_ds)}  Valid: {len(valid_ds)}  Test: {len(test_ds)}")

    # ---------------------------
    # Build model
    # ---------------------------
    model = DeepCNet(
        n_encoder=args.n_enc_layer,
        out_dim=64,
        head=4,
        embedding_dim=gene_embeddings.shape[1],
        epi_dim=2002,
        n_enhancer=args.n_enhancers,
        device=device,
    ).to(device)

    print(f"Model_name: {model.name}")

    # ---------------------------
    # Prepare save path
    # ---------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d-%H")
    save_dir = os.path.join(args.save_dir, f"{args.cell}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------
    # Train
    # ---------------------------
    train(
        model=model,
        dataset=train_ds,
        valid_dataset=valid_ds,
        fold_idx=args.fold_idx,
        save_dir=save_dir,
        lr=args.lr,
        batch_size=args.batch_size,
        EPOCHS=args.epochs,
        device=device,
    )

    # ---------------------------
    # Test
    # ---------------------------
    test(
        model=model,
        fold_idx = args.fold_idx,
        test_ds=test_ds,
        save_dir=save_dir,
        batch_size=args.batch_size,
        device=device,
    )

    print("\n### Finished training + testing ###\n")

    # ---------------------------
    # Extract attention (optional)
    # ---------------------------
    if extract_attn:
        from deepcnet.training.extract_attention import extract_attention_single_cond

        print("\n[Extract Attention] Running attention extraction...")

        extract_attention_single_cond(
            model=model,
            dataset=full_dataset,     # Use full dataset
            save_dir=save_dir,
            save_name=f"attn_fold{args.fold_idx}.h5",
            batch_size=args.batch_size,
            device=device,
        )
if __name__ == "__main__":
    main()
