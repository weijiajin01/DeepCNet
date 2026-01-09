# DeepCNet

DeepCNet is a multimodal deep learning framework that predicts gene expression and promoter-enhancer (p-e) interactions from multimodal inputs, including:

- enhancer accessibility  
- enhancer–promoter distance  
- sequence-based epigenetic embeddings
- gene embeddings learned from multi-cell-type autoencoder

DeepCNet provides two training modes:

- **DeepCNet-S** — single-condition model  
- **DeepCNet-D** — two-condition joint model with condition-specific attention bias (Control vs Disease)

Both models produce gene expression predictions and p-e attention maps.

---

## Installation

DeepCNet requires **Python 3.10** and **PyTorch ≥ 2.0**.

### 1. Create environment

```bash
conda env create -f environment.yml
conda activate deepcnet
```

## 2. Train DeepCNet-S (Single Condition)

DeepCNet-S trains a single-condition model using epigenetic embeddings, peak accessibility features, and precomputed gene embeddings. The model outputs a trained checkpoint, a CSV file of predicted gene expression, and (optionally) an H5 file containing the p-e attention matrix when `--extract_attn` is enabled.


```bash
python scripts/train_deepcnet_s.py \
    --cell Mic \
    --activity_folder examples/example_data/deepcnet/normal_Mic \
    --expression_folder examples/example_data/deepcnet/normal_Mic \
    --embeddings_path examples/example_data/deepcnet/normal_Mic/gene_embedding_toy.npy \
    --save_dir examples/example_outs/normal_Mic \
    --extract_attn
```

## 3. Train DeepCNet-D (Two Conditions)

DeepCNet-D jointly models two biological conditions (e.g., control vs. disease) using condition-specific peak accessibility features, gene expression profiles, gene embeddings, and shared promoter and enhancer epigenetic embeddings. The model outputs a trained two-condition checkpoint, predicted gene expression for both conditions, and (optionally) an H5 file containing condition-specific promoter–enhancer attention matrices when `--extract_attn` is enabled.


```bash
python scripts/train_deepcnet_d.py \
    --cell Mic \
    --activity_folder_Control examples/example_data/deepcnet/normal_Mic \
    --activity_folder_Disease examples/example_data/deepcnet/disease_Mic \
    --expression_folder_Control examples/example_data/deepcnet/normal_Mic \
    --expression_folder_Disease examples/example_data/deepcnet/disease_Mic \
    --embeddings_path_Control examples/example_data/deepcnet/normal_Mic/gene_embedding_toy.npy \
    --embeddings_path_Disease examples/example_data/deepcnet/disease_Mic/gene_embedding_toy.npy \
    --save_dir examples/example_outs/2cond_Mic \
    --extract_attn
```

## 4. Example Dataset

We provide a minimal toy dataset under `examples/example_data/` for running the full DeepCNet and DeepCNet-D training pipelines. The corresponding toy outputs are generated in `examples/example_outs/`.

## 5. Citation

If you use DeepCNet or DeepCNet-D in your research, please cite our work: