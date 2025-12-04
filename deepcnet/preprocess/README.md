# Preprocessing

This folder contains preprocessing code used to generate the inputs required by DeepCNet and DeepCNet-D. It provides scripts for producing:

- **Gene embeddings** derived from cell-specific gene activity matrices computed from scATAC data (generated using the `GeneActivity()` function from the **Signac** R package). 
- **Sequence embeddings** for promoters and enhancers, extracted using pretrained DeepSEA models.


## 1. Generate gene embeddings
```bash
python preprocess/get_gene_embedding.py \
    --gene_activity examples/example_data/GeneActivity_example_toy.csv \
    --celltype_file examples/example_data/cell_celltypes_example.txt \
    --weights_file examples/example_data/celltype_weight_matrix_alpha0.5_example.csv \
    --fold "fold_1" \
    --output_dir ${OUTPUT_DIR} \
    --anchor_celltype Exc \
```

## 2. Generate sequence-based epigenetic embeddings

This script converts 2000 bp FASTA sequences into epigenetic feature embeddings using a pretrained Expecto Beluga model.

Download Expecto Beluga model:
```bash
wget -O ./expecto/deepsea.beluga.pth https://zenodo.org/record/3402406/files/deepsea.beluga.pth?download=1 

```

To generate the epi embeddings:
```bash
python preprocess/get_epi_embedding.py \
    --fasta examples/example_data/epi_embedding/promoter_sequence_toy.fasta \
    --model {MODEL_PATH} \
    --output examples/example_data/epi_embedding/promoter_epi_embedding_toy.h5
```

To prepare inputs suitable for DeepCNet, the epigenetic embeddings for promoters and enhancers should be organized into an array of shape **(N, 1 + E, emb_dim)**, where *N* is the number of genes, *1* corresponds to the promoter, *E* is the number of enhancers, and *emb_dim* is the embedding dimension. An example of the required format can be found in `epi_embedding_toy.h5`.