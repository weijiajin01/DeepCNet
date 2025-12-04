#!/usr/bin/env python3
import argparse
import os
import sys

from deepcnet.preprocess.extract_sequence_features import fasta_to_embedding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--input_size", type=int, default=2000)

    args = parser.parse_args()

    fasta_to_embedding(
        fasta_path=args.fasta,
        model_path=args.model,
        output_h5=args.output,
        batch_size=args.batch_size,
        input_size=args.input_size
    )

if __name__ == "__main__":
    main()
