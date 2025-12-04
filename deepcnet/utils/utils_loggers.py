import torch
import pandas as pd
import numpy as np


def get_lr(optimizer):
    """Return current learning rate."""
    for group in optimizer.param_groups:
        return group["lr"]


class Logger:
    """Simple metric logger."""

    def __init__(self, columns, verbose=False):
        self.columns = columns
        self.verbose = verbose

    def start(self):
        """Initialize log storage."""
        self.data = {c: [] for c in self.columns}
        if self.verbose:
            print("\t".join(self.columns))

    def add(self, values):
        """Append a row of metrics."""
        assert len(values) == len(self.columns)
        for c, v in zip(self.columns, values):
            self.data[c].append(v)

        if self.verbose:
            formatted = [round(v, 4) if isinstance(v, float) else v for v in values]
            print("\t".join(map(str, formatted)))

    def save(self, path):
        """Save logged metrics to file."""
        pd.DataFrame(self.data).to_csv(path, sep="\t", index=False)


class EarlyStopping:
    """Stop training if validation loss does not improve."""

    def __init__(self, patience=3, delta=0.0, verbose=False, path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path

        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.min_loss = np.Inf

    def __call__(self, current_loss, model, epoch_idx):
        score = -current_loss

        if self.best_score is None:
            self.best_score = score
            self._save(current_loss, model, epoch_idx)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self._save(current_loss, model, epoch_idx)
            self.counter = 0

    def _save(self, loss, model, epoch_idx):
        """Save state dict when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss improved ({self.min_loss:.6f} to {loss:.6f}). Saving checkpoint.")

        ckpt = {
            "epoch": epoch_idx,
            "model_state": model.state_dict(),
            "val_loss": loss,
        }
        torch.save(ckpt, self.path)
        print(f"Saved checkpoint: {self.path}")

        self.min_loss = loss