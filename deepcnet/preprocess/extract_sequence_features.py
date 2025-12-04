import torch
from torch import nn
import numpy as np
from Bio import SeqIO
import math
import h5py

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super().__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, x):
        out = []
        for module in self._modules.values():
            out.append(module(x))
        return out if out else x

class Lambda(LambdaBase):
    def forward(self, x):
        return self.lambda_func(self.forward_prepare(x))

class Beluga(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),
                              nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),
                              nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

def load_beluga_model(model_path):
    model = Beluga()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def encode_seqs(seqs, inputsize=2000):
    arr = np.zeros((len(seqs), 4, inputsize), np.bool_)
    mp = {
        'A':[1,0,0,0],'G':[0,1,0,0],'C':[0,0,1,0],'T':[0,0,0,1],
        'N':[0,0,0,0],'H':[0,0,0,0],'-':[0,0,0,0],
        'a':[1,0,0,0],'g':[0,1,0,0],'c':[0,0,1,0],'t':[0,0,0,1]
    }
    for i, s in enumerate(seqs):
        cline = s[(len(s)-inputsize)//2 : len(s)-(len(s)-inputsize)//2]
        for j, c in enumerate(cline):
            arr[i, :, j] = mp.get(c, [0,0,0,0])
    return arr.astype(np.float32)

def fasta_to_embedding(fasta_path, model_path, output_h5, batch_size=64, input_size=2000):
    seqs = [str(r.seq) for r in SeqIO.parse(fasta_path, "fasta")]
    onehot = encode_seqs(seqs, inputsize=input_size)
    model = load_beluga_model(model_path)

    with h5py.File(output_h5, "w") as f:
        dset = None
        for i in range(0, len(onehot), batch_size):
            batch = torch.from_numpy(onehot[i:i+batch_size]).unsqueeze(2)
            emb = model(batch).detach().numpy().copy()

            if dset is None:
                dset = f.create_dataset(
                    "epigenetics",
                    data=emb,
                    maxshape=(None, emb.shape[1]),
                    chunks=True
                )
            else:
                old = dset.shape[0]
                dset.resize(old + emb.shape[0], axis=0)
                dset[old:old+emb.shape[0]] = emb
