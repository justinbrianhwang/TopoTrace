"""Build CIFAR-10 topology-targeted + matched forget splits (plan D5/D6).

Run: conda run -n tda python scripts/make_m4_splits.py
Output: results/m4_splits.npz
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.cifar import load_cifar10
from topotrace.resnet import get_embeddings, train_resnet
from topotrace.targeted import make_matched_split, make_targeted_split

X, y, _, _ = load_cifar10(str(ROOT / "data"))
tf, tr = make_targeted_split(X, y, n_forget=2500, train_fn=train_resnet,
                             embed_fn=get_embeddings)
mf, mr = make_matched_split(y, tf)
np.savez(ROOT / "results" / "m4_splits.npz", targeted_forget=tf,
         targeted_retain=tr, matched_forget=mf, matched_retain=mr)
print("targeted:", np.bincount(y[tf], minlength=10))
print("matched: ", np.bincount(y[mf], minlength=10))
