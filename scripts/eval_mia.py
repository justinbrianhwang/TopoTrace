"""Evaluate loss-based membership inference on saved checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.attacks import loss_mia_auc
from topotrace.mnist import make_class_forget_split, make_random_forget_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--model", choices=("resnet", "cnn"), default="resnet")
    args = parser.parse_args()

    models_dir = args.result_dir / "models"
    if not models_dir.is_dir():
        print(f"No models directory at {models_dir}; nothing to evaluate yet.")
        return

    if args.model == "cnn":
        from topotrace.cnn import SmallCNN as Model
        from topotrace.mnist import load_mnist as load_data
    else:
        from topotrace.cifar import load_cifar10 as load_data
        from topotrace.resnet import ResNet18C as Model

    X, y, X_test, y_test = load_data(str(ROOT / "data"))
    scenario = args.result_dir.name.rsplit("_", 1)[-1]
    if scenario == "class":
        forget_idx, _ = make_class_forget_split(y, 9)
    elif scenario == "random":
        forget_idx, _ = make_random_forget_split(y, .05, 0)
    else:
        parser.error("result directory name must end in '_random' or '_class'")

    aucs = {}
    for checkpoint in sorted(models_dir.glob("*.pt")):
        condition, _ = checkpoint.stem.rsplit("_", 1)
        model = Model()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                         weights_only=True))
        aucs.setdefault(condition, []).append(
            loss_mia_auc(model, X, y, forget_idx, X_test, y_test))

    for condition, values in aucs.items():
        print(f"{condition}: {np.mean(values):.4f}±{np.std(values):.4f}")
    (args.result_dir / "mia.json").write_text(json.dumps(aucs, indent=2))


if __name__ == "__main__":
    main()
