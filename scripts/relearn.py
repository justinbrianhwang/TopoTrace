"""Measure forget-set relearning curves for M4 checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topotrace.cifar import load_cifar10
from topotrace.mnist import make_class_forget_split, make_random_forget_split
from topotrace import resnet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--conditions", default="retrain,retrain2,finetune,neggrad,scrub,ssd")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    parser.add_argument("--record-steps", type=int, default=0,
                        help="also record forget acc every N optimizer steps "
                             "during the FIRST epoch (0 = off)")
    args = parser.parse_args()

    X, y, _, _ = load_cifar10()
    scenario = args.result_dir.name.rsplit("_", 1)[-1]
    if scenario.startswith("random"):
        forget_idx, _ = make_random_forget_split(y, float(scenario[6:] or 5) / 100, 0)
    elif scenario == "class":
        forget_idx, _ = make_class_forget_split(y, 9)
    elif scenario in ("targeted", "matched"):
        forget_idx = np.load(ROOT / "results" / "m4_splits.npz")[f"{scenario}_forget"]
    else:
        parser.error("result directory must end in random, class, targeted, or matched")

    device = torch.device(args.device or resnet._default_device())
    resnet._default_device = lambda: device
    X_forget = torch.as_tensor(X[forget_idx], dtype=torch.float32)
    y_forget = torch.as_tensor(y[forget_idx], dtype=torch.long)
    output = {}

    for condition in args.conditions.split(","):
        checkpoints = sorted(
            (args.result_dir / "models").glob(f"{condition}_*.pt"),
            key=lambda path: int(path.stem.rsplit("_", 1)[1]),
        )[:args.limit]
        curves = []
        step_curves = []
        for checkpoint in checkpoints:
            seed = int(checkpoint.stem.rsplit("_", 1)[1])
            model = resnet.ResNet18C().to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
            loss_fn = torch.nn.CrossEntropyLoss()
            generator = torch.Generator().manual_seed(42 + seed)
            curve = [resnet.evaluate(model, X, y, forget_idx)]
            step_curve = [curve[0]]
            step = 0
            for epoch in range(args.epochs):
                model.train()
                for batch in torch.randperm(len(y_forget), generator=generator).split(128):
                    optimizer.zero_grad()
                    loss_fn(model(X_forget[batch].to(device)), y_forget[batch].to(device)).backward()
                    optimizer.step()
                    step += 1
                    if args.record_steps and epoch == 0 and step % args.record_steps == 0:
                        step_curve.append(resnet.evaluate(model, X, y, forget_idx))
                        model.train()
                curve.append(resnet.evaluate(model, X, y, forget_idx))
            curves.append(curve)
            if args.record_steps:
                step_curves.append(step_curve)

        auc = [float(np.mean(curve)) for curve in curves]
        epochs_to_95 = [next((i for i, acc in enumerate(curve) if acc >= .95), None)
                        for curve in curves]
        output[condition] = {"curves": curves, "auc": auc, "epochs_to_95": epochs_to_95}
        if args.record_steps:
            output[condition]["step_curves"] = step_curves
        reached = [epoch for epoch in epochs_to_95 if epoch is not None]
        mean_epoch = f"{np.mean(reached):.2f}" if reached else "None"
        print(f"{condition}: auc {np.mean(auc):.4f}±{np.std(auc):.4f}, "
              f"epochs_to_95 {mean_epoch}")

    (args.result_dir / "relearn.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
