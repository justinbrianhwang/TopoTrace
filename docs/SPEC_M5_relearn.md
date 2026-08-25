# M5 spec — relearning experiment (plan §15.2, operational validation)

Goal: models with residual topology should relearn the forget set faster
than exact-retrained models. Produce per-model relearning curves from the
checkpoints saved under results/m4_random/models/.

New file: `scripts/relearn.py` (write ONLY this file).

CLI: `conda run -n tda python scripts/relearn.py results/m4_random
[--epochs 5] [--conditions retrain,retrain2,finetune,neggrad,scrub,ssd]`

Procedure per checkpoint {condition}_{seed}.pt (topotrace.resnet.ResNet18C
state dict):
1. Load CIFAR-10 (topotrace.cifar) and the scenario forget split exactly as
   scripts/eval_mia.py does (dir name suffix random -> 
   make_random_forget_split(y, .05, 0); class -> make_class_forget_split(y, 9);
   targeted/matched -> results/m4_splits.npz keys {scenario}_forget).
2. Identical relearning protocol for every model (plan §15.2): SGD
   lr=0.01, momentum=0.9, batch 128, CrossEntropy, train on the FULGET
   forget set only, seeded torch.Generator(seed=42+model seed) for
   shuffling, NO augmentation, NO scheduler. Record forget-set accuracy
   before step 0 and after every epoch (evaluate from topotrace.resnet).
3. Curve stats: auc = mean of the recorded accuracies (including step 0);
   epochs_to_95 = first epoch index reaching >= 0.95 (or None).

Output:
- results/<dir>/relearn.json:
  {condition: {"curves": [[acc...] per seed], "auc": [..], "epochs_to_95": [..]}}
- print per condition: mean±std auc, mean epochs_to_95 (ignoring None).

GPU note: a long training chain is running on the GPU. VERIFY on CPU with
--epochs 1 --conditions retrain --limit 1 (add a --limit N option: only the
first N checkpoints per condition) plus device fallback: accept a --device
flag (default None -> topotrace.resnet._default_device()). One CPU
relearning epoch on 2500 samples is ~1-2 min — acceptable. Full GPU run is
done later by the PM; do not run it yourself.

Typo note: "FULGET" above means FULL forget set. Keep code minimal.
