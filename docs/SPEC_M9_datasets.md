# M9 spec — dataset generalization: CIFAR-100 subclass, FashionMNIST, SVHN

Goal (plan §6.2, D3, §3.3): run the same audit on more datasets. Data root
`data/` already contains all of them (torchvision layout).

Files:

```
src/topotrace/cifar.py     # EDIT: add load_cifar100 + subclass split
src/topotrace/mnist.py     # EDIT: add load_fashion_mnist
src/topotrace/svhn.py      # new: load_svhn
src/topotrace/resnet.py    # EDIT: num_classes parameter
scripts/run_exp.py         # new generic runner
```

## Loader additions

- `load_fashion_mnist(root="data")` — torchvision FashionMNIST, same shape/
  normalization pattern as load_mnist (use its own mean/std: .2860/.3530).
- `load_svhn(root="data")` — torchvision SVHN split train/test (files
  train_32x32.mat / test_32x32.mat are directly in root; SVHN(root=...)
  finds them). X (n,3,32,32) float32 normalized mean .4377,.4438,.4728 /
  std .1980,.2010,.1970; y int64 (labels already 0-9 in torchvision).
- `load_cifar100(root="data")` — CIFAR100, same normalization as CIFAR-10.
- `make_subclass_forget_split(y_fine, fine_cls: int)` — forget = all samples
  of one fine class (plan D3; deleting one subclass of a superclass, e.g.
  fine class 30 'dolphin' within aquatic mammals). Lives in cifar.py.

## resnet.py edit

`ResNet18C(num_classes: int = 10)` and `train_resnet(..., num_classes=10)`
passing it to fresh inits. Nothing else changes (existing callers keep
working via defaults).

## run_exp.py — generic runner

`conda run -n tda python scripts/run_exp.py <dataset> <scenario> [n_seeds]`
- dataset in {cifar10, cifar100, svhn, fashionmnist}; scenario in
  {random, class}; results dir results/exp_<dataset>_<scenario>/.
- Registry per dataset: loader, train fn, embeddings fn, evaluate fn,
  num_classes, class-deletion target (cifar10: class 9; cifar100: fine
  class 30; svhn: class 9; fashionmnist: class 9), model kind
  (fashionmnist -> topotrace.cnn SmallCNN pipeline; others ->
  resnet with num_classes).
- Body identical to scripts/run_m4.py: probe via topotrace.mnist.make_probe,
  10-seed default conditions (original/retrain/retrain2/noop/finetune/
  neggrad/scrub/ssd), accuracy table, model state_dicts under models/,
  embeddings.npz, persistence + TRR metrics, metrics.json. Import and reuse
  run_m4's helpers if convenient, otherwise self-contained copy of that
  flow — prefer a shared function refactor ONLY if it stays simple.

Verify WITHOUT long GPU jobs (a training chain owns the GPU): import-check
everything, run loaders (print shapes), and one CPU smoke:
`run_exp.py fashionmnist random 1` with a temporary env override making it
tiny is NOT required — instead verify the runner logic by monkeypatching in
a tiny test script in the scratch dir: subset X/y to 3000 samples, epochs=1
via editing nothing (pass a --smoke flag on run_exp.py: n first 3000 train
samples, 1 epoch, 2 seeds, probe 100+100). --smoke must not write into
results/ (use results/_smoke_* and delete after). Keep code minimal.
