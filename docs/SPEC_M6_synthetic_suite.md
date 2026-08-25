# M6 spec — full synthetic benchmark suite (plan §6.1)

Extend M1 to the remaining ground-truth-topology benchmarks. Everything is
CPU-cheap MLP work; reuse topotrace.{models,train,topology,metrics,stats}.

Files:

```
src/topotrace/synthetic.py    # EDIT: add the four generators below
scripts/run_m1_suite.py       # new runner
```

## synthetic.py additions (keep existing functions unchanged)

Each returns (X float32 (n,2), y int64) with class 1 = a Gaussian
"distractor" blob far from the structure (so the task is learnable), and a
companion split function returning (forget_idx, retain_idx). Sizes ~500
structure + 250 blob, noise seeded via default_rng(seed).

1. `make_satellites(seed)` / `split_satellites(X, y)` — class 0: three
   Gaussian clusters (std .1) at triangle vertices radius 1 plus a SMALL
   satellite cluster (60 pts, std .05) at (2, 0). Forget = satellite
   cluster members (track indices when generating; recompute by proximity
   to (2,0) < .5). Expected change: H0 component count/merge scale.
2. `make_bridge(seed)` / `split_bridge(X, y)` — class 0: two clusters at
   (±1, 0) plus 60 "bridge" points uniform on the segment between them
   (y-jitter .03). Forget = bridge points (|x|<.6 & |y|<.15). Expected:
   H0 merge scale jumps.
3. `make_figure_eight(seed)` / `split_figure_eight(X, y)` — class 0: two
   tangent unit circles centered (-1,0),(1,0), noise .05. Forget = arc of
   the RIGHT circle with angle in [-pi/4, pi/4) about its center.
   Expected: one of two H1 bars weakens.
4. `make_canary(seed)` / `split_canary(X, y)` — labels depend ONLY on a
   linear boundary: class = (x0 > 0). Class-0 region additionally contains
   a ring (radius .6, center (-1.2, 0), 200 pts) among 300 background
   points uniform in [-2.2,-.2]x[-1,1]; class 1 mirrored background on the
   right, no ring. Forget = ring points (distance to ring circle < .12).
   Expected: topology changes with almost no output change (canary).

## run_m1_suite.py

For each benchmark in {ring (existing make_ring_dataset/make_forget_split),
satellites, bridge, figure_eight, canary}:
- 10 seeds; conditions: original, retrain (1000+s), retrain2 (3000+s),
  noop, finetune (50 epochs from original, 2000+s) — exactly the M1 recipe
  via topotrace.train.train.
- probe = all class-0 points; embeddings h2; chordal distances; H0+H1
  diagrams; per-dim persistence-image vectors (make_imager per benchmark
  per dim on all diagrams of that dim).
- per benchmark and hom dim: I_topo, permutation p (topotrace.stats,
  original vs retrain vectors), TRR/alpha/eta for noop/retrain2/finetune
  (topotrace.metrics), plus mean test-accuracy of original models on a
  held-out sample of the same distribution (generate with seed 777).
- ALSO for canary only: report mean |forget-set prediction agreement|
  between original and retrain (fraction of identical argmax) to show
  outputs barely change.
- print one compact table; save results/m1_suite/metrics.json.
- figure: results/m1_suite/figure_suite.png — one row per benchmark:
  scatter of the dataset (forget points highlighted) + H1 (or H0 for
  satellites/bridge) persistence diagrams for original/retrain seed 0.

Verify by running the full suite: `conda run -n tda python
scripts/run_m1_suite.py` (~10-20 min CPU). A big GPU job is running — keep
everything on CPU (torch default device cpu is fine for MLPs).
Keep code minimal.
