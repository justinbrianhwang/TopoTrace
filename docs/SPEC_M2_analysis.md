# M2 analysis spec — layer/homology sweep + statistics (plan §14, §16)

Problem: at 10 seeds, penultimate-H1 shows I_topo < 0 (seed variation >
deletion effect). Find where (layer × homology) the class-deletion imprint
is statistically real before interpreting any TRR.

Inputs: `results/m2_class/embeddings.npz` (written by run_m2.py, may not
exist yet while the run finishes). Keys are `{condition}_{seed}_{layer}`
with condition in {original, retrain, retrain2, noop, finetune, neggrad,
scrub, ssd}, seed 0..9, layer in {penultimate, logits}; each value is
(600, d) float32 probe embeddings.

New files (write ONLY these; topology.py: only the stated change):

```
src/topotrace/stats.py       # new
scripts/analyze_m2.py        # new
src/topotrace/topology.py    # EDIT: generalize vectorize() to a dim param
```

## topology.py change

`vectorize(dgms, imager, dim: int = 1)` — persistence image of `dgms[dim]`
instead of hard-coded H1. Default keeps current behavior; do not change
anything else in the file. For dim=0, drop the diagonal H0 point at
(0, cap) if death==birth, and pass the (k,2) birth/death array exactly as
for H1 (persim handles birth=0 rows).

## stats.py

```python
def energy_distance(A: list[np.ndarray], B: list[np.ndarray]) -> float:
    """2*mean d(a,b) - mean d(a,a') - mean d(b,b'), euclidean, off-diagonal
    means for the within terms."""

def permutation_pvalue(A, B, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sample permutation test with energy_distance as statistic;
    pool A+B, shuffle labels (np rng), p = (1 + #{perm >= obs}) / (1 + n_perm)."""

def bootstrap_ci(A, B, stat_fn, n_boot: int = 1000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of stat_fn(A_resampled, B_resampled), resampling each
    list with replacement (same sizes)."""
```

Demo in `__main__`: two clearly separated Gaussian vector samples give
p < 0.01; identical distributions give p > 0.1. Assert both.

## analyze_m2.py

For each layer in (penultimate, logits) and hom_dim in (0, 1):
1. diagrams per model: chordal_distance_matrix -> persistence_diagrams
   (reuse topotrace.topology; compute each model's diagrams once per layer
   and reuse across hom_dims).
2. one imager per (layer, hom_dim) fit on all diagrams of that dim
   (make_imager works on any dim's diagram list), vectors via
   vectorize(..., dim=hom_dim).
3. Imprint: I_topo = D_OR - D_RR from topotrace.metrics distances
   (v_O = original vectors, v_R = retrain vectors);
   permutation_pvalue(v_O, v_R); bootstrap_ci for I_topo
   (stat_fn computing I_topo from resampled v_O, v_R).
4. Print one row per (layer, hom): D_RR, D_OR, I_topo, CI, p.
5. For cells with p < 0.05: print per-method TRR/alpha/eta table
   (trr_metrics) for noop, retrain2, finetune, neggrad, scrub, ssd,
   plus permutation_pvalue(v_method, v_R) per method.
6. Save everything printed also to results/m2_class/analysis.json.

CLI: `conda run -n tda python scripts/analyze_m2.py [results/m2_class]`.
sys.path bootstrap like scripts/run_m2.py.

Verify: run `conda run -n tda python src/topotrace/stats.py` (must pass).
If embeddings.npz does not exist yet, verify analyze_m2.py end-to-end on a
small synthetic npz you write to the scratch dir with the same key format
(e.g. 8 conditions x 3 seeds x 2 layers, 60x8 random embeddings), then
delete it. Keep code minimal.
