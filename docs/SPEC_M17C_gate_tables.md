# M17C — declared gate in code, layer fixes, full result matrix

Fixes findings 6, 9, 13, 18 of the final adversarial review. Edit ONLY:
scripts/analyze_m2.py, scripts/layer_profile.py, scripts/make_tables.py.

1. analyze_m2.py — implement the DECLARED conjunctive gate: a cell's gate
   is open iff permutation p < .05 AND the bootstrap CI of I_topo lies
   strictly above 0. Method tests run (and are stored under "methods")
   only for gate-open cells; store "gate_open": true/false per cell in
   analysis.json. Then RERUN on every results/* dir containing
   embeddings.npz.
2. layer_profile.py — (a) remove the duplicate: penultimate equals
   avg-pooled layer4, so keep layers stem, layer1..layer4 (label layer4 as
   "layer4 (=penultimate)") plus logits; (b) compute a seed-level
   bootstrap CI for each layer cell's I_topo and use the SAME conjunctive
   gate for hollow markers. Rerun on results/m4_random (embeddings_layers
   .npz exists; reuse it instead of re-extracting if straightforward).
3. make_tables.py —
   - Table 2 caption: state that TRR/alpha/eta come from the frozen-grid
     penultimate-H1 cell.
   - Table 3 caption: "Proximity-based oracle-equivalence decisions.
     Stars mark BH-significant DIFFERENCE tests (method vs oracle)."
   - Table 4 caption: note all variants are penultimate-H1.
   - NEW table6_matrix.tex: the complete scenario-by-cell evidence matrix.
     One row per (scenario, cell) over every results dir that has
     analysis.json among: m4_random1/m4_random/m4_random10/m4_class/
     m4_targeted/m4_matched/m4_random_pretrained/exp_cifar100_class/
     exp_fashionmnist_random/exp_fashionmnist_class/exp_svhn_random/
     exp_svhn_class/m2_class. Columns: scenario, cell, I_topo, gate p,
     gate open (y/n), anchor(retrain2) BH q, worst approximate-method BH
     q (BH computed within scenario over all stored method x cell raw
     p's; blank when gate closed). \scriptsize, longtable NOT allowed —
     if too long, split into two table environments (part 1/part 2).
   - Rerun make_tables.py after step 1's reruns finish (they run in this
     same task: do step 1 reruns yourself before generating tables).

Verify everything end-to-end; print the matrix. Keep code minimal.
