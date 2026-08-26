# M23 — certified removal probe: does the audit pass a provably removed model?

This is the sharpest available test of whether the audit tracks genuine
removal or merely training-path divergence. Certified removal
(Guo et al., ICML 2020) is guaranteed only for a convex model, so we use
their deep-network protocol: a FROZEN feature extractor plus an
L2-regularized multinomial logistic regression on its features, and we
audit at the logits layer, where the guarantee applies.

New file ONLY: `scripts/certified_probe.py`. CIFAR-10 random 5%, GPU ok.

Setup (all conditions share ONE frozen extractor = the original ResNet-18
of seed 0, results/m4_random/models/original_0.pt; extract 512-d
penultimate features once for train, test, and the saved probe indices):

1. Randomness source = the objective perturbation b of certified removal
   (NOT initialization). For draw t = 0..9, sample
   b_t ~ N(0, sigma^2 I) with sigma = 0.01 (record it), and define the
   objective L_S(w) = sum_{i in S} CE(w, phi_i, y_i) + (lambda/2)||w||^2
   + b_t . w, with lambda = 1e-2 * |S|. Solve to convergence
   (full-batch LBFGS or Newton; torch on GPU is fine).
2. Conditions, each with 10 draws:
   - `original`: minimizer of L_D (all training data)
   - `retrain` (oracle): minimizer of L_{D_r} (retain only), draws 0..9
   - `retrain2` (held-out anchor): minimizer of L_{D_r}, draws 10..19
   - `certified`: from `original` of draw t, apply the Newton removal step
     w' = w + H^{-1} * grad, where H is the Hessian of L_{D_r} at w and
     grad = the gradient contributed by the forget points (so that w'
     approximates the minimizer of L_{D_r}); report the gradient-residual
     norm ||grad L_{D_r}(w')|| as the certificate proxy.
   - `noop`: the `original` weights unchanged.
3. Fingerprints: logits on the probe = phi_probe @ W^T for each model;
   then the standard audit, frozen protocol: chordal distances,
   H0/H1 persistence diagrams, grid fitted on original+retrain only,
   I_topo + bootstrap CI + gate permutation p; method tests for
   certified / noop / retrain2 with BH over the family.
4. Also report utility: retain/forget/test accuracy per condition, and
   the mean L2 distance ||w_certified - w_retrain|| versus
   ||w_retrain - w_retrain'|| (oracle-internal spread) — the parameter-space
   analogue of the topological comparison.
5. Save results/m4_random/certified_probe.json; print a compact table.

The scientifically interesting outcomes are both informative and must be
reported honestly whichever occurs: the audit RETAINS `certified` while
rejecting `noop` (evidence the audit tracks removal), or it rejects both
(bounding what a rejection means). Do not tune anything to force either.

Verify end-to-end. Keep code minimal.
