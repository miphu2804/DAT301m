# DESCN Notebook Remaining Audit

## Scope

This audit compares `descn_implementation.ipynb` with the local `DESCN/` source
and the KDD 2022 DESCN paper for the main ACIC/Lazada model comparison path.
Ablation tables are intentionally out of scope.

## Fixed / Aligned

- ACIC data now loads `DESCN/data/ACIC2019_epilepsy_dataset/syn_bin_set.5.train.npz`
  and `syn_bin_set.5.test.npz` instead of notebook-only synthetic data.
- The ACIC generator follows `SG_Generate_High_Dim_Binary/generate_simEpilepsy.R`
  Mod 4 and reproduces `psi0_mod4 = 0.2916273836`, matching the paper note
  `0.2916274`.
- The generated `.npz` files use the DESCN source contract:
  `x=(40000,178,5)` and labels/effects as `(40000,5)`.
- The five ACIC experiment dimensions are packed from independent contiguous
  R-generated blocks instead of repeating one block five times.
- AUUC/Qini now delegates to `sklift.metrics.qini_auc_score`, matching
  `DESCN/eval.py` and `DESCN/eval4real_data.py`.
- ACIC model selection uses lowest `sqrt_pehe`, matching the synthetic
  `eval.py` early-selection behavior. Lazada keeps best AUUC, matching
  `eval4real_data.py`.
- ACIC hyperparameters and loss weights for TARNet, CFR(MMD), X-network, and
  DESCN match the corresponding `DESCN/conf4models/ACIC2019/*.txt` files.
- The final comparison plot now follows the dataset-specific paper metrics:
  ACIC plots `sqrtPEHE` and `e_ATE`; Lazada plots AUUC and `e_ATT`.

## Remaining Differences

- The notebook is a TensorFlow/Keras reimplementation. The source repo is
  PyTorch. Architecture, main layer counts, output equations, and loss terms
  are mirrored, but exact numerical reproduction is not guaranteed.
- The original DESCN repo does not include a CSV-to-`.npz` manifest for ACIC.
  The current script uses exact Mod4 generation, then packs files `1-100` for
  train and `101-200` for test. This preserves the paper/source shape and DGP,
  but it is not provably the authors' original split.
- The paper Table 2 includes Causal Forest, BART, TARNet, CFRwass, CFRmmd,
  X-network, and DESCN. The notebook currently runs TARNet, CFR(MMD),
  X-network, and DESCN.
- The local repo contains code/configs for X-learner and CFRwass, plus a
  separate notebook for Causal Forest/BART on real data. The current notebook
  does not run X-learner, CFRwass, Causal Forest, or BART.
- CFR(MMD)/Wasserstein distances are approximated in TensorFlow. The source
  repo uses PyTorch utilities and `geomloss`; results can differ even when the
  loss weights match.
- Optimizer regularization is framework-dependent: the source uses
  `torch.optim.Adam(..., weight_decay=cfg.l2)`, while the notebook uses Keras
  Adam with `weight_decay=l2`.
- The notebook saves prediction artifacts in the public-code-style `.npz`
  layout, but it does not call `DESCN/eval.py` or `DESCN/eval4real_data.py`
  directly after training.

## Model Compare Coverage

For the main DESCN story, the notebook covers the most important neural
comparison path:

- TARNet
- CFR(MMD)
- X-network
- DESCN

For a full Table 2-style reproduction, add at least:

- CFR(Wass), supported by `DESCN/conf4models/ACIC2019/CFRwass.txt`
- X-learner, supported by `DESCN/x_learner_main.py`
- Causal Forest and BART, treated as external/classical baselines in the paper

## Unresolved Questions

- The exact authors' ACIC train/test CSV grouping is not recoverable from the
  local repo files.
- Whether to keep the notebook focused on the four neural DESCN-path models or
  expand it to full Table 2 coverage depends on the expected report scope.
