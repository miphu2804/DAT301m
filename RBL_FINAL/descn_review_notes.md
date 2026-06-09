# DESCN Notebook Review Notes

Scope: checked `RBL_FINAL/descn_implementation.ipynb` against `RBL_FINAL/2207.09920v3 (1).pdf` and the local `DESCN/` implementation. The TensorFlow migration itself is intentionally excluded from the mismatch list per request.

## Fixed in the notebook

1. Propensity BCE used the wrong argument order.
   - Source: `DESCN/main.py` trains `BCEWithLogitsLoss(p_prpsy_logit, t)`.
   - Notebook issue: `bce_logit_w()` expected `(labels, logits, positive_weight)` but was called with prediction/logit first.
   - Fix: calls now pass treatment labels first and propensity logits second in both `train_step()` and `evaluate()`.

2. ESTR/ESCR BCE used the wrong argument order.
   - Paper Eq. 4 defines labels as `y_i & w_i` for ESTR and `y_i & (1-w_i)` for ESCR, with model predictions `mu1*pi` and `mu0*(1-pi)`.
   - Notebook issue: weighted BCE calls passed predictions first and labels second.
   - Fix: calls now pass labels first, predictions second.

3. `evaluate()` referenced an undefined variable.
   - Notebook issue: `prpsy_logit` was used even though the model output variable was named `p_logit`.
   - Fix: renamed model outputs descriptively and uses `propensity_logit` consistently.

4. `e_ATT` used the production/RCT observed ATT formula even when synthetic ground truth `tau` was available.
   - Paper/source distinction: synthetic evaluation uses true treatment effect for PEHE/ATE; production without true tau uses observed treated-control response difference for ATT.
   - Fix: when `treatment_effect_true` is provided, `e_att` compares predicted ITE on treated units against true `tau` on treated units. The observed ATT fallback remains only for data without true tau.

5. Variable names used shorthand.
   - Examples: `X_tr`, `Y_tr`, `T_tr`, `E_tr`, `x_vl`, `idx_tr`, `m`, `hist`, `bl_pehe`, `p_m`.
   - Fix: renamed visible notebook variables to descriptive names such as `features_train`, `outcomes_train`, `treatments_train`, `randomized_flags_train`, `validation_indices`, `tarnet_model`, `history`, and `baseline_sqrt_pehe`.

6. IPM helper functions were not safe in graph execution.
   - Notebook issue: the old `wass()`/`mmd()` helpers used Python conditionals on TensorFlow tensor shapes.
   - Fix: replaced them with TensorFlow-safe helper functions using `tf.cond` and descriptive names. CFR(MMD) still uses the notebook's compact IPM approximation rather than reproducing full PyTorch GeomLoss behavior.

## Still intentionally not a full paper reproduction

1. Dataset is a toy synthetic generator.
   - Paper uses ACIC 2019 Epilepsy-derived synthetic data with 178 covariates and a Lazada production dataset with 83 covariates.
   - Notebook uses 50 generated covariates and smaller train/test sizes for a runnable demonstration.

2. Paper baselines are broader.
   - Paper compares X-learner (NN), Causal Forest, BART, TARNet, CFRwass, CFRmmd, X-network, and DESCN.
   - Notebook only trains TARNet, CFR(MMD), X-network, and DESCN.

3. Reported numbers cannot be compared directly to Table 2.
   - The notebook data, IPM helper, and framework differ from the paper experiment setup.
   - The table in the notebook should be read as a sanity comparison across implemented variants, not as a reproduction of KDD Table 2.

4. Hyperparameters are demonstration-scale.
   - Paper/source configs vary by dataset: ACIC uses batch size 500 and 15 epochs; Lazada uses batch size 5000 and 5 epochs.
   - Notebook keeps batch size 500 and 15 epochs but uses the smaller toy dataset and the local compact model dimensions.
   - The X-network and DESCN loss weights are kept close to the local Lazada-style configs in `DESCN/conf4models/lzd_real_data/`, not the ACIC configs.

## Checked and kept

1. ESN equations remain aligned: `ESTR = propensity * treated_response_probability`, `ESCR = (1 - propensity) * control_response_probability`.
2. DESCN loss structure remains aligned: no direct `L_TR/L_CR` for DESCN, using `L_pi + L_ESTR + L_ESCR + L_CrossTR + L_CrossCR`.
3. Cross heads remain in logit space: `sigmoid(control_response_logit + pseudo_effect_logit)` and `sigmoid(treated_response_logit - pseudo_effect_logit)`.
4. Final ITE prediction remains `treated_response_probability - control_response_probability`, matching the local source code output `p_h1 - p_h0`.

## Unresolved questions

1. Should this notebook stay a compact teaching/demo notebook, or should it be converted into a closer ACIC reproduction using the original `.npz` data contract?
2. Should the notebook include ESN+TARNet and ESN+CFR ablations from paper Section 5.2?
3. Should X-learner (NN), BART, and Causal Forest baselines be added, or is the current deep-model-only comparison enough for the assignment?
