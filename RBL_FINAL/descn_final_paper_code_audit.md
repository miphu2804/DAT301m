# DESCN final audit against paper and public code

Scope: `RBL_FINAL/descn_implementation.ipynb` compared with the DESCN paper and the public `DESCN/` implementation. This audit intentionally excludes ablation coverage and the already-known TensorFlow-vs-PyTorch loss/IPM numerical differences.

## Current alignment

The notebook now matches the public Lazada production protocol on the main experimental surface:

- Uses the public Lazada train/test CSVs instead of synthetic data.
- Builds 5 experiment subsets from the biased train CSV with `train_test_split(..., test_size=0.9, random_state=exp_i)`.
- Reuses the randomized Lazada test CSV for all experiments.
- Uses 80/20 train/validation split inside each experiment, matching `val_rate: 0.2`.
- Uses production hyperparameters: `share_dim=128`, `base_dim=64`, `batch_size=5000`, `lr=0.001`, `l2=0.001`, `do_rate=0.1`, `epochs=5`, `decay_rate=0.95`, `decay_step_size=1`.
- Uses `BatchNorm1d: true` equivalent in the shared network, with TensorFlow BatchNorm configured close to PyTorch defaults.
- Uses seed `2` and a sequential validation permutation to match the public `main.py` run setup more closely.
- Uses DESCN architecture semantics from `model/models.py`: shared backbone, propensity head, `mu1`, `mu0`, pseudo `tau`, ESTR/ESCR, and final ITE `p_h1 - p_h0`.
- Uses public-code loss weights for TARNet, CFR(MMD), X-network, and DESCN.
- Uses StepLR-style learning rate updates instead of TensorFlow exponential step scheduling.
- Evaluates each epoch and selects the best AUUC epoch per experiment, matching `eval4real_data.py`.
- Saves public-code-style test prediction artifacts under `results/lzd_real_tf/` with `p_prpsy`, `p_yf`, `p_ycf`, `p_tau`, `loss`, and `val`.
- Linux/RunPod dependency path uses `tensorflow[and-cuda]==2.17.1`; macOS keeps TensorFlow Metal gated to Apple Silicon.

## Fixed in the final audit pass

These were the remaining differences found outside ablation/loss and were fixed:

1. BatchNorm was disabled in the notebook while Lazada public configs use `BatchNorm1d: true`.
   - Fixed by setting `HYPERPARAMS['use_batchnorm'] = True`.
   - TensorFlow BatchNorm uses `momentum=0.9, epsilon=1e-5`, corresponding to PyTorch `momentum=0.1, eps=1e-5`.

2. Notebook RNG did not match the public code closely.
   - Fixed setup seed to `seed_everything(2)`.
   - Removed per-experiment reseeding from model wrappers.
   - Validation split now uses a sequential `np.random.RandomState(2).permutation(...)`, closer to `seed_torch(2)` plus `np.random.permutation(...)` in `main.py`.

3. Dense layer initialization used Keras defaults instead of public PyTorch initialization.
   - Fixed by adding `torch_style_dense(...)`.
   - Uses fan-in variance scaling with untruncated normal and zero bias, matching the intent of `init_weights()` in `model/models.py`.

## Accepted remaining differences

These remain by design or due to the TensorFlow migration:

1. Framework migration is not numerically identical.
   - TensorFlow execution, Keras layers, optimizer internals, and CUDA kernels will not reproduce PyTorch bit-for-bit.

2. IPM implementation is still a TensorFlow approximation.
   - Public code uses `geomloss.SamplesLoss` in PyTorch.
   - Notebook keeps differentiable TensorFlow approximations so CFR(MMD)/WASS can train inside the TensorFlow graph.

3. Weight decay is close but not guaranteed identical.
   - Public code uses PyTorch `optim.Adam(..., weight_decay=cfg.l2)`.
   - Notebook uses TensorFlow Adam behavior. This is a numeric optimizer difference, not a protocol difference.

4. Paper-scale production dataset cannot be reproduced from the public CSV.
   - The public Lazada CSV is smaller than the proprietary production dataset reported in the paper table.
   - Results should be compared to public-code behavior, not expected to match the paper table exactly.

5. Ablation/model coverage is intentionally narrower.
   - Notebook focuses on TARNet, CFR(MMD), X-network, and DESCN.
   - It does not attempt full public notebook ablations such as ES_TARNet, ES_CFR variants, CFR(WASS), X-learner, or X-learner with propensity score.

## Verification performed

- Parsed all notebook code cells successfully after changes.
- Ran TensorFlow smoke training with BatchNorm enabled.
- Verified StepLR values `[0.001, 0.00095]` for a 2-epoch smoke run.
- Verified selected epoch equals `argmax(history['auuc'])`.
- Verified saved `.npz` artifact shape follows `units x experiment x outputs`.
- Removed temporary smoke artifact after verification.

## Unresolved questions

- None blocking for the requested production RunPod run.
- If exact PyTorch numeric reproduction becomes the goal, the TensorFlow migration itself should be replaced by running the original PyTorch code path instead of continuing to approximate PyTorch internals in TensorFlow.
