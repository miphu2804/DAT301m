# DESCN: Deep Entire Space Cross Networks for Individual Treatment Effect Estimation

> **Paper**: Kailiang Zhong, Fengtong Xiao, Yan Ren, et al. (Alibaba Group, Lazada)  
> **Published**: KDD '22, August 14–18, 2022, Washington, DC, USA  
> **arXiv**: 2207.09920v3 [cs.LG]  
> **Code**: https://github.com/kailiang-zhong/DESCN  
> **DOI**: https://doi.org/10.1145/3534678.3539198

---

## 1. Problem Definition

### 1.1 Setting

Individual Treatment Effect (ITE) estimation under the **Neyman-Rubin potential outcome framework**, with **binary treatment** and **binary outcome**.

- Observed data: `D = {(y_i, x_i, w_i)}` for i = 1..n
- `x_i ∈ R^d`: covariates (features)
- `w_i ∈ {0, 1}`: binary treatment indicator
- `y_i ∈ {0, 1}`: binary outcome
- `y_i(1)`, `y_i(0)`: potential outcomes under treatment and control respectively
- Propensity score: `π(x) = P(W=1 | X=x)`

**Key challenge**: Only one potential outcome is observed per individual:
```
y_i = w_i · y_i(1) + (1-w_i) · y_i(0)
```
Therefore, the ground truth uplift `y_i(1) - y_i(0)` is never directly observable.

### 1.2 Definitions

| Symbol | Name | Definition |
|--------|------|------------|
| `μ₁(x)` | Treated Response (TR) | `E[Y | W=1, X=x]` |
| `μ₀(x)` | Control Response (CR) | `E[Y | W=0, X=x]` |
| `τ(x)` | Individual Treatment Effect (ITE/CATE) | `μ₁(x) - μ₀(x)` |
| `π(x)` | Propensity Score | `P(W=1 | X=x)` |
| `T` | Treated sample space | `{i : w_i = 1}` |
| `C` | Control sample space | `{i : w_i = 0}` |

### 1.3 Three Standard Assumptions

1. **Consistency**: If individual i receives treatment w_i, then y_i = y_i(w_i) — the observed outcome equals the potential outcome for the received treatment.
2. **Ignorability** (Unconfoundedness): `Y(1), Y(0) ⊥ W | X` — no hidden confounders; treatment assignment is independent of potential outcomes given observed covariates.
3. **Overlap** (Positivity): `0 < π(x) < 1, ∀x ∈ X` — every individual has a non-zero probability of receiving either treatment or control.

### 1.4 Two Core Problems

#### Problem 1 — Treatment Bias
- Treatment assignment follows propensity `π(x)`, not random.
- Treated and control groups have **systematically different distributions** due to confounders.
- E.g., vouchers are only distributed to **inactive users** (treated), while **active users** receive nothing (control). The control group is inherently more active.
- Consequence: model struggles to learn unbiased representations across groups.

#### Problem 2 — Sample Imbalance
- Population sizes of treated and control groups **differ significantly**.
- E.g., free membership given to most users → treated >> control.
- E.g., vouchers only for promotion-sensitive users → treated << control.
- Consequence: hard to learn accurate ITE; requires extra calibration.

---

## 2. Model Architecture — DESCN

DESCN combines two novel components:
1. **Entire Space Network (ESN)** — addresses treatment bias
2. **X-network** — addresses sample imbalance

### 2.1 Entire Space Network (ESN)

**Core idea**: Learn response functions in the **entire sample space** (T ∪ C), not in sub-sample spaces (T or C separately).

Leverages the chain rule of conditional probability:

```
ESTR = P(Y, W=1 | X) = P(Y | W=1, X) · P(W=1 | X) = μ₁ · π
ESCR = P(Y, W=0 | X) = P(Y | W=0, X) · P(W=0 | X) = μ₀ · (1-π)
```

**Architecture** (Figure 1a):
```
                      ┌─────────────┐
  X (features) ──────►│ Shared Layers│
                      └──────┬──────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  π Head  │  │  μ₁ Head │  │  μ₀ Head │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │ π            │ μ₁          │ μ₀
             │              │             │
             └──────┬───────┘    ┌────────┘
                    ▼            ▼
              ESTR = μ₁·π   ESCR = μ₀·(1-π)
```

**Key property**: A treated sample contributes to learning BOTH μ₁ AND μ₀ (and vice versa). Counterfactual information is derived in an integrated manner from the entire sample.

**Loss function**:
```
L_ESN = α · L_π + β₁ · L_ESTR + β₀ · L_ESCR
```

where:
- `L_π`: cross-entropy for propensity prediction `π̂(x_i)` vs `w_i`
- `L_ESTR`: cross-entropy for `μ̂₁(x_i) · π̂(x_i)` vs `y_i & w_i` (logical AND)
- `L_ESCR`: cross-entropy for `μ̂₀(x_i) · (1-π̂(x_i))` vs `y_i & (1-w_i)`
- `α, β₁, β₀`: hyper-parameters controlling loss weights

**Connection to IPW** (proved in paper):
The paper shows ESN implicitly performs Inverse Probability Weighting:
```
ATE = E[ESTR / π] - E[ESCR / (1-π)] = E[μ₁] - E[μ₀]
```

**ESN is pluggable**: It can be added to any uplift model that outputs μ₁ and μ₂ estimates (TARNet, CFR, etc.), not just DESCN.

### 2.2 X-network

**Core idea**: Introduce an intermediate variable **Pseudo Treatment Effect (PTE) τ'** that connects TR and CR, enabling the model to learn both response functions in a more balanced manner.

**Architecture** (Figure 1b):
```
                      ┌─────────────┐
  X (features) ──────►│ Shared Layers│
                      └──────┬──────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌──────────┐       ┌──────────┐        ┌──────────┐
   │ TR Net   │       │ CR Net   │        │ PTE Net  │
   │   μ̂₁     │       │   μ̂₀     │        │   τ̂'     │
   └────┬─────┘       └────┬─────┘        └────┬─────┘
        │                  │                   │
        │       ┌──────────┘                   │
        │       │                              │
        ▼       ▼                              │
   μ̂₁' = σ(σ⁻¹(μ̂₀) + σ⁻¹(τ̂'))                  │
   Cross Treated Response                      │
        │                                      │
        │       ┌──────────────────────────────┘
        │       │
        ▼       ▼
   μ̂₀' = σ(σ⁻¹(μ̂₁) - σ⁻¹(τ̂'))
   Cross Control Response
```

**Why operate in logit space (σ⁻¹)?**
1. Numerical stability — addition/subtraction on logits is well-behaved.
2. Guarantees outputs stay in [0, 1] range after sigmoid.
3. When μ̂₀ or μ̂₁ is close to 0 or 1, σ⁻¹ magnifies the uplift signal, making learning easier for the MLP.

**Intuition**:
- **Cross Treated Response** μ̂₁': "What would the outcome be if control individuals were treated?"
- **Cross Control Response** μ̂₀': "What would the outcome be if treated individuals were not treated?"

**Loss functions** (computed in sub-sample spaces):
```
L_TR  = (1/|T|) Σ_{i∈T} l(y_i, μ̂₁(x_i))        # Treated Response loss
L_CR  = (1/|C|) Σ_{i∈C} l(y_i, μ̂₀(x_i))        # Control Response loss
L_CrossTR = (1/|T|) Σ_{i∈T} l(y_i, μ̂₁'(x_i))   # Cross Treated Response loss
L_CrossCR = (1/|C|) Σ_{i∈C} l(y_i, μ̂₀'(x_i))   # Cross Control Response loss
```

where `l(·)` is cross-entropy loss, `σ` is sigmoid.

**Difference from X-learner**: X-learner learns ITE in two stages (base models first, then ITE estimators). X-network learns everything **jointly in an end-to-end** manner — base learners and ITE are trained together with shared representations.

### 2.3 DESCN = ESN + X-network

**Full architecture** (Figure 1c): X-network wrapped in ESN.

Key change: `L_TR` and `L_CR` from X-network are **removed** because TR and CR are now connected to propensity π as ESTR and ESCR, trained in the entire space.

**Full loss function**:
```
L_DESCN = L_ESN + γ₁ · L_CrossTR + γ₀ · L_CrossCR

        = α · L_π + β₁ · L_ESTR + β₀ · L_ESCR
          + γ₁ · L_CrossTR + γ₀ · L_CrossCR
```

**Training flow**:
1. Forward pass through shared layers → extract representation
2. π, μ₁, μ₀, τ' heads compute their outputs
3. ESTR = μ₁ · π, ESCR = μ₀ · (1-π)
4. Cross responses: μ₁' = σ(σ⁻¹(μ₀) + σ⁻¹(τ')), μ₀' = σ(σ⁻¹(μ₁) - σ⁻¹(τ'))
5. Compute 5 losses and back-propagate

**Why this works**:
- Shared network learns a comprehensive representation capturing propensity, both responses, and pseudo treatment effect simultaneously.
- ESN ensures training uses entire sample space → reduces treatment bias.
- X-network's PTE connects TR and CR → balances learning when one group is small.

---

## 3. Datasets

### 3.1 Epilepsy (Synthetic)

| Property | Train | Test |
|----------|-------|------|
| Size | 40K | 40K |
| Treated | 20.2K | 19.8K |
| Control | 19.8K | 20.1K |
| Positive outcome rate | 45.9% | 45.3% |
| Covariates | 178 | — |
| Ground truth ITE | ✅ Available | ✅ Available |

Generated from ACIC 2019 Data Challenge DGP using Epileptic Seizure Recognition Dataset covariates. Uses "Mod 4" (complex models + treatment effect heterogeneity).

### 3.2 Production (Real-world from Lazada)

| Property | Train | Test |
|----------|-------|------|
| Size | 4.17M | 0.91M |
| Treated | 0.92M | 0.47M |
| Control | 3.25M | 0.43M |
| Positive outcome rate | 2.0% | 3.5% |
| Covariates | 83 | — |
| Ground truth ITE | ❌ Not available | ❌ Not available |

**Key design**:
- **Training set**: Collected from real voucher distribution with **selective treatment** → strong treatment bias + natural sample imbalance.
- **Testing set**: Users not affected by targeting strategy, treatment follows **RCT (randomized controlled trials)** → unbiased evaluation.
- First **public industrial dataset** with both biased training and randomized testing.

---

## 4. Baselines

| Model | Type | Key Characteristics |
|-------|------|---------------------|
| **X-learner (NN)** | Meta-learner | Two-stage: base models → imputed ITE → weighted ITE |
| **BART** | Bayesian Tree | Sum-of-trees, S-learner style, handles high-dim predictors |
| **Causal Forest** | Random Forest | Non-parametric, builds causal trees with different sub-sampling |
| **TARNet** | Deep Learning | Two-headed NN with shared layers for TR and CR |
| **CFR_wass** | Deep Learning | TARNet + Wasserstein distance regularization for balanced representations |
| **CFR_mmd** | Deep Learning | TARNet + MMD regularization for balanced representations |

---

## 5. Evaluation Metrics

### 5.1 For Epilepsy (ground truth ITE available)

**√ε_PEHE** (Precision in Estimation of Heterogeneous Effect):
```
ε_PEHE = (1/n) Σ_i [ (μ̂₁(x_i) - μ̂₀(x_i)) - τ(x_i) ]²
√ε_PEHE = sqrt(ε_PEHE)
```
- Measures **individual-level** CATE prediction accuracy.
- Lower is better.

**ε_ATE** (Absolute Error in Average Treatment Effect):
```
ε_ATE = | (1/n) Σ_i (μ̂₁(x_i) - μ̂₀(x_i)) - (1/n) Σ_i τ(x_i) |
```
- Measures **average** ITE prediction accuracy.
- Lower is better.

### 5.2 For Production (no ground truth ITE)

**AUUC** (Area Under Uplift Curve / Qini coefficient):
- Evaluates **uplift score ranking performance**.
- Computed by ranking users by predicted uplift, then comparing cumulative incremental gains.
- Higher is better.

**ε_ATT** (Error in Average Treatment Effect on the Treated):
```
ATT = (1/|T|) Σ_{i∈T} y_i - (1/|C|) Σ_{i∈C} y_i
ε_ATT = | (1/|T|) Σ_{i∈T} (μ̂₁(x_i) - μ̂₀(x_i)) - ATT |
```
- Uses the **randomized test set** to compute true ATT.
- Lower is better.

### 5.3 Improvement Calculation
```
Impr(BaselineModel) = (E(Model) - E(BaselineModel)) / E(BaselineModel) × 100%
```
Note: negative improvement for metrics where lower is better = actual improvement.

---

## 6. Results

### 6.1 Overall Performance (Table 2)

| Model | √ε_PEHE | ε_ATE | AUUC | ε_ATT |
|-------|---------|-------|------|-------|
| X-learner (NN) | 0.1556 | 0.0378 | 0.0234 | 0.0076 |
| Causal Forest | 0.1519 | 0.0663 | 0.0132 | 0.0123 |
| BART | 0.1387 | 0.0389 | 0.0222 | 0.0312 |
| TARNet | 0.1373 | 0.0405 | 0.0309 | 0.0106 |
| CFR_wass | 0.1363 | 0.0263 | 0.0261 | 0.0266 |
| CFR_mmd | 0.1344 | 0.0305 | 0.0324 | 0.0258 |
| X-network | 0.1289 | 0.0245 | 0.0324 | 0.0048 |
| **DESCN** | **0.1241** | **0.0058** | **0.0340** | **0.0039** |

**Key takeaways**:
- DESCN beats all baselines across all metrics.
- **+7.6%** over CFR_mmd on √ε_PEHE.
- **+4.9%** over CFR_mmd on AUUC.
- Most dramatic: **+80% on ε_ATE**, **+84% on ε_ATT** — DESCN excels at average treatment effect estimation.

### 6.2 ESN Ablation (Q1)

Adding ESN to existing models:

| | √ε_PEHE | AUUC |
|---|---|---|
| TARNet → ESN+TARNet | **+3.9%** ↑ | **+10.0%** ↑ |
| CFR_wass → ESN+CFR_wass | -1.3% ↓ | +1.1% ↑ |
| CFR_mmd → ESN+CFR_mmd | -15.7% ↓ | +2.1% ↑ |
| X-network → DESCN (ESN) | **+3.7%** ↑ | **+4.9%** ↑ |

**Why ESN sometimes hurts CFR**: CFR already uses IPM losses (Wasserstein/MMD) to force balanced representations. The additional propensity information from ESN may **conflict** with that regularization. However, ESN consistently improves ε_ATE across all models.

**Trade-off observation**: Debiasing (via ESN or CFR) tends to improve individual estimation (√ε_PEHE) but can hurt average estimation (ε_ATE), and vice versa. This is a known phenomenon also noted in Shalit et al. (2017).

### 6.3 X-network Ablation (Q2)

Comparing X-network vs TARNet (both without ESN):

| | √ε_PEHE | AUUC |
|---|---|---|
| TARNet | 0.1373 | 0.0309 |
| X-network | 0.1289 | 0.0324 |
| Improvement | **+9.6%** | **+10.0%** |

Plus significant improvements in ε_ATE and ε_ATT. The PTE network effectively bridges TR and CR, enabling more balanced learning.

---

## 7. Hyper-parameter Settings

### Epilepsy Dataset
| Parameter | Value |
|-----------|-------|
| Hidden units (shared) | 128 |
| FC layers | 3 |
| L2 regularization | 0.01 |
| Learning rate | 0.001 (no decay) |
| Batch size | 500 |
| Epochs | 15 |

### Production Dataset
| Parameter | Value |
|-----------|-------|
| Hidden units (shared) | 128 |
| Hidden units (sub-models) | 64 |
| FC layers | 3 |
| L2 regularization | 0.001 |
| Learning rate | 0.001 |
| LR decay | 0.95 |
| Batch size | 5000 |
| Epochs | 5 |

### Loss weights (model-specific)
- `α, β₁, β₀`: ESN loss weights
- `γ₁, γ₀`: Cross response loss weights
- These need tuning per dataset.

---

## 8. Mathematical Details for Implementation

### 8.1 Forward Pass

```
h = SharedNetwork(x)                    # Shared representation

π̂   = σ( MLP_π(h) )                    # Propensity head → sigmoid
μ̂₁  = σ( MLP_TR(h) )                   # TR head → sigmoid
μ̂₀  = σ( MLP_CR(h) )                   # CR head → sigmoid
τ̂'  = σ( MLP_PTE(h) )                  # PTE head → sigmoid

ESTR = μ̂₁ · π̂                           # Entire Space Treated Response
ESCR = μ̂₀ · (1 - π̂)                     # Entire Space Control Response

# Cross responses (in logit space)
logit_μ̂₁' = logit(μ̂₀) + logit(τ̂')      # logit(x) = ln(x/(1-x))
logit_μ̂₀' = logit(μ̂₁) - logit(τ̂')
μ̂₁' = σ(logit_μ̂₁')                      # Cross Treated Response
μ̂₀' = σ(logit_μ̂₀')                      # Cross Control Response
```

### 8.2 Loss Computation

```python
# Binary cross-entropy
def bce(y_true, y_pred):
    return -(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))

# Propensity loss: predict w from π̂(x)
L_π = mean( bce(w_i, π̂_i) )

# ESTR loss: predict (y AND w) from ESTR
L_ESTR = mean( bce(y_i * w_i, μ̂₁_i * π̂_i) )

# ESCR loss: predict (y AND (1-w)) from ESCR
L_ESCR = mean( bce(y_i * (1-w_i), μ̂₀_i * (1-π̂_i)) )

# Cross TR loss: only on treated samples
t_mask = (w_i == 1)
L_CrossTR = mean( bce(y_i[t_mask], μ̂₁'_i[t_mask]) )

# Cross CR loss: only on control samples
c_mask = (w_i == 0)
L_CrossCR = mean( bce(y_i[c_mask], μ̂₀'_i[c_mask]) )

# Total
L = α*L_π + β₁*L_ESTR + β₀*L_ESCR + γ₁*L_CrossTR + γ₀*L_CrossCR
```

### 8.3 Inference (ITE Prediction)

```
τ̂(x) = μ̂₁(x) - μ̂₀(x)
```

### 8.4 Numerical Stability Notes

- Use `eps = 1e-7` clamping for all sigmoid/log operations to avoid log(0).
- The logit function: `logit(p) = ln(p / (1-p))` with `p ∈ (0,1)`.
- In practice, implement logit addition as:
  ```python
  def sigmoid(x): return 1 / (1 + exp(-clip(x, -20, 20)))
  def logit(p): return log(clip(p, eps, 1-eps) / clip(1-p, eps, 1-eps))
  ```

---

## 9. Strengths & Limitations

### Strengths
1. **Integrated solution**: First end-to-end model tackling both treatment bias and sample imbalance simultaneously.
2. **Pluggable ESN**: Can enhance any existing uplift model (TARNet, CFR, etc.).
3. **Real-world validation**: Tested on large-scale industrial dataset with RCT-based evaluation.
4. **Public resources**: Code and dataset released for reproducibility.
5. **Strong theoretical grounding**: ESN's connection to IPW is formally shown.

### Limitations
1. **Binary only**: Only handles binary treatment (W ∈ {0,1}) and binary outcome (Y ∈ {0,1}). Extensions needed for multi-treatment or continuous outcomes.
2. **ESN-CFR conflict**: ESN can hurt CFR's performance due to conflicting regularization, requiring careful tuning.
3. **Many hyper-parameters**: 5 loss weights (α, β₁, β₀, γ₁, γ₀) to tune.
4. **Sigmoid output assumption**: The logit-space trick assumes both responses and PTE are bounded in [0,1], which may not hold for all types of outcomes.
5. **Scalability**: The 5-head architecture with shared layers may be heavy for very high-dimensional feature spaces.

---

## 10. Key Implementation Notes

1. **Shared layers**: A 3-layer MLP with 128 hidden units. All 4 heads (π, μ₁, μ₀, τ') branch from the same shared representation.
2. **Head networks**: Each head is a small MLP (e.g., single dense layer + sigmoid).
3. **ESN multiplication is NOT a computational graph detach**: Gradients flow through both μ̂ and π̂ into the shared layers when computing ESTR/ESCR loss.
4. **Cross response logit trick**: The `σ⁻¹(μ̂₀) + σ⁻¹(τ̂')` operation should allow gradients to flow to both μ̂₀ and τ̂'.
5. **Masked losses**: L_CrossTR is only computed on T (w=1 samples), L_CrossCR only on C (w=0 samples). L_ESTR and L_ESCR are computed on ALL samples.
6. **Inference mode**: Only need μ̂₁ and μ̂₀ heads; τ̂', π̂, and cross connections are training-only constructs.
