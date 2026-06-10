from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACIC_DIR = ROOT / "DESCN" / "data" / "ACIC2019_epilepsy_dataset"
RAW_DATA = ACIC_DIR / "raw" / "data.csv"
GENERATED_BASE = ACIC_DIR / "generated_all_mods"
TRAIN_NPZ = ACIC_DIR / "syn_bin_set.5.train.npz"
TEST_NPZ = ACIC_DIR / "syn_bin_set.5.test.npz"
SUMMARY_JSON = ROOT / "RBL_FINAL" / "acic_generation_summary.json"

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/apurvnnd/"
    "Epileptic-Seizure-Recognition-Using-ANN/master/data.csv"
)

# ============================================================
# R script — generates all 4 Epilepsy Mods with ground truth.
#   Mod 1: simple main-terms models
#   Mod 2: model misspecification
#   Mod 3: treatment effect heterogeneity
#   Mod 4: treatment heterogeneity + instrumental variables
#
# Seeds match the original generate_simEpilepsy.R exactly.
# n.b is unified to 2000 for consistent experiment dimensions
# (the original Mod 1-2 used 1500; 2000 matches Mod 3-4).
# ============================================================

R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
data_path   <- args[[1]]
out_base    <- args[[2]]
sample_kind <- args[[3]]
files_per_mod <- as.integer(args[[4]])

if (sample_kind != "default") {
  suppressWarnings(RNGkind(sample.kind = sample_kind))
}

d.raw <- read.csv(data_path, header = TRUE)
if (ncol(d.raw) != 180) {
  stop(sprintf("Expected data.csv with 180 columns, got %s", ncol(d.raw)))
}

d <- d.raw[, -c(1, 180)]
colnames(d) <- paste0("V", 1:ncol(d))
keepCoef <- seq(1, 178, 4)
n.coef <- length(keepCoef)
n <- nrow(d)

# Shared pre-computations for Mod 2 / Mod 4
W.ratio <- log(abs(d[, keepCoef[-n.coef] + 1]) + 1) / (abs(d[, keepCoef[-n.coef] + 3]) + 1)
random.cols <- c(1, 51, 81, 111, 151)
W.interact <- d[, random.cols] * d[, random.cols + 10]
temp.mod2 <- as.matrix(cbind(d[, c(random.cols, random.cols + 10)], W.ratio, W.interact))
colnames(temp.mod2) <- paste0("V", 1:ncol(temp.mod2))

n.b <- 2000  # unified bootstrap size

# ============================================================
# Mod 1 — Simple main-terms models
# ============================================================
cat("\n--- Generating Mod 1 ---\n")
set.seed(6)
beta.A.mod1 <- runif(n.coef, -.04, .094) / apply(d[, keepCoef], 2, sd)
logitA.mod1 <- -.4 + as.matrix(d[, keepCoef]) %*% beta.A.mod1
beta.Y.mod1 <- 3 * beta.A.mod1
logit.drs.mod1 <- -.7 + as.matrix(d[, keepCoef]) %*% beta.Y.mod1
psi0.mod1 <- mean(plogis(-3 + logit.drs.mod1) - plogis(logit.drs.mod1))

set.seed(1)
for (i in 1:files_per_mod) {
  b <- sample(1:n, n.b, replace = TRUE)
  A <- rbinom(n.b, 1, plogis(logitA.mod1)[b])
  Y <- rbinom(n.b, 1, plogis(-3 * A + logit.drs.mod1)[b])
  mu0 <- as.vector(plogis(logit.drs.mod1[b]))
  mu1 <- as.vector(plogis(-3 + logit.drs.mod1[b]))
  tau <- mu1 - mu0
  ycf <- ifelse(A == 1, mu0, mu1)
  out <- data.frame(source_row=b, Y=Y, A=A, d[b, ], mu0=mu0, mu1=mu1, tau=tau, ycf=ycf)
  dir.create(file.path(out_base, "mod1"), recursive=TRUE, showWarnings=FALSE)
  write.csv(out, file.path(out_base, "mod1", paste0("epilepsyMod1_", i, ".csv")), row.names=FALSE)
}
cat(sprintf("Mod 1 psi0: %.10f\n", psi0.mod1))

# ============================================================
# Mod 2 — Model misspecification
# ============================================================
cat("\n--- Generating Mod 2 ---\n")
set.seed(21)
beta.A.mod2 <- runif(ncol(temp.mod2), -.02, .09) / apply(temp.mod2, 2, sd)
logitA.mod2 <- -.5 + temp.mod2 %*% beta.A.mod2
beta.Y.mod2 <- 2.4 * beta.A.mod2
logit.drs.mod2 <- -1.8 + as.matrix(temp.mod2[, -1]) %*% beta.Y.mod2[-1] + as.matrix(d[, c(150, 160)]) %*% c(-.04, .06)
psi0.mod2 <- mean(plogis(3 + logit.drs.mod2) - plogis(logit.drs.mod2))

set.seed(2)
for (i in 1:files_per_mod) {
  b <- sample(1:n, n.b, replace = TRUE)
  A <- rbinom(n.b, 1, plogis(logitA.mod2)[b])
  Y <- rbinom(n.b, 1, plogis(3 * A + logit.drs.mod2)[b])
  mu0 <- as.vector(plogis(logit.drs.mod2[b]))
  mu1 <- as.vector(plogis(3 + logit.drs.mod2[b]))
  tau <- mu1 - mu0
  ycf <- ifelse(A == 1, mu0, mu1)
  out <- data.frame(source_row=b, Y=Y, A=A, d[b, ], mu0=mu0, mu1=mu1, tau=tau, ycf=ycf)
  dir.create(file.path(out_base, "mod2"), recursive=TRUE, showWarnings=FALSE)
  write.csv(out, file.path(out_base, "mod2", paste0("epilepsyMod2_", i, ".csv")), row.names=FALSE)
}
cat(sprintf("Mod 2 psi0: %.10f\n", psi0.mod2))

# ============================================================
# Mod 3 — Treatment effect heterogeneity
# ============================================================
cat("\n--- Generating Mod 3 ---\n")
set.seed(3)
beta.A.mod3 <- runif(n.coef, -.03, .1) / apply(d[, keepCoef], 2, sd)
logitA.mod3 <- -.5 + as.matrix(d[, keepCoef]) %*% beta.A.mod3
beta.Y.mod3 <- 2 * beta.A.mod3
logit.drs.mod3 <- -1 + as.matrix(d[, keepCoef]) %*% beta.Y.mod3
cond30 <- as.numeric(d[, keepCoef[30]] < mean(d[, keepCoef[30]]))
cond4  <- as.numeric(d[, keepCoef[4]]  < 0)
psi0.mod3 <- mean(plogis((.5 + .6 * cond30 + .8 * cond4) + logit.drs.mod3) - plogis(logit.drs.mod3))

set.seed(23)
for (i in 1:files_per_mod) {
  b <- sample(1:n, n.b, replace = TRUE)
  A <- rbinom(n.b, 1, plogis(logitA.mod3)[b])
  te <- .5 + .6 * cond30[b] + .8 * cond4[b]
  Y <- rbinom(n.b, 1, plogis(A * te + logit.drs.mod3)[b])
  mu0 <- as.vector(plogis(logit.drs.mod3[b]))
  mu1 <- as.vector(plogis(te + logit.drs.mod3[b]))
  tau <- mu1 - mu0
  ycf <- ifelse(A == 1, mu0, mu1)
  out <- data.frame(source_row=b, Y=Y, A=A, d[b, ], mu0=mu0, mu1=mu1, tau=tau, ycf=ycf)
  dir.create(file.path(out_base, "mod3"), recursive=TRUE, showWarnings=FALSE)
  write.csv(out, file.path(out_base, "mod3", paste0("epilepsyMod3_", i, ".csv")), row.names=FALSE)
}
cat(sprintf("Mod 3 psi0: %.10f\n", psi0.mod3))

# ============================================================
# Mod 4 — Treatment heterogeneity + instrumental variables
# ============================================================
cat("\n--- Generating Mod 4 ---\n")
set.seed(40)
beta.A.mod4 <- runif(ncol(temp.mod2), -.1, .12) / apply(temp.mod2, 2, sd)
logitA.mod4 <- -.1 + temp.mod2 %*% beta.A.mod4
beta.Y.mod4 <- 2 * beta.A.mod4
beta.Y.mod4[1:5] <- 0
logit.drs.mod4 <- -1.8 + as.matrix(temp.mod2) %*% beta.Y.mod4 + as.matrix(d[, c(150, 160)]) %*% c(-.005, -0.02)
psi0.mod4 <- mean(plogis(2 + .01 * d[, 160] + logit.drs.mod4) - plogis(logit.drs.mod4))

set.seed(4)
for (i in 1:files_per_mod) {
  b <- sample(1:n, n.b, replace = TRUE)
  A <- rbinom(n.b, 1, plogis(logitA.mod4)[b])
  Y <- rbinom(n.b, 1, plogis(A * (2 + .01 * d[b, 160]) + logit.drs.mod4)[b])
  mu0 <- as.vector(plogis(logit.drs.mod4[b]))
  mu1 <- as.vector(plogis(2 + .01 * d[b, 160] + logit.drs.mod4[b]))
  tau <- mu1 - mu0
  ycf <- ifelse(A == 1, mu0, mu1)
  out <- data.frame(source_row=b, Y=Y, A=A, d[b, ], mu0=mu0, mu1=mu1, tau=tau, ycf=ycf)
  dir.create(file.path(out_base, "mod4"), recursive=TRUE, showWarnings=FALSE)
  write.csv(out, file.path(out_base, "mod4", paste0("epilepsyMod4_", i, ".csv")), row.names=FALSE)
}
cat(sprintf("Mod 4 psi0: %.10f\n", psi0.mod4))

# Save psi0 values
psi0_lines <- c(
  sprintf("%.10f", psi0.mod1),
  sprintf("%.10f", psi0.mod2),
  sprintf("%.10f", psi0.mod3),
  sprintf("%.10f", psi0.mod4)
)
writeLines(psi0_lines, file.path(out_base, "psi0_all_mods.txt"))
cat("\nDone.\n")
"""


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def ensure_inputs() -> None:
    if not RAW_DATA.exists():
        RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading UCI epilepsy data to {RAW_DATA}")
        urllib.request.urlretrieve(RAW_DATA_URL, RAW_DATA)

    data = pd.read_csv(RAW_DATA, nrows=5)
    if data.shape[1] != 180:
        raise ValueError(f"{RAW_DATA} should have 180 columns; got {data.shape[1]}")


def generate_csv(sample_kind: str, files_per_mod: int, force: bool) -> None:
    if force and GENERATED_BASE.exists():
        shutil.rmtree(GENERATED_BASE)

    all_mod_dirs = [GENERATED_BASE / f"mod{m}" for m in range(1, 5)]
    all_exist = all(
        len(list(d.glob("epilepsyMod*_*.csv"))) >= files_per_mod
        for d in all_mod_dirs
    )
    psi0_file = GENERATED_BASE / "psi0_all_mods.txt"
    if all_exist and psi0_file.exists():
        print(f"Using existing generated CSVs in {GENERATED_BASE}")
        return

    GENERATED_BASE.mkdir(parents=True, exist_ok=True)
    r_script_path = GENERATED_BASE / "generate_all_mods.R"
    r_script_path.write_text(R_SCRIPT)

    try:
        run(["Rscript", str(r_script_path), str(RAW_DATA), str(GENERATED_BASE), sample_kind, str(files_per_mod)])
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Rscript is required to generate ACIC data. Install R on the server first."
        ) from exc


def load_block(mod: int, file_numbers: list[int]) -> dict[str, np.ndarray]:
    """Load a contiguous block of CSV files from one Mod and stack rows."""
    mod_dir = GENERATED_BASE / f"mod{mod}"
    frames = [pd.read_csv(mod_dir / f"epilepsyMod{mod}_{i}.csv") for i in file_numbers]
    frame = pd.concat(frames, ignore_index=True)
    feature_columns = [f"V{i}" for i in range(1, 179)]
    return {
        "x": frame[feature_columns].to_numpy(np.float32),
        "t": frame["A"].to_numpy(np.float32),
        "yf": frame["Y"].to_numpy(np.float32),
        "ycf": frame["ycf"].to_numpy(np.float32),
        "mu0": frame["mu0"].to_numpy(np.float32),
        "mu1": frame["mu1"].to_numpy(np.float32),
        "tau": frame["tau"].to_numpy(np.float32),
        "e": np.zeros(len(frame), dtype=np.float32),
    }


def stack_experiments(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    stacked: dict[str, np.ndarray] = {}
    for key in blocks[0]:
        values = [block[key] for block in blocks]
        stacked[key] = np.stack(values, axis=2 if key == "x" else 1)
    return stacked


def make_file_groups(start: int, experiment_count: int, files_per_experiment: int) -> list[list[int]]:
    return [
        list(range(start + i * files_per_experiment, start + (i + 1) * files_per_experiment))
        for i in range(experiment_count)
    ]


def summarize(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    t_all = arrays["t"]
    yf_all = arrays["yf"]
    tau_all = arrays["tau"]
    per_experiment = []
    for experiment_index in range(t_all.shape[1]):
        treated = int(t_all[:, experiment_index].sum())
        positive_outcome = int(yf_all[:, experiment_index].sum())
        row_count = int(t_all.shape[0])
        per_experiment.append({
            "experiment": experiment_index + 1,
            "treated": treated,
            "control": row_count - treated,
            "positive_outcome": positive_outcome,
            "treated_ratio": float(t_all[:, experiment_index].mean()),
            "outcome_rate": float(yf_all[:, experiment_index].mean()),
            "mean_tau": float(tau_all[:, experiment_index].mean()),
        })
    return {
        "x_shape": list(arrays["x"].shape),
        "treated_mean": float(t_all.sum(axis=0).mean()),
        "control_mean": float((t_all.shape[0] - t_all.sum(axis=0)).mean()),
        "positive_outcome_mean": float(yf_all.sum(axis=0).mean()),
        "treated_ratio_mean": float(t_all.mean(axis=0).mean()),
        "outcome_rate_mean": float(yf_all.mean(axis=0).mean()),
        "mean_tau_mean": float(tau_all.mean(axis=0).mean()),
        "per_experiment": per_experiment,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sample-kind", default="Rounding", choices=["Rounding", "Rejection", "default"])
    parser.add_argument("--experiments-per-mod", type=int, default=5)
    parser.add_argument("--files-per-experiment", type=int, default=20)
    parser.add_argument("--files-per-mod", type=int, default=200,
                        help="Total CSV files per Mod (must be >= 2 * experiments_per_mod * files_per_experiment)")
    args = parser.parse_args()

    ensure_inputs()
    generate_csv(args.sample_kind, args.files_per_mod, args.force)

    # Build experiment blocks: each experiment = contiguous files from one Mod.
    # Train uses first half of files, test uses second half (per Mod).
    half = args.files_per_mod // 2
    train_start = 1
    test_start = 1 + half

    all_train_blocks: list[dict[str, np.ndarray]] = []
    all_test_blocks: list[dict[str, np.ndarray]] = []
    train_file_groups: list[list[int]] = []
    test_file_groups: list[list[int]] = []

    for mod in range(1, 5):
        mod_train_groups = make_file_groups(train_start, args.experiments_per_mod, args.files_per_experiment)
        mod_test_groups = make_file_groups(test_start, args.experiments_per_mod, args.files_per_experiment)

        for group in mod_train_groups:
            all_train_blocks.append(load_block(mod, group))
            train_file_groups.append(group)
        for group in mod_test_groups:
            all_test_blocks.append(load_block(mod, group))
            test_file_groups.append(group)

    total_experiments = 4 * args.experiments_per_mod
    print(f"\nTotal experiments: {total_experiments} (4 Mods × {args.experiments_per_mod} experiments)")
    print(f"Train experiments: {len(all_train_blocks)} | Test experiments: {len(all_test_blocks)}")

    train_arrays = stack_experiments(all_train_blocks)
    test_arrays = stack_experiments(all_test_blocks)

    np.savez(TRAIN_NPZ, **train_arrays)
    np.savez(TEST_NPZ, **test_arrays)
    print(f"Saved:\n  {TRAIN_NPZ}\n  {TEST_NPZ}")

    # Read psi0 values
    psi0_values = [float(v) for v in (GENERATED_BASE / "psi0_all_mods.txt").read_text().strip().split()]
    psi0_by_mod = {f"mod{m}": psi0_values[m - 1] for m in range(1, 5)}

    summary = {
        "raw_data": str(RAW_DATA),
        "generated_dir": str(GENERATED_BASE),
        "sample_kind": args.sample_kind,
        "files_per_mod": args.files_per_mod,
        "n_bootstrap": 2000,
        "experiments_per_mod": args.experiments_per_mod,
        "files_per_experiment": args.files_per_experiment,
        "total_experiments": total_experiments,
        "psi0_by_mod": psi0_by_mod,
        "psi0_mean": sum(psi0_values) / len(psi0_values),
        "expected_paper_psi0_mod4": 0.2916274,
        "expected_paper_psi0_mod2": 0.2165881,
        "train": summarize(train_arrays),
        "test": summarize(test_arrays),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
