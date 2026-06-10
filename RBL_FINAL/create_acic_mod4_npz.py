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
GENERATED_DIR = ACIC_DIR / "generated_mod4"
TRAIN_NPZ = ACIC_DIR / "syn_bin_set.5.train.npz"
TEST_NPZ = ACIC_DIR / "syn_bin_set.5.test.npz"
SUMMARY_JSON = ROOT / "RBL_FINAL" / "acic_mod4_generation_summary.json"

RAW_DATA_URL = (
    "https://raw.githubusercontent.com/apurvnnd/"
    "Epileptic-Seizure-Recognition-Using-ANN/master/data.csv"
)


R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[[1]]
out_dir <- args[[2]]
sample_kind <- args[[3]]
generated_count <- as.integer(args[[4]])

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
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

W.ratio <- log(abs(d[, keepCoef[-n.coef] + 1]) + 1) / (abs(d[, keepCoef[-n.coef] + 3]) + 1)
random.cols <- c(1, 51, 81, 111, 151)
W.interact <- d[, random.cols] * d[, random.cols + 10]
temp.mod2 <- as.matrix(cbind(d[, c(random.cols, random.cols + 10)], W.ratio, W.interact))
colnames(temp.mod2) <- paste0("V", 1:ncol(temp.mod2))

set.seed(40)
beta.A.mod4 <- runif(ncol(temp.mod2), -.1, .12) / apply(temp.mod2, 2, sd)
logitA.mod4 <- -.1 + temp.mod2 %*% beta.A.mod4
beta.Y.mod4 <- 2 * beta.A.mod4
beta.Y.mod4[1:5] <- 0
logit.drs.mod4 <- -1.8 + as.matrix(temp.mod2) %*% beta.Y.mod4 + as.matrix(d[, c(150, 160)]) %*% c(-.005, -0.02)

set.seed(21)
psi0.mod4 <- mean(plogis(2 + .01 * d[, 160] + logit.drs.mod4) - plogis(logit.drs.mod4))

d.mod4 <- data.frame(Y = NA, A = NA, d)
set.seed(4)
n.b <- 2000

for (i in 1:generated_count) {
  b <- sample(1:n, n.b, replace = TRUE)
  d.mod4$A[b] <- rbinom(n.b, 1, plogis(logitA.mod4)[b])
  d.mod4$Y[b] <- rbinom(n.b, 1, plogis(d.mod4$A * (2 + .01 * d[, 160]) + logit.drs.mod4)[b])

  mu0 <- as.vector(plogis(logit.drs.mod4[b]))
  mu1 <- as.vector(plogis(2 + .01 * d[b, 160] + logit.drs.mod4[b]))
  a <- d.mod4$A[b]
  ycf <- ifelse(a == 1, mu0, mu1)

  out <- data.frame(
    source_row = b,
    Y = d.mod4$Y[b],
    A = a,
    d[b, ],
    mu0 = mu0,
    mu1 = mu1,
    tau = mu1 - mu0,
    ycf = ycf
  )
  write.csv(out, file = file.path(out_dir, paste0("epilepsyMod4", i, ".csv")), row.names = FALSE)
}

writeLines(sprintf("%.10f", psi0.mod4), file.path(out_dir, "psi0_mod4.txt"))
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


def generate_csv(sample_kind: str, generated_count: int, force: bool) -> None:
    if force and GENERATED_DIR.exists():
        shutil.rmtree(GENERATED_DIR)

    generated_files = list(GENERATED_DIR.glob("epilepsyMod4*.csv"))
    if len(generated_files) == generated_count and (GENERATED_DIR / "psi0_mod4.txt").exists():
        print(f"Using existing generated CSVs in {GENERATED_DIR}")
        return

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    r_script_path = GENERATED_DIR / "generate_mod4_exact.R"
    r_script_path.write_text(R_SCRIPT)

    try:
        run(["Rscript", str(r_script_path), str(RAW_DATA), str(GENERATED_DIR), sample_kind, str(generated_count)])
    except FileNotFoundError as exc:
        raise RuntimeError("Rscript is required to generate exact ACIC Mod 4 data. Install R on the server first.") from exc


def load_block(file_numbers: list[int]) -> dict[str, np.ndarray]:
    frames = [pd.read_csv(GENERATED_DIR / f"epilepsyMod4{i}.csv") for i in file_numbers]
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
        per_experiment.append(
            {
                "experiment": experiment_index + 1,
                "treated": treated,
                "control": row_count - treated,
                "positive_outcome": positive_outcome,
                "treated_ratio": float(t_all[:, experiment_index].mean()),
                "outcome_rate": float(yf_all[:, experiment_index].mean()),
                "mean_tau": float(tau_all[:, experiment_index].mean()),
            }
        )

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
    parser.add_argument("--experiments", type=int, default=5)
    parser.add_argument("--files-per-experiment", type=int, default=20)
    parser.add_argument("--train-start", type=int, default=1)
    parser.add_argument("--test-start", type=int, default=101)
    parser.add_argument("--generated-count", type=int, default=200)
    args = parser.parse_args()

    ensure_inputs()
    generate_csv(args.sample_kind, args.generated_count, args.force)

    train_groups = make_file_groups(args.train_start, args.experiments, args.files_per_experiment)
    test_groups = make_file_groups(args.test_start, args.experiments, args.files_per_experiment)
    all_files = [file_number for group in train_groups + test_groups for file_number in group]
    if min(all_files) < 1 or max(all_files) > args.generated_count:
        raise ValueError("train/test windows must stay within generated files")

    train_arrays = stack_experiments([load_block(group) for group in train_groups])
    test_arrays = stack_experiments([load_block(group) for group in test_groups])

    np.savez(TRAIN_NPZ, **train_arrays)
    np.savez(TEST_NPZ, **test_arrays)

    psi0 = float((GENERATED_DIR / "psi0_mod4.txt").read_text().strip())
    summary = {
        "raw_data": str(RAW_DATA),
        "generated_dir": str(GENERATED_DIR),
        "sample_kind": args.sample_kind,
        "generated_count": args.generated_count,
        "psi0_mod4": psi0,
        "expected_paper_psi0_mod4": 0.2916274,
        "split_note": (
            "The DESCN repo does not include the original CSV-to-npz split manifest. "
            "This script uses exact Mod4 generation, then packs independent contiguous "
            "20-file blocks into each experiment dimension: files 1-100 for train and "
            "101-200 for test by default."
        ),
        "files_per_experiment": args.files_per_experiment,
        "train_files": [args.train_start, args.train_start + args.experiments * args.files_per_experiment - 1],
        "test_files": [args.test_start, args.test_start + args.experiments * args.files_per_experiment - 1],
        "train_file_groups": [[group[0], group[-1]] for group in train_groups],
        "test_file_groups": [[group[0], group[-1]] for group in test_groups],
        "experiment_count": args.experiments,
        "train": summarize(train_arrays),
        "test": summarize(test_arrays),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
