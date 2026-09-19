"""Patient-stratified bootstrap 95% CIs and two-tailed permutation
significance testing for FN/FP capture under tail-based prioritization,
Benjamini-Hochberg corrected within each method's own family.

Input: per-pathology, per-autoencoder case CSVs, columns subject_id,
true_label, pred_label_thr_youden, pred_prob, mse, ssim, ppw,
edge_diff_norm. Same format as stage 05
(see ../05_figures_cost_benefit/csv_files/).
Output: output/permutation_test_results.csv
"""

import os
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "05_figures_cost_benefit", "csv_files"))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = [
    "merged_output_atelectasis_tiny", "merged_output_atelectasis_vae",
    "merged_output_cardiomegaly_tiny", "merged_output_cardiomegaly_vae",
    "merged_output_edema_tiny", "merged_output_edema_vae",
    "merged_output_pleuraleffusion_tiny", "merged_output_pleuraleffusion_vae",
    "merged_output_pneumothorax_tiny", "merged_output_pneumothorax_vae",
]

RECONSTRUCTION_METRICS = ["mse", "ssim", "ppw", "edge_diff_norm"]
THRESHOLDS = [0.01, 0.05]

# Direction of "worse reconstruction" per metric: mse/edge_diff_norm are error
# measures (higher = worse), ssim/ppw are similarity measures (lower = worse).
METRIC_DIRECTION = {"mse": "high", "ssim": "low", "ppw": "low", "edge_diff_norm": "high"}

N_BOOTSTRAP = 1000
N_PERMUTATION = 1000
RANDOM_SEED = 42
CI_LOW, CI_HIGH = 2.5, 97.5

rng = np.random.default_rng(RANDOM_SEED)


def get_tail_positions(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    """Positions (0..n-1) of the top `threshold` fraction of `values`, in the given direction."""
    n = int(np.ceil(len(values) * threshold))
    if n <= 0 or len(values) == 0:
        return np.array([], dtype=int)
    n = min(n, len(values))
    if direction == "high":
        return np.argpartition(-values, n - 1)[:n]
    return np.argpartition(values, n - 1)[:n]


def compute_fn_metrics(true_label: np.ndarray, pred_label: np.ndarray, flagged_pos: np.ndarray) -> dict:
    """FN/FP capture, FN ratio, FN yield, and FN enrichment for a given flagged subset."""
    is_fn = (pred_label == 0) & (true_label == 1)
    is_fp = (pred_label == 1) & (true_label == 0)

    total_fn, total_fp, total_n = int(is_fn.sum()), int(is_fp.sum()), len(true_label)
    flagged_fn = int(is_fn[flagged_pos].sum())
    flagged_fp = int(is_fp[flagged_pos].sum())
    flagged_n = len(flagged_pos)
    flagged_errors = flagged_fn + flagged_fp

    fn_capture = (flagged_fn / total_fn * 100) if total_fn > 0 else np.nan
    fn_ratio = (flagged_fn / flagged_errors * 100) if flagged_errors > 0 else np.nan
    fp_capture = (flagged_fp / total_fp * 100) if total_fp > 0 else np.nan
    fn_yield = (flagged_fn / flagged_n * 100) if flagged_n > 0 else np.nan
    baseline_fn_rate = (total_fn / total_n * 100) if total_n > 0 else np.nan
    fn_enrichment = (fn_yield / baseline_fn_rate) if baseline_fn_rate else np.nan

    return dict(
        flagged_total=flagged_n, flagged_fn=flagged_fn, flagged_fp=flagged_fp,
        total_fn=total_fn, total_fp=total_fp,
        fn_capture_pct=fn_capture, fn_ratio_pct=fn_ratio, fp_capture_pct=fp_capture,
        fn_yield_pct=fn_yield, fn_enrichment=fn_enrichment,
    )


def build_subject_position_map(subject_ids: np.ndarray):
    """Precompute, for each unique subject_id, the row positions belonging to it."""
    order = np.argsort(subject_ids, kind="stable")
    sorted_ids = subject_ids[order]
    unique_ids, start_idx, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    return unique_ids, order, start_idx, counts


def resample_positions_by_subject(unique_ids, order, start_idx, counts) -> np.ndarray:
    """Resample patients with replacement; every resampled patient brings all of their rows."""
    n_subj = len(unique_ids)
    draw = rng.integers(0, n_subj, size=n_subj)
    starts, cnts = start_idx[draw], counts[draw]
    total = int(cnts.sum())
    rep_starts = np.repeat(starts, cnts)
    offsets = np.arange(total) - np.repeat(np.cumsum(cnts) - cnts, cnts)
    return order[rep_starts + offsets]


def analyze_file(fname: str) -> list:
    df = pd.read_csv(os.path.join(DATA_DIR, fname + ".csv"), sep=";")
    pathology, model = fname.replace("merged_output_", "").rsplit("_", 1)

    true_label = df["true_label"].to_numpy()
    pred_label = df["pred_label_thr_youden"].to_numpy()
    uncertainty = np.abs(df["pred_prob"].to_numpy() - 0.5)
    metric_arrays = {m: df[m].to_numpy() for m in RECONSTRUCTION_METRICS if m in df.columns}

    subject_ids = df["subject_id"].to_numpy()
    unique_ids, order, start_idx, counts = build_subject_position_map(subject_ids)
    n_total = len(df)

    rows = []
    for threshold in THRESHOLDS:
        thr_label = f"{int(threshold * 100)}pct"

        methods = [("confidence_baseline", "pred_prob_uncertainty", uncertainty, "low")]
        methods += [("autoencoder", m, metric_arrays[m], METRIC_DIRECTION[m])
                    for m in RECONSTRUCTION_METRICS if m in metric_arrays]

        for method, metric_name, values, direction in methods:
            flagged_pos = get_tail_positions(values, threshold, direction)
            observed = compute_fn_metrics(true_label, pred_label, flagged_pos)

            boot_fn_capture = np.empty(N_BOOTSTRAP)
            boot_fn_ratio = np.empty(N_BOOTSTRAP)
            boot_fp_capture = np.empty(N_BOOTSTRAP)
            boot_fn_yield = np.empty(N_BOOTSTRAP)
            boot_fn_enrichment = np.empty(N_BOOTSTRAP)

            for b in range(N_BOOTSTRAP):
                pos = resample_positions_by_subject(unique_ids, order, start_idx, counts)
                boot_values = values[pos]
                boot_flagged = get_tail_positions(boot_values, threshold, direction)
                m = compute_fn_metrics(true_label[pos], pred_label[pos], boot_flagged)
                boot_fn_capture[b] = m["fn_capture_pct"]
                boot_fn_ratio[b] = m["fn_ratio_pct"]
                boot_fp_capture[b] = m["fp_capture_pct"]
                boot_fn_yield[b] = m["fn_yield_pct"]
                boot_fn_enrichment[b] = m["fn_enrichment"]

            n_flagged_observed = len(flagged_pos)
            perm_fn_capture = np.empty(N_PERMUTATION)
            perm_fn_ratio = np.empty(N_PERMUTATION)
            perm_fp_capture = np.empty(N_PERMUTATION)

            for p in range(N_PERMUTATION):
                rand_pos = rng.choice(n_total, size=n_flagged_observed, replace=False)
                m = compute_fn_metrics(true_label, pred_label, rand_pos)
                perm_fn_capture[p] = m["fn_capture_pct"]
                perm_fn_ratio[p] = m["fn_ratio_pct"]
                perm_fp_capture[p] = m["fp_capture_pct"]

            # One-tailed p-values (+1 smoothing): upper tail tests "better than
            # chance", lower tail tests "worse than chance"; both drawn from the
            # same N_PERMUTATION random selections above, at no extra cost.
            p_fn_capture = (1 + np.sum(perm_fn_capture >= observed["fn_capture_pct"])) / (N_PERMUTATION + 1)
            p_fn_ratio = (1 + np.sum(perm_fn_ratio >= observed["fn_ratio_pct"])) / (N_PERMUTATION + 1)
            p_fp_capture = (1 + np.sum(perm_fp_capture >= observed["fp_capture_pct"])) / (N_PERMUTATION + 1)
            p_fn_capture_lower = (1 + np.sum(perm_fn_capture <= observed["fn_capture_pct"])) / (N_PERMUTATION + 1)

            rows.append({
                "pathology": pathology, "model": model, "threshold": thr_label,
                "method": method, "metric": metric_name,
                **observed,
                "fn_capture_ci_low": np.nanpercentile(boot_fn_capture, CI_LOW),
                "fn_capture_ci_high": np.nanpercentile(boot_fn_capture, CI_HIGH),
                "fn_ratio_ci_low": np.nanpercentile(boot_fn_ratio, CI_LOW),
                "fn_ratio_ci_high": np.nanpercentile(boot_fn_ratio, CI_HIGH),
                "fp_capture_ci_low": np.nanpercentile(boot_fp_capture, CI_LOW),
                "fp_capture_ci_high": np.nanpercentile(boot_fp_capture, CI_HIGH),
                "fn_yield_ci_low": np.nanpercentile(boot_fn_yield, CI_LOW),
                "fn_yield_ci_high": np.nanpercentile(boot_fn_yield, CI_HIGH),
                "fn_enrichment_ci_low": np.nanpercentile(boot_fn_enrichment, CI_LOW),
                "fn_enrichment_ci_high": np.nanpercentile(boot_fn_enrichment, CI_HIGH),
                "perm_p_fn_capture": p_fn_capture,
                "perm_p_fn_ratio": p_fn_ratio,
                "perm_p_fp_capture": p_fp_capture,
                "perm_p_fn_capture_lower": p_fn_capture_lower,
            })

    return rows


def main():
    all_rows = []
    for fname in FILES:
        print(f"{fname} ...")
        all_rows.extend(analyze_file(fname))

    df_out = pd.DataFrame(all_rows)

    # Benjamini-Hochberg correction across the 80 autoencoder cells
    # (pathology x architecture x threshold x reconstruction metric) forms
    # the correction family reported as significant/non-significant
    # throughout the manuscript. The 20 confidence-baseline cells are
    # reported alongside for completeness but are not part of that
    # correction family.
    is_autoencoder = df_out["method"] == "autoencoder"
    for col in ["perm_p_fn_capture", "perm_p_fn_ratio", "perm_p_fp_capture", "perm_p_fn_capture_lower"]:
        q_col = col.replace("perm_p_", "bh_q_")
        r_col = col.replace("perm_p_", "bh_reject_")

        # Confidence-baseline rows: reported for completeness, using their
        # natural position within the full 100-cell pool.
        reject_all, qvals_all, _, _ = multipletests(df_out[col].to_numpy(), alpha=0.05, method="fdr_bh")
        df_out[q_col] = qvals_all
        df_out[r_col] = reject_all

        # Autoencoder rows: corrected within their own 80-cell family, the
        # one reported as significant/non-significant throughout the
        # manuscript.
        reject_ae, qvals_ae, _, _ = multipletests(
            df_out.loc[is_autoencoder, col].to_numpy(), alpha=0.05, method="fdr_bh"
        )
        df_out.loc[is_autoencoder, q_col] = qvals_ae
        df_out.loc[is_autoencoder, r_col] = reject_ae

    out_path = os.path.join(OUTPUT_DIR, "permutation_test_results.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
