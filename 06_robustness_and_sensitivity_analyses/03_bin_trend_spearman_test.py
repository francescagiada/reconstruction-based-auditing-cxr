"""Weighted mean bin position per outcome class, and a per-bin Spearman
trend test (bin index vs. FN fraction), Benjamini-Hochberg corrected.

Input: output/bin_wise_counts/*_100bins.csv (from the sibling script)
Output: output/reference_positions.csv, output/per_bin_trend_test.csv
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(SCRIPT_DIR, "output", "bin_wise_counts")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def weighted_mean_position(df, count_col):
    w = df[count_col].to_numpy(dtype=float)
    x = df["bin_center"].to_numpy(dtype=float)
    return float(np.sum(w * x) / np.sum(w)) if w.sum() > 0 else np.nan


def main():
    rows_ref, rows_trend = [], []

    for fpath in sorted(glob.glob(os.path.join(BIN_DIR, "*_100bins.csv"))):
        stem = os.path.basename(fpath).replace("_100bins.csv", "")
        parts = stem.split("_")
        pathology, model, metric = parts[0], parts[1], "_".join(parts[2:])

        df = pd.read_csv(fpath).sort_values("bin_center").reset_index(drop=True)

        rows_ref.append({
            "pathology": pathology, "model": model, "metric": metric,
            "fn_pos": weighted_mean_position(df, "fn_count"),
            "fp_pos": weighted_mean_position(df, "fp_count"),
            "tp_pos": weighted_mean_position(df, "tp_count"),
            "tn_pos": weighted_mean_position(df, "tn_count"),
        })

        total = (df["tp_count"] + df["tn_count"] + df["fp_count"] + df["fn_count"]).to_numpy(dtype=float)
        fn_frac = np.where(total > 0, df["fn_count"].to_numpy(dtype=float) / total, np.nan)
        bin_idx = np.arange(len(df))
        mask = ~np.isnan(fn_frac)
        rho, p = spearmanr(bin_idx[mask], fn_frac[mask]) if mask.sum() >= 3 else (np.nan, np.nan)
        rows_trend.append({
            "pathology": pathology, "model": model, "metric": metric,
            "spearman_rho": rho, "p_value": p, "n_bins": int(mask.sum()),
        })

    df_ref = pd.DataFrame(rows_ref).sort_values(["pathology", "model", "metric"])
    df_trend = pd.DataFrame(rows_trend).sort_values(["pathology", "model", "metric"])
    reject, qvals, _, _ = multipletests(df_trend["p_value"].to_numpy(), alpha=0.05, method="fdr_bh")
    df_trend["bh_q"] = qvals
    df_trend["bh_reject"] = reject

    df_ref.to_csv(os.path.join(OUTPUT_DIR, "reference_positions.csv"), index=False)
    df_trend.to_csv(os.path.join(OUTPUT_DIR, "per_bin_trend_test.csv"), index=False)

    print(f"{len(df_ref)} cells processed.")
    sig_bh = df_trend[df_trend["bh_reject"]]
    print(f"{len(sig_bh)} of {len(df_trend)} cells show a significant monotonic trend after BH correction.")
    print("\nAggregate mean TP/TN/FN/FP position across all cells:")
    print(df_ref[["fn_pos", "fp_pos", "tp_pos", "tn_pos"]].mean())


if __name__ == "__main__":
    main()
