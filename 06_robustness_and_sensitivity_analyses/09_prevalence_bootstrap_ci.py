"""Step 6: patient-stratified bootstrap 95% CI for the reweighted FN
ratio.

Input:
    output/step1_realistic_prevalence.csv
    per-pathology TinyAE case CSVs (see ../05_figures_cost_benefit/csv_files/)
Output:
    output/step6_bootstrap_fn_ratio_reweighted.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / ".." / "05_figures_cost_benefit" / "csv_files"
OUTPUT_DIR = SCRIPT_DIR / "output"
N_BOOT = 1000
SEED = 42


def load_prevalence():
    prev = pd.read_csv(OUTPUT_DIR / "step1_realistic_prevalence.csv")
    return dict(zip(prev["pathology"], prev["realistic_prevalence"]))


def reweighted_fn_ratio(df, flagged, is_fn, is_pos, w_pos, w_neg):
    w = np.where(is_pos, w_pos, w_neg)
    is_fp = (~is_pos) & (df["pred_label_thr_youden"] == 1).values
    FN_flag = ((is_fn & flagged) * w).sum()
    FP_flag = ((is_fp & flagged) * w).sum()
    return FN_flag / (FN_flag + FP_flag) * 100 if (FN_flag + FP_flag) else np.nan


def main():
    rng = np.random.default_rng(SEED)
    prevalence_map = load_prevalence()
    rows = []

    for pathology, prevalence in prevalence_map.items():
        df = pd.read_csv(DATA_DIR / f"merged_output_{pathology}_tiny.csv", sep=";")
        df.columns = df.columns.str.strip().str.lower()

        flagged_full = (df["ppw"] <= df["ppw"].quantile(0.05)).values
        is_pos_full = (df["true_label"] == 1).values
        is_fn_full = is_pos_full & (df["pred_label_thr_youden"] == 0).values
        w_pos, w_neg = prevalence / 0.5, (1 - prevalence) / 0.5
        point_estimate = reweighted_fn_ratio(df, flagged_full, is_fn_full, is_pos_full, w_pos, w_neg)

        subjects = df["subject_id"].unique()
        boot_vals = []
        for _ in range(N_BOOT):
            sampled_subj = rng.choice(subjects, size=len(subjects), replace=True)
            sub = df[df["subject_id"].isin(sampled_subj)].reset_index(drop=True)
            flagged_b = (sub["ppw"] <= sub["ppw"].quantile(0.05)).values
            is_pos_b = (sub["true_label"] == 1).values
            is_fn_b = is_pos_b & (sub["pred_label_thr_youden"] == 0).values
            val = reweighted_fn_ratio(sub, flagged_b, is_fn_b, is_pos_b, w_pos, w_neg)
            if not np.isnan(val):
                boot_vals.append(val)

        lo, hi = np.percentile(np.array(boot_vals), [2.5, 97.5])
        rows.append(dict(pathology=pathology, realistic_prevalence_pct=round(prevalence * 100, 2),
                          fn_ratio_reweighted_point=round(point_estimate, 2),
                          bootstrap_ci_lo=round(lo, 2), bootstrap_ci_hi=round(hi, 2)))
        print(f"{pathology:18s} point={point_estimate:6.2f}  95% CI [{lo:6.2f}, {hi:6.2f}]")

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "step6_bootstrap_fn_ratio_reweighted.csv", index=False)


if __name__ == "__main__":
    main()
