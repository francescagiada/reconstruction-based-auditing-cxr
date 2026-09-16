"""Step 2: prevalence-reweighted FN capture / FN ratio / FP capture / FN
yield via importance reweighting (w_pos = p/0.5, w_neg = (1-p)/0.5).

Input:
    output/step1_realistic_prevalence.csv
    per-pathology, per-autoencoder case CSVs (see
    ../05_figures_cost_benefit/csv_files/)
Output:
    output/step2_reweighted_all_cells.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / ".." / "05_figures_cost_benefit" / "csv_files"
OUTPUT_DIR = SCRIPT_DIR / "output"

PATHOLOGIES = ["atelectasis", "cardiomegaly", "edema", "pleuraleffusion", "pneumothorax"]
AUTOENCODERS = ["tiny", "vae"]
METRIC_DIRECTION = {"mse": "high", "ssim": "low", "ppw": "low", "edge_diff_norm": "high"}
THRESHOLDS = {"1pct": 0.01, "5pct": 0.05}


def load_prevalence():
    prev = pd.read_csv(OUTPUT_DIR / "step1_realistic_prevalence.csv")
    return dict(zip(prev["pathology"], prev["realistic_prevalence"]))


def compute_cell(df, metric, direction, thr, prevalence):
    if direction == "low":
        flagged = df[metric] <= df[metric].quantile(thr)
    else:
        flagged = df[metric] >= df[metric].quantile(1 - thr)

    is_pos, is_neg = df["true_label"] == 1, df["true_label"] == 0
    is_fn = is_pos & (df["pred_label_thr_youden"] == 0)
    is_fp = is_neg & (df["pred_label_thr_youden"] == 1)

    w_pos, w_neg = prevalence / 0.5, (1 - prevalence) / 0.5
    w = np.where(is_pos, w_pos, w_neg)

    # Unweighted (balanced test set).
    FN_total_bal, FP_total_bal = is_fn.sum(), is_fp.sum()
    FN_flag_bal, FP_flag_bal = (is_fn & flagged).sum(), (is_fp & flagged).sum()
    N_flag_bal, N_total_bal = flagged.sum(), len(df)

    fn_capture_bal = FN_flag_bal / FN_total_bal if FN_total_bal else np.nan
    fp_capture_bal = FP_flag_bal / FP_total_bal if FP_total_bal else np.nan
    fn_ratio_bal = FN_flag_bal / (FN_flag_bal + FP_flag_bal) if (FN_flag_bal + FP_flag_bal) else np.nan
    fn_yield_bal = FN_flag_bal / N_flag_bal if N_flag_bal else np.nan
    flag_rate_bal = N_flag_bal / N_total_bal

    # Reweighted (realistic prevalence).
    FN_total_rw = (is_fn * w).sum()
    FP_total_rw = (is_fp * w).sum()
    FN_flag_rw = ((is_fn & flagged) * w).sum()
    FP_flag_rw = ((is_fp & flagged) * w).sum()
    N_flag_rw, N_total_rw = (flagged * w).sum(), w.sum()

    fn_capture_rw = FN_flag_rw / FN_total_rw if FN_total_rw else np.nan
    fp_capture_rw = FP_flag_rw / FP_total_rw if FP_total_rw else np.nan
    fn_ratio_rw = FN_flag_rw / (FN_flag_rw + FP_flag_rw) if (FN_flag_rw + FP_flag_rw) else np.nan
    fn_yield_rw = FN_flag_rw / N_flag_rw if N_flag_rw else np.nan
    flag_rate_rw = N_flag_rw / N_total_rw

    return dict(
        fn_capture_balanced=round(fn_capture_bal * 100, 3), fn_capture_reweighted=round(fn_capture_rw * 100, 3),
        fp_capture_balanced=round(fp_capture_bal * 100, 3), fp_capture_reweighted=round(fp_capture_rw * 100, 3),
        fn_ratio_balanced=round(fn_ratio_bal * 100, 3), fn_ratio_reweighted=round(fn_ratio_rw * 100, 3),
        fn_yield_balanced=round(fn_yield_bal * 100, 3), fn_yield_reweighted=round(fn_yield_rw * 100, 3),
        flag_rate_balanced=round(flag_rate_bal * 100, 3), flag_rate_reweighted=round(flag_rate_rw * 100, 3),
    )


def main():
    prevalence_map = load_prevalence()
    rows = []
    for pathology in PATHOLOGIES:
        prevalence = prevalence_map[pathology]
        for ae in AUTOENCODERS:
            df = pd.read_csv(DATA_DIR / f"merged_output_{pathology}_{ae}.csv", sep=";")
            df.columns = df.columns.str.strip().str.lower()
            for metric, direction in METRIC_DIRECTION.items():
                for thr_name, thr in THRESHOLDS.items():
                    cell = compute_cell(df, metric, direction, thr, prevalence)
                    rows.append(dict(pathology=pathology, autoencoder=ae, metric=metric, threshold=thr_name,
                                      realistic_prevalence_pct=round(prevalence * 100, 2), **cell))

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "step2_reweighted_all_cells.csv", index=False)
    print(f"Saved {len(out)} rows.")

    max_fn_diff = (out["fn_capture_balanced"] - out["fn_capture_reweighted"]).abs().max()
    max_fp_diff = (out["fp_capture_balanced"] - out["fp_capture_reweighted"]).abs().max()
    print(f"Max |FN_capture_balanced - FN_capture_reweighted| across all cells: {max_fn_diff:.6f} pct points")
    print(f"Max |FP_capture_balanced - FP_capture_reweighted| across all cells: {max_fp_diff:.6f} pct points")
    print("(should be ~0, confirming the invariance argument above)")


if __name__ == "__main__":
    main()
