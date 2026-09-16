"""Sensitivity of FN capture to excluding the test-split label-update
cases, for PPW at both review thresholds and both autoencoder architectures.

Input:
    output/label_updates_full_list.csv (from the sibling script)
    per-pathology, per-autoencoder case CSVs (see
    ../05_figures_cost_benefit/csv_files/)
Output:
    output/sensitivity_excluding_updates_ppw.csv
"""
import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "05_figures_cost_benefit", "csv_files"))
UPDATES_PATH = os.path.join(SCRIPT_DIR, "output", "label_updates_full_list.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "output")

PATHOLOGY_FILE = {
    "Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Edema": "edema",
    "Pleural Effusion": "pleuraleffusion", "Pneumothorax": "pneumothorax",
}
THRESHOLDS = {"1pct": 0.01, "5pct": 0.05}


def get_tail_positions(values, threshold):
    n = int(np.ceil(len(values) * threshold))
    return np.argsort(values)[:n]  # ppw: lower = worse reconstruction


def fn_capture(true_label, pred_label, flagged_pos):
    is_fn = (true_label == 1) & (pred_label == 0)
    total_fn = is_fn.sum()
    flagged_fn = is_fn[flagged_pos].sum()
    return flagged_fn, total_fn, (100 * flagged_fn / total_fn if total_fn > 0 else np.nan)


def main():
    updates = pd.read_csv(UPDATES_PATH)
    updates_test = updates[updates["split"] == "test"]

    rows = []
    for pathology_col, folder in PATHOLOGY_FILE.items():
        excl = set(zip(
            updates_test.loc[updates_test.pathology_col == pathology_col, "subject_id"],
            updates_test.loc[updates_test.pathology_col == pathology_col, "study_id"],
        ))
        for model in ["tiny", "vae"]:
            df = pd.read_csv(os.path.join(DATA_DIR, f"merged_output_{folder}_{model}.csv"), sep=";")
            true_label = df["true_label"].to_numpy()
            pred_label = df["pred_label_thr_youden"].to_numpy()
            values = df["ppw"].to_numpy()

            is_excluded = np.array([(sid, stid) in excl for sid, stid in zip(df["subject_id"], df["study_id"])])

            for thr_label, thr in THRESHOLDS.items():
                flagged_pos = get_tail_positions(values, thr)
                flagged_fn_orig, total_fn_orig, capture_orig = fn_capture(true_label, pred_label, flagged_pos)

                keep = ~is_excluded
                true_k, pred_k, values_k = true_label[keep], pred_label[keep], values[keep]
                flagged_pos_k = get_tail_positions(values_k, thr)
                flagged_fn_k, total_fn_k, capture_k = fn_capture(true_k, pred_k, flagged_pos_k)

                rows.append({
                    "pathology": pathology_col, "model": model, "threshold": thr_label,
                    "n_excluded_cases": int(is_excluded.sum()),
                    "fn_capture_pct_original": round(capture_orig, 3),
                    "fn_capture_pct_excluding_updates": round(capture_k, 3),
                    "abs_diff_pct_points": round(abs(capture_orig - capture_k), 3),
                    "total_fn_original": int(total_fn_orig), "total_fn_excluding": int(total_fn_k),
                })

    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, "sensitivity_excluding_updates_ppw.csv"), index=False)
    print(out.to_string(index=False))
    print(f"\nMax absolute shift across all rows: {out['abs_diff_pct_points'].max():.3f} percentage points")


if __name__ == "__main__":
    main()
