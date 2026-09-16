"""Patient-stratified bootstrap 95% CIs for baseline classifier performance
(Accuracy, Sensitivity, Specificity, AUC) at the Youden-optimized threshold.

Input: per-pathology case CSVs (see ../05_figures_cost_benefit/csv_files/),
columns subject_id, true_label, pred_prob, pred_label_thr_youden.
Output: output/baseline_metrics_bootstrap_ci.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).resolve().parent / ".." / "05_figures_cost_benefit" / "csv_files"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATHOLOGIES = ["atelectasis", "cardiomegaly", "edema", "pleuraleffusion", "pneumothorax"]
N_BOOT = 1000
SEED = 42


def compute_metrics(df):
    y_true, y_pred, y_prob = df["true_label"].values, df["pred_label_thr_youden"].values, df["pred_prob"].values
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    except ValueError:
        auc = np.nan
    return acc, sens, spec, auc


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    for pathology in PATHOLOGIES:
        df = pd.read_csv(DATA_DIR / f"merged_output_{pathology}_tiny.csv", sep=";")
        df.columns = df.columns.str.strip().str.lower()
        point_acc, point_sens, point_spec, point_auc = compute_metrics(df)

        subjects = df["subject_id"].unique()
        boot_acc, boot_sens, boot_spec, boot_auc = [], [], [], []
        for _ in range(N_BOOT):
            sampled_subj = rng.choice(subjects, size=len(subjects), replace=True)
            a, s, sp, auc = compute_metrics(df[df["subject_id"].isin(sampled_subj)])
            boot_acc.append(a); boot_sens.append(s); boot_spec.append(sp); boot_auc.append(auc)

        def ci(vals):
            vals = np.array([v for v in vals if not np.isnan(v)])
            return np.percentile(vals, [2.5, 97.5])

        acc_lo, acc_hi = ci(boot_acc)
        sens_lo, sens_hi = ci(boot_sens)
        spec_lo, spec_hi = ci(boot_spec)
        auc_lo, auc_hi = ci(boot_auc)

        rows.append(dict(
            pathology=pathology,
            accuracy=round(point_acc * 100, 1), accuracy_ci_lo=round(acc_lo * 100, 1), accuracy_ci_hi=round(acc_hi * 100, 1),
            sensitivity=round(point_sens * 100, 1), sensitivity_ci_lo=round(sens_lo * 100, 1), sensitivity_ci_hi=round(sens_hi * 100, 1),
            specificity=round(point_spec * 100, 1), specificity_ci_lo=round(spec_lo * 100, 1), specificity_ci_hi=round(spec_hi * 100, 1),
            auc=round(point_auc, 3), auc_ci_lo=round(auc_lo, 3), auc_ci_hi=round(auc_hi, 3),
        ))
        print(f"{pathology:18s} AUC={point_auc:.3f} [{auc_lo:.3f},{auc_hi:.3f}]")

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "baseline_metrics_bootstrap_ci.csv", index=False)


if __name__ == "__main__":
    main()
