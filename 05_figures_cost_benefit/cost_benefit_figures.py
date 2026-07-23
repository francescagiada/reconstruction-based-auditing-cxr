"""Cost-benefit and enrichment figures for reconstruction-based FN/FP prioritization.

Pipeline: for each (pathology, autoencoder, metric) combination, cases are ranked by
reconstruction score and split into a direction-corrected tail (see METRIC_DIRECTION
below) at the 1% and 5% percentile thresholds. Confusion-matrix counts within each tail
are aggregated into false-negative/false-positive capture rates, which drive the two
published figures (cost-benefit curves, FN enrichment) and the baseline classification
performance summary.

Input: per-pathology, per-autoencoder CSVs in BASE_DIR, one row per case, with columns
true_label, pred_label_thr_youden, pred_prob, and the four reconstruction metrics
(mse, ssim, ppw, edge_diff_norm).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score

BASE_DIR = Path(r"./csv_files")
METRICS = ["mse", "ssim", "ppw", "edge_diff_norm"]

# Direction of "worse reconstruction" per metric: mse/edge_diff_norm are error measures
# (higher = worse, so the flagged tail is the right/high tail); ssim/ppw are similarity
# measures (lower = worse, so the flagged tail is the left/low tail).
METRIC_DIRECTION = {"mse": "high", "ssim": "low", "ppw": "low", "edge_diff_norm": "high"}
TAIL_FOR_DIRECTION = {"high": "right", "low": "left"}


def load_case_tables(base_dir: Path) -> list[tuple[str, str, pd.DataFrame]]:
    """Load each per-pathology/per-autoencoder CSV, keyed by (pathology, autoencoder)."""
    tables = []
    for csv_path in base_dir.glob("*.csv"):
        df = pd.read_csv(csv_path, sep=";")
        df.columns = df.columns.str.strip().str.lower()
        pathology, autoencoder = csv_path.stem.split("_")[-2:]
        tables.append((pathology.lower(), autoencoder.lower(), df))
    return tables


def tail_confusion_counts(tables: list[tuple[str, str, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Confusion-matrix counts within each percentile tail, plus whole-dataset FN/FP totals.

    Whole-dataset totals are computed once per (pathology, autoencoder), not summed across
    the per-metric/per-tail bins below: summing there would count each case once per bin it
    falls into (4 metrics x 2 tails), inflating the denominator used for capture rates.
    """
    tail_rows, size_rows = [], []
    for pathology, autoencoder, df in tables:
        y_true, y_pred = df["true_label"], df["pred_label_thr_youden"]
        total_fn = int(((y_true == 1) & (y_pred == 0)).sum())
        total_fp = int(((y_true == 0) & (y_pred == 1)).sum())
        size_rows.append({
            "pathology": pathology, "autoencoder": autoencoder,
            "total_images": len(df), "total_fn": total_fn, "total_fp": total_fp,
        })

        for metric in METRICS:
            values = df[metric].dropna()
            low_1, high_1 = values.quantile(0.01), values.quantile(0.99)
            low_5, high_5 = values.quantile(0.05), values.quantile(0.95)
            tails = {
                ("left", "0_1"): df[metric] < low_1,
                ("right", "0_1"): df[metric] > high_1,
                ("left", "1_5"): (df[metric] >= low_1) & (df[metric] < low_5),
                ("right", "1_5"): (df[metric] <= high_1) & (df[metric] > high_5),
            }
            for (tail, severity), condition in tails.items():
                df_tail = df[condition]
                if df_tail.empty or df_tail["true_label"].nunique() < 2:
                    continue
                tn, fp, fn, tp = confusion_matrix(
                    df_tail["true_label"], df_tail["pred_label_thr_youden"], labels=[0, 1]
                ).ravel()
                tail_rows.append({
                    "pathology": pathology, "autoencoder": autoencoder, "metric": metric,
                    "tail": tail, "severity": severity, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                })
    return pd.DataFrame(tail_rows), pd.DataFrame(size_rows)


def build_operational_table(tail_counts: pd.DataFrame, dataset_sizes: pd.DataFrame) -> pd.DataFrame:
    """FN/FP capture and ratio at the 1% and 5% thresholds, one row per (pathology, autoencoder, metric, tail)."""
    rows = []
    for (pathology, autoencoder), group in tail_counts.groupby(["pathology", "autoencoder"]):
        size_row = dataset_sizes[
            (dataset_sizes["pathology"] == pathology) & (dataset_sizes["autoencoder"] == autoencoder)
        ].iloc[0]
        total_images, total_fn, total_fp = size_row["total_images"], size_row["total_fn"], size_row["total_fp"]

        for (metric, tail), df_mt in group.groupby(["metric", "tail"]):
            df_1 = df_mt[df_mt["severity"] == "0_1"]
            df_5 = df_mt[df_mt["severity"].isin(["0_1", "1_5"])]
            fn_1, fp_1, n_1 = df_1["FN"].sum(), df_1["FP"].sum(), df_1[["TN", "TP", "FP", "FN"]].sum().sum()
            fn_5, fp_5, n_5 = df_5["FN"].sum(), df_5["FP"].sum(), df_5[["TN", "TP", "FP", "FN"]].sum().sum()
            rows.append({
                "pathology": pathology, "autoencoder": autoencoder, "metric": metric, "tail": tail,
                "flag_rate_1pct": n_1 / total_images, "flag_rate_5pct": n_5 / total_images,
                "FN_capture_1pct": fn_1 / total_fn if total_fn > 0 else 0,
                "FN_capture_5pct": fn_5 / total_fn if total_fn > 0 else 0,
                "FP_capture_1pct": fp_1 / total_fp if total_fp > 0 else 0,
                "FP_capture_5pct": fp_5 / total_fp if total_fp > 0 else 0,
                "FN_ratio_1pct": fn_1 / (fn_1 + fp_1) if (fn_1 + fp_1) > 0 else 0,
                "FN_ratio_5pct": fn_5 / (fn_5 + fp_5) if (fn_5 + fp_5) > 0 else 0,
            })
    return pd.DataFrame(rows)


def apply_direction_correction(operational_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the direction-corrected tail per metric (see METRIC_DIRECTION).

    operational_df has both the left and right tail for every metric; averaging both
    together, instead of keeping only the tail associated with worse reconstruction
    quality for that metric, inflates FN-capture and enrichment figures (this was the
    source of a two-tailed-pooling error in an earlier version of this pipeline, fixed
    here by filtering to the correct tail before any further aggregation).
    """
    worst_tail = operational_df["metric"].map(METRIC_DIRECTION).map(TAIL_FOR_DIRECTION)
    return operational_df[operational_df["tail"] == worst_tail].copy()


def plot_cost_benefit(tradeoff_df: pd.DataFrame, out_dir: Path) -> None:
    """Cost-benefit curves: false-negative capture rate vs. percentage of cases flagged for review."""
    out_dir.mkdir(exist_ok=True)
    for autoencoder in tradeoff_df["autoencoder"].unique():
        df_ae = tradeoff_df[tradeoff_df["autoencoder"] == autoencoder]
        plt.figure(figsize=(8, 6))
        for metric in df_ae["metric"].unique():
            df_m = df_ae[df_ae["metric"] == metric]
            fn_1, fn_5 = df_m["FN_capture_1pct"].mean(), df_m["FN_capture_5pct"].mean()
            plt.plot([1, 5], [fn_1 * 100, fn_5 * 100], marker="o", label=metric.upper())
        plt.xlabel("Percentage of Cases Flagged for Review (%)")
        plt.ylabel("Percentage of Total False Negatives Intercepted (%)")
        plt.title(f"Cost–Benefit Curve ({autoencoder.upper()})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / f"cost_benefit_{autoencoder}.png", dpi=300)
        plt.close()


def plot_enrichment(tradeoff_df: pd.DataFrame, out_dir: Path) -> None:
    """FN enrichment: proportion of flagged errors that are false negatives, at the 1% threshold."""
    out_dir.mkdir(exist_ok=True)
    for autoencoder in tradeoff_df["autoencoder"].unique():
        df_ae = tradeoff_df[tradeoff_df["autoencoder"] == autoencoder]
        enrichment = df_ae.groupby("metric")["FN_ratio_1pct"].mean().sort_values(ascending=False)
        plt.figure(figsize=(8, 6))
        bars = plt.bar(enrichment.index.str.upper(), enrichment.values * 100)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}%", ha="center")
        plt.ylabel("Proportion of Flagged Errors that are False Negatives (%)")
        plt.title(f"False Negative Enrichment (1% Threshold) – {autoencoder.upper()}")
        plt.ylim(0, 100)
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(out_dir / f"enrichment_{autoencoder}.png", dpi=300)
        plt.close()


def compute_baseline_performance(base_dir: Path) -> pd.DataFrame:
    """Accuracy, sensitivity, specificity, and AUC per (pathology, autoencoder), at the Youden threshold."""
    rows = []
    for csv_path in base_dir.glob("*.csv"):
        df = pd.read_csv(csv_path, sep=";")
        df.columns = df.columns.str.strip().str.lower()
        pathology, autoencoder = csv_path.stem.split("_")[-2:]
        y_true, y_pred, y_prob = df["true_label"], df["pred_label_thr_youden"], df["pred_prob"]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "pathology": pathology.lower(), "autoencoder": autoencoder.lower(), "n_samples": len(df),
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
            "auc": roc_auc_score(y_true, y_prob),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })
    return pd.DataFrame(rows).sort_values(["pathology", "autoencoder"])


def main():
    tables = load_case_tables(BASE_DIR)
    tail_counts, dataset_sizes = tail_confusion_counts(tables)
    operational_df = build_operational_table(tail_counts, dataset_sizes)
    tradeoff_df = apply_direction_correction(operational_df)
    tradeoff_df.to_csv("tradeoff_table.csv", index=False)

    plot_cost_benefit(tradeoff_df, Path("cost_benefit_plots"))
    plot_enrichment(tradeoff_df, Path("enrichment_plots"))

    baseline_df = compute_baseline_performance(BASE_DIR)
    baseline_df.to_csv("baseline_performance_summary.csv", index=False)


if __name__ == "__main__":
    main()
