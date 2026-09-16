"""Bin-wise TP/TN/FP/FN counts across the full reconstruction score
distribution, 100 equal-width bins per (pathology, autoencoder, metric).

Input: per-pathology, per-autoencoder case CSVs (see
../05_figures_cost_benefit/csv_files/).
Output: output/bin_wise_counts/<pathology>_<model>_<metric>_100bins.csv, .png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "05_figures_cost_benefit", "csv_files"))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "bin_wise_counts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = [
    "merged_output_atelectasis_tiny", "merged_output_atelectasis_vae",
    "merged_output_cardiomegaly_tiny", "merged_output_cardiomegaly_vae",
    "merged_output_edema_tiny", "merged_output_edema_vae",
    "merged_output_pleuraleffusion_tiny", "merged_output_pleuraleffusion_vae",
    "merged_output_pneumothorax_tiny", "merged_output_pneumothorax_vae",
]
RECONSTRUCTION_METRICS = ["mse", "ssim", "ppw", "edge_diff_norm"]
N_BINS = 100


def compute_hist_counts(values_all, values_group, bins):
    edges = np.histogram_bin_edges(values_all, bins=bins)
    counts_group, _ = np.histogram(values_group, bins=edges)
    return edges, counts_group


def analyze_file_metric(df, pathology, model, metric):
    true_label = df["true_label"]
    pred_label = df["pred_label_thr_youden"]
    is_tp = (pred_label == 1) & (true_label == 1)
    is_tn = (pred_label == 0) & (true_label == 0)
    is_fp = (pred_label == 1) & (true_label == 0)
    is_fn = (pred_label == 0) & (true_label == 1)

    vals = df[metric].astype(float)
    edges, tp_counts = compute_hist_counts(vals, vals[is_tp], N_BINS)
    _, tn_counts = compute_hist_counts(vals, vals[is_tn], N_BINS)
    _, fp_counts = compute_hist_counts(vals, vals[is_fp], N_BINS)
    _, fn_counts = compute_hist_counts(vals, vals[is_fn], N_BINS)

    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    total_counts = tp_counts + tn_counts + fp_counts + fn_counts
    total_safe = np.where(total_counts == 0, 1, total_counts)

    out = pd.DataFrame({
        "bin_center": bin_centers, "bin_left": edges[:-1], "bin_right": edges[1:],
        "tp_count": tp_counts, "tn_count": tn_counts, "fp_count": fp_counts, "fn_count": fn_counts,
        "total_count": total_counts,
        "p_tp": tp_counts / total_safe, "p_tn": tn_counts / total_safe,
        "p_fp": fp_counts / total_safe, "p_fn": fn_counts / total_safe,
    })
    out.to_csv(os.path.join(OUTPUT_DIR, f"{pathology}_{model}_{metric}_100bins.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    bar_width = (edges[1] - edges[0]) * 0.9
    ax.bar(bin_centers, tn_counts, width=bar_width, color="#4C72B0", label="TN")
    ax.bar(bin_centers, tp_counts, width=bar_width, bottom=tn_counts, color="#55A868", label="TP")
    ax.bar(bin_centers, fp_counts, width=bar_width, bottom=tn_counts + tp_counts, color="#DD8452", label="FP")
    ax.bar(bin_centers, fn_counts, width=bar_width, bottom=tn_counts + tp_counts + fp_counts, color="#C44E52", label="FN")
    ax.set_title(f"{pathology} / {model} / {metric.upper()}: TP/TN/FP/FN across {N_BINS} bins")
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{pathology}_{model}_{metric}_100bins.png"), dpi=150)
    plt.close(fig)
    return out


def main():
    for fname in FILES:
        df = pd.read_csv(os.path.join(DATA_DIR, fname + ".csv"), sep=";")
        pathology, model = fname.replace("merged_output_", "").rsplit("_", 1)
        for metric in RECONSTRUCTION_METRICS:
            print(f"{pathology} / {model} / {metric} ...")
            analyze_file_metric(df, pathology, model, metric)
    print(f"\nCSV and PNG per combination saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
