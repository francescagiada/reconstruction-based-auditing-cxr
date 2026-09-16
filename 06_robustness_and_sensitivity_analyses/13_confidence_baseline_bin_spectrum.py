"""Full-spectrum comparison against the classifier-confidence baseline,
in 100 equal-width bins of |pred_prob - 0.5|.

Input: per-pathology, per-autoencoder case CSVs (see
../05_figures_cost_benefit/csv_files/).
Output: output/bin_wise_confidence/<pathology>_<model>_confidence_100bins.csv, .png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "05_figures_cost_benefit", "csv_files"))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "bin_wise_confidence")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = [
    "merged_output_atelectasis_tiny", "merged_output_atelectasis_vae",
    "merged_output_cardiomegaly_tiny", "merged_output_cardiomegaly_vae",
    "merged_output_edema_tiny", "merged_output_edema_vae",
    "merged_output_pleuraleffusion_tiny", "merged_output_pleuraleffusion_vae",
    "merged_output_pneumothorax_tiny", "merged_output_pneumothorax_vae",
]
N_BINS = 100


def compute_hist_counts(values_all, values_group, bins):
    edges = np.histogram_bin_edges(values_all, bins=bins)
    counts_group, _ = np.histogram(values_group, bins=edges)
    return edges, counts_group


def analyze_file(df, pathology, model):
    true_label, pred_label = df["true_label"], df["pred_label_thr_youden"]
    uncertainty = (df["pred_prob"] - 0.5).abs()

    is_tp = (pred_label == 1) & (true_label == 1)
    is_tn = (pred_label == 0) & (true_label == 0)
    is_fp = (pred_label == 1) & (true_label == 0)
    is_fn = (pred_label == 0) & (true_label == 1)

    edges, tp_counts = compute_hist_counts(uncertainty, uncertainty[is_tp], N_BINS)
    _, tn_counts = compute_hist_counts(uncertainty, uncertainty[is_tn], N_BINS)
    _, fp_counts = compute_hist_counts(uncertainty, uncertainty[is_fp], N_BINS)
    _, fn_counts = compute_hist_counts(uncertainty, uncertainty[is_fn], N_BINS)

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
    out.to_csv(os.path.join(OUTPUT_DIR, f"{pathology}_{model}_confidence_100bins.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    bar_width = (edges[1] - edges[0]) * 0.9
    ax.bar(bin_centers, tn_counts, width=bar_width, color="#4C72B0", label="TN")
    ax.bar(bin_centers, tp_counts, width=bar_width, bottom=tn_counts, color="#55A868", label="TP")
    ax.bar(bin_centers, fp_counts, width=bar_width, bottom=tn_counts + tp_counts, color="#DD8452", label="FP")
    ax.bar(bin_centers, fn_counts, width=bar_width, bottom=tn_counts + tp_counts + fp_counts, color="#C44E52", label="FN")
    ax.set_title(f"{pathology} / {model}: TP/TN/FP/FN across confidence spectrum |p_hat-0.5| ({N_BINS} bins)")
    ax.set_xlabel("|p_hat - 0.5|  (0 = maximum uncertainty, 0.5 = maximum confidence)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{pathology}_{model}_confidence_100bins.png"), dpi=150)
    plt.close(fig)
    return out


def main():
    summaries = []
    for fname in FILES:
        df = pd.read_csv(os.path.join(DATA_DIR, fname + ".csv"), sep=";")
        pathology, model = fname.replace("merged_output_", "").rsplit("_", 1)
        print(f"{pathology} / {model} ...")
        out = analyze_file(df, pathology, model)

        lo, hi = out["bin_left"].min(), out["bin_right"].max()
        rng = hi - lo
        fn_pos = np.average(out["bin_center"], weights=out["fn_count"]) if out["fn_count"].sum() > 0 else np.nan
        fp_pos = np.average(out["bin_center"], weights=out["fp_count"]) if out["fp_count"].sum() > 0 else np.nan
        summaries.append({
            "pathology": pathology, "model": model,
            "total_fn": int(out["fn_count"].sum()), "total_fp": int(out["fp_count"].sum()),
            "fn_norm_position": (fn_pos - lo) / rng, "fp_norm_position": (fp_pos - lo) / rng,
        })

    pd.DataFrame(summaries).to_csv(os.path.join(OUTPUT_DIR, "_summary_confidence_bin_positions.csv"), index=False)
    print(f"\nSaved summary to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
