"""Step 4: cross-check of Step 2 via actual resampling simulation (PPW,
5% threshold, TinyAE, all 5 pathologies, 500 repetitions per cell).

Input:
    output/step1_realistic_prevalence.csv, output/step2_reweighted_all_cells.csv
    per-pathology TinyAE case CSVs (see ../05_figures_cost_benefit/csv_files/)
Output:
    output/step4_simulation_crosscheck.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / ".." / "05_figures_cost_benefit" / "csv_files"
OUTPUT_DIR = SCRIPT_DIR / "output"
N_SIM = 500
SEED = 42


def load_prevalence():
    prev = pd.read_csv(OUTPUT_DIR / "step1_realistic_prevalence.csv")
    return dict(zip(prev["pathology"], prev["realistic_prevalence"]))


def simulate_cell(df, metric, direction, thr, prevalence, rng):
    if direction == "low":
        flagged = (df[metric] <= df[metric].quantile(thr)).values
    else:
        flagged = (df[metric] >= df[metric].quantile(1 - thr)).values

    is_pos = (df["true_label"] == 1).values
    is_neg = ~is_pos
    pos_idx, neg_idx = np.where(is_pos)[0], np.where(is_neg)[0]
    n_neg = len(neg_idx)

    # Keep all negatives, subsample positives so pos/(pos+neg) == prevalence.
    n_pos_target = min(int(round(n_neg * prevalence / (1 - prevalence))), len(pos_idx))

    is_fn = is_pos & (df["pred_label_thr_youden"] == 0).values
    is_fp = is_neg & (df["pred_label_thr_youden"] == 1).values

    fn_captures, fn_ratios = [], []
    for _ in range(N_SIM):
        sampled_pos = rng.choice(pos_idx, size=n_pos_target, replace=False)
        keep = np.zeros(len(df), dtype=bool)
        keep[sampled_pos] = True
        keep[neg_idx] = True

        FN_total, FP_total = (is_fn & keep).sum(), (is_fp & keep).sum()
        FN_flag, FP_flag = (is_fn & keep & flagged).sum(), (is_fp & keep & flagged).sum()

        fn_captures.append(FN_flag / FN_total * 100 if FN_total else np.nan)
        fn_ratios.append(FN_flag / (FN_flag + FP_flag) * 100 if (FN_flag + FP_flag) else np.nan)

    return np.nanmean(fn_captures), np.nanmean(fn_ratios)


def main():
    rng = np.random.default_rng(SEED)
    prevalence_map = load_prevalence()
    step2 = pd.read_csv(OUTPUT_DIR / "step2_reweighted_all_cells.csv")

    rows = []
    for pathology, prevalence in prevalence_map.items():
        df = pd.read_csv(DATA_DIR / f"merged_output_{pathology}_tiny.csv", sep=";")
        df.columns = df.columns.str.strip().str.lower()

        sim_fn_capture, sim_fn_ratio = simulate_cell(df, "ppw", "low", 0.05, prevalence, rng)
        ref = step2[(step2.pathology == pathology) & (step2.autoencoder == "tiny") &
                     (step2.metric == "ppw") & (step2.threshold == "5pct")].iloc[0]

        rows.append(dict(
            pathology=pathology, realistic_prevalence_pct=round(prevalence * 100, 2),
            fn_capture_reweighted=ref["fn_capture_reweighted"], fn_capture_simulated=round(sim_fn_capture, 3),
            fn_ratio_reweighted=ref["fn_ratio_reweighted"], fn_ratio_simulated=round(sim_fn_ratio, 3),
        ))

    out = pd.DataFrame(rows)
    out["fn_capture_diff"] = (out["fn_capture_reweighted"] - out["fn_capture_simulated"]).round(3)
    out["fn_ratio_diff"] = (out["fn_ratio_reweighted"] - out["fn_ratio_simulated"]).round(3)
    out.to_csv(OUTPUT_DIR / "step4_simulation_crosscheck.csv", index=False)

    print(f"Max |diff| FN_capture: {out['fn_capture_diff'].abs().max():.3f} pct points")
    print(f"Max |diff| FN_ratio:   {out['fn_ratio_diff'].abs().max():.3f} pct points")
    print(f"(N_SIM={N_SIM} draws per cell; small differences are expected finite-sample noise)")


if __name__ == "__main__":
    main()
