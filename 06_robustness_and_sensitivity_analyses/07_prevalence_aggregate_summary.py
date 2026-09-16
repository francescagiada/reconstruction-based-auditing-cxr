"""Step 3: aggregate summary of the reweighted results, grouped by metric
and threshold.

Input: output/step2_reweighted_all_cells.csv
Output: output/step3_aggregate_comparison.csv
"""
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    df = pd.read_csv(OUTPUT_DIR / "step2_reweighted_all_cells.csv")

    agg = df.groupby(["metric", "threshold"]).agg(
        avg_fn_capture_balanced=("fn_capture_balanced", "mean"),
        avg_fn_capture_reweighted=("fn_capture_reweighted", "mean"),
        avg_fn_ratio_balanced=("fn_ratio_balanced", "mean"),
        avg_fn_ratio_reweighted=("fn_ratio_reweighted", "mean"),
        avg_fp_capture_balanced=("fp_capture_balanced", "mean"),
        avg_fp_capture_reweighted=("fp_capture_reweighted", "mean"),
        avg_flag_rate_balanced=("flag_rate_balanced", "mean"),
        avg_flag_rate_reweighted=("flag_rate_reweighted", "mean"),
    ).round(2).reset_index()

    agg.to_csv(OUTPUT_DIR / "step3_aggregate_comparison.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'step3_aggregate_comparison.csv'}")


if __name__ == "__main__":
    main()
