"""Step 1: realistic prevalence of each target pathology in the full
eligible MIMIC-CXR-CheXpert population, before 1:1 balancing.

Input: your_chexpert_labels_post_update.csv, one row per study, one column
per pathology (1.0/0.0/NaN/-1.0).
Output: output/step1_realistic_prevalence.csv
"""
import pandas as pd
from pathlib import Path

SOURCE = Path("your_chexpert_labels_post_update.csv")
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATHOLOGY_COLUMN_MAP = {
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "edema": "Edema",
    "pleuraleffusion": "Pleural Effusion",
    "pneumothorax": "Pneumothorax",
}


def main():
    df = pd.read_csv(SOURCE)
    n_total = len(df)
    print(f"Source file: {SOURCE.name}, {n_total} total studies")

    rows = []
    for key, col in PATHOLOGY_COLUMN_MAP.items():
        n_pos = int((df[col] == 1.0).sum())
        prevalence = n_pos / n_total
        rows.append(dict(pathology=key, column=col, n_positive=n_pos, n_total_source=n_total,
                          realistic_prevalence=round(prevalence, 4), realistic_prevalence_pct=round(prevalence * 100, 2)))
        print(f"  {key:18s}: {n_pos:6d} / {n_total} = {prevalence*100:.2f}%")

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "step1_realistic_prevalence.csv", index=False)


if __name__ == "__main__":
    main()
