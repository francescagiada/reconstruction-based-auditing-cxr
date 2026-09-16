"""Demographic stratification (sex, age, race) of the flagged review
subset, via MIMIC-IV linkage.

Input:
  - per-pathology, per-autoencoder case CSVs (see
    ../05_figures_cost_benefit/csv_files/): subject_id, the four
    reconstruction metrics, true_label, pred_label_thr_youden.
  - your_mimic_iv_admissions.csv: subject_id, race, admittime.
  - your_mimic_iv_patients.csv: subject_id, gender, anchor_age.
  - your_mimic_cxr_metadata.csv: subject_id, study_id, dicom_id, ViewPosition.
Output: output/part1_chi2_tests_BH.csv, part1_proportions.csv,
part2_sex_all_metrics_thresholds.csv, part3_case_mix_decomposition.csv,
part4_bootstrap_ppw_sex_OR.csv, part5_view_position_control.csv,
part6_age_stratified.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / ".." / "05_figures_cost_benefit" / "csv_files"
MIMIC_IV_ADMISSIONS = Path("your_mimic_iv_admissions.csv")
MIMIC_IV_PATIENTS = Path("your_mimic_iv_patients.csv")
CXR_METADATA = Path("your_mimic_cxr_metadata.csv")
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATHOLOGIES = ["atelectasis", "cardiomegaly", "edema", "pleuraleffusion", "pneumothorax"]
AUTOENCODERS = ["tiny", "vae"]
METRIC_DIRECTION = {"mse": "high", "ssim": "low", "ppw": "low", "edge_diff_norm": "high"}
THRESHOLDS = {"1pct": 0.01, "5pct": 0.05}


# ---------------------------------------------------------------------------
# Demographic maps
# ---------------------------------------------------------------------------

def build_race_map():
    adm = pd.read_csv(MIMIC_IV_ADMISSIONS, usecols=["subject_id", "race", "admittime"])
    adm["admittime"] = pd.to_datetime(adm["admittime"])

    def to_bucket(r):
        r = str(r).upper()
        if r.startswith("WHITE"):
            return "White"
        if r.startswith("BLACK"):
            return "Black"
        if r.startswith("ASIAN"):
            return "Asian"
        return "Unknown/Other"

    adm["race_bucket"] = adm["race"].apply(to_bucket)

    def resolve(group):
        counts = group["race_bucket"].value_counts()
        top = counts[counts == counts.max()].index
        if len(top) == 1:
            return top[0]
        return group.sort_values("admittime").iloc[-1]["race_bucket"]

    race_map = adm.groupby("subject_id", group_keys=False).apply(resolve, include_groups=False)
    race_map.name = "race_bucket"
    return race_map.reset_index()


def build_demo_map():
    pat = pd.read_csv(MIMIC_IV_PATIENTS, usecols=["subject_id", "gender", "anchor_age"])
    assert pat["subject_id"].is_unique, "duplicate subject_id in patients table would corrupt every join below"

    def age_band(a):
        if pd.isna(a):
            return "Unknown"
        if a < 40:
            return "<40"
        if a < 60:
            return "40-59"
        if a < 80:
            return "60-79"
        return "80+"

    pat["age_band"] = pat["anchor_age"].apply(age_band)
    return pat[["subject_id", "gender", "anchor_age", "age_band"]]


def build_view_map():
    meta = pd.read_csv(CXR_METADATA, usecols=["dicom_id", "ViewPosition"])
    meta["dicom_id"] = meta["dicom_id"].astype(str)
    return meta


def load_test_csv(pathology, ae, race_map, demo_map, view_map):
    df = pd.read_csv(DATA_DIR / f"merged_output_{pathology}_{ae}.csv", sep=";")
    df.columns = df.columns.str.strip().str.lower()
    n_raw = len(df)

    df["dicom_id"] = df["filename"].str.replace(".jpg", "", regex=False)
    df = df.merge(race_map, on="subject_id", how="left")
    df = df.merge(demo_map, on="subject_id", how="left")
    df = df.merge(view_map, on="dicom_id", how="left")
    assert len(df) == n_raw, "row count changed after merge -- a join key is duplicated somewhere"

    df["race_bucket"] = df["race_bucket"].fillna("Unknown/Other")
    df["gender"] = df["gender"].fillna("Unknown")
    df["age_band"] = df["age_band"].fillna("Unknown")
    return df


# ---------------------------------------------------------------------------
# Part 1: chi-square disparity tests, PPW, all pathologies x architectures x
# thresholds x demographic dimensions
# ---------------------------------------------------------------------------

def chi_square_vs_full(full_counts, flagged_counts, categories):
    full = np.array([full_counts.get(c, 0) for c in categories])
    flagged = np.array([flagged_counts.get(c, 0) for c in categories])
    non_flagged = full - flagged
    table = np.array([flagged, non_flagged])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, dof, _ = chi2_contingency(table)
    return chi2, p


def part1_chi_square(race_map, demo_map, view_map):
    rows, prop_rows = [], []
    for pathology in PATHOLOGIES:
        for ae in AUTOENCODERS:
            df = load_test_csv(pathology, ae, race_map, demo_map, view_map)
            n_total = len(df)
            for thr_name in ["1pct", "5pct"]:
                thr = THRESHOLDS[thr_name]
                cutoff = df["ppw"].quantile(thr)  # ppw worst = low, per METRIC_DIRECTION
                flagged = df[df["ppw"] <= cutoff]
                n_flagged = len(flagged)
                for dim in ["race_bucket", "gender", "age_band"]:
                    full_counts = df[dim].value_counts().to_dict()
                    flagged_counts = flagged[dim].value_counts().to_dict()
                    categories = sorted(set(full_counts) | set(flagged_counts))
                    chi2, pval = chi_square_vs_full(full_counts, flagged_counts, categories)
                    for cat in categories:
                        full_n, flagged_n = full_counts.get(cat, 0), flagged_counts.get(cat, 0)
                        prop_rows.append(dict(
                            pathology=pathology, autoencoder=ae, threshold=thr_name, dimension=dim, category=cat,
                            full_test_set_pct=round(100 * full_n / n_total, 2) if n_total else np.nan,
                            flagged_subset_pct=round(100 * flagged_n / n_flagged, 2) if n_flagged else np.nan,
                            full_test_set_n=full_n, flagged_subset_n=flagged_n,
                        ))
                    rows.append(dict(pathology=pathology, autoencoder=ae, threshold=thr_name, dimension=dim,
                                      n_total=n_total, n_flagged=n_flagged, chi2=chi2, p_value=pval))

    chi2_df = pd.DataFrame(rows).dropna(subset=["p_value"])
    rej, qval, _, _ = multipletests(chi2_df["p_value"], method="fdr_bh")
    chi2_df["q_value"] = qval
    chi2_df["significant_bh"] = rej
    prop_df = pd.DataFrame(prop_rows)

    chi2_df.to_csv(OUTPUT_DIR / "part1_chi2_tests_BH.csv", index=False)
    prop_df.to_csv(OUTPUT_DIR / "part1_proportions.csv", index=False)
    return chi2_df, prop_df


# ---------------------------------------------------------------------------
# Part 2: does the sex effect generalize across all 4 metrics / 3 thresholds,
# or is it PPW-specific? Logistic regression controlling for true_label.
# ---------------------------------------------------------------------------

def part2_all_metrics_thresholds(race_map, demo_map, view_map):
    results = []
    for pathology in PATHOLOGIES:
        for ae in AUTOENCODERS:
            df = load_test_csv(pathology, ae, race_map, demo_map, view_map)
            df = df[df["gender"].isin(["M", "F"])].copy()
            df["is_male"] = (df["gender"] == "M").astype(int)
            for metric, direction in METRIC_DIRECTION.items():
                for thr_name, thr in THRESHOLDS.items():
                    if direction == "low":
                        flagged = (df[metric] <= df[metric].quantile(thr)).astype(int)
                    else:
                        flagged = (df[metric] >= df[metric].quantile(1 - thr)).astype(int)
                    X = sm.add_constant(df[["is_male", "true_label"]])
                    try:
                        model = sm.Logit(flagged, X).fit(disp=0)
                        coef, pval = model.params["is_male"], model.pvalues["is_male"]
                    except Exception:
                        coef, pval = np.nan, np.nan
                    results.append(dict(pathology=pathology, autoencoder=ae, metric=metric, threshold=thr_name,
                                         OR=np.exp(coef) if pd.notna(coef) else np.nan, p_value=pval))
    res = pd.DataFrame(results)
    res.to_csv(OUTPUT_DIR / "part2_sex_all_metrics_thresholds.csv", index=False)
    return res


# ---------------------------------------------------------------------------
# Part 3: is the male-skew explained by case-mix (the positive class being
# more male, combined with the flagged tail being enriched in positives)?
# ---------------------------------------------------------------------------

def part3_case_mix_decomposition(race_map, demo_map, view_map):
    rows = []
    for pathology in PATHOLOGIES:
        for ae in AUTOENCODERS:
            df = load_test_csv(pathology, ae, race_map, demo_map, view_map)
            df = df[df["gender"].isin(["M", "F"])].copy()
            pos, neg = df[df["true_label"] == 1], df[df["true_label"] == 0]
            pos_m, neg_m = (pos["gender"] == "M").mean(), (neg["gender"] == "M").mean()

            flagged = df[df["ppw"] <= df["ppw"].quantile(0.05)]
            obs_m = (flagged["gender"] == "M").mean() * 100
            flagged_pos_rate = (flagged["true_label"] == 1).mean()
            expected_m = (flagged_pos_rate * pos_m + (1 - flagged_pos_rate) * neg_m) * 100

            rows.append(dict(pathology=pathology, autoencoder=ae, observed_pct_male_flagged=round(obs_m, 1),
                              expected_pct_male_from_case_mix=round(expected_m, 1),
                              unexplained_residual=round(obs_m - expected_m, 1)))
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "part3_case_mix_decomposition.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Part 4: patient-stratified bootstrap CI for the PPW sex OR (5pct, TinyAE)
# ---------------------------------------------------------------------------

def part4_bootstrap(race_map, demo_map, view_map, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for pathology in PATHOLOGIES:
        df = load_test_csv(pathology, "tiny", race_map, demo_map, view_map)
        df = df[df["gender"].isin(["M", "F"])].reset_index(drop=True)
        df["flagged"] = df["ppw"] <= df["ppw"].quantile(0.05)
        subjects = df["subject_id"].unique()

        ors = []
        for _ in range(n_boot):
            sampled_subj = rng.choice(subjects, size=len(subjects), replace=True)
            boot = df[df["subject_id"].isin(sampled_subj)]
            m_rate, f_rate = boot[boot.gender == "M"]["flagged"].mean(), boot[boot.gender == "F"]["flagged"].mean()
            if 0 < f_rate < 1 and 0 < m_rate < 1:
                ors.append((m_rate / (1 - m_rate)) / (f_rate / (1 - f_rate)))
        ors = np.array(ors)
        lo, hi = np.percentile(ors, [2.5, 97.5])
        rows.append(dict(pathology=pathology, OR_median=round(np.median(ors), 2),
                          ci_lo=round(lo, 2), ci_hi=round(hi, 2), excludes_1=bool(lo > 1 or hi < 1)))
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "part4_bootstrap_ppw_sex_OR.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Part 5: does the sex effect survive controlling for AP/PA acquisition view?
# ---------------------------------------------------------------------------

def part5_view_position_control(race_map, demo_map, view_map):
    rows = []
    for pathology in PATHOLOGIES:
        df = load_test_csv(pathology, "tiny", race_map, demo_map, view_map)
        df = df[df["gender"].isin(["M", "F"]) & df["ViewPosition"].isin(["AP", "PA"])].copy()
        df["is_male"] = (df["gender"] == "M").astype(int)
        df["is_ap"] = (df["ViewPosition"] == "AP").astype(int)
        df["flagged"] = (df["ppw"] <= df["ppw"].quantile(0.05)).astype(int)

        X = sm.add_constant(df[["is_male", "true_label", "is_ap"]])
        model = sm.Logit(df["flagged"], X).fit(disp=0)
        rows.append(dict(pathology=pathology,
                          is_male_OR=round(np.exp(model.params["is_male"]), 2), is_male_p=model.pvalues["is_male"],
                          is_ap_OR=round(np.exp(model.params["is_ap"]), 2), is_ap_p=model.pvalues["is_ap"]))
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "part5_view_position_control.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# Part 6: is the effect present within every age band (not driven by one)?
# ---------------------------------------------------------------------------

def part6_age_stratified(race_map, demo_map, view_map):
    rows = []
    for pathology in PATHOLOGIES:
        df = load_test_csv(pathology, "tiny", race_map, demo_map, view_map)
        df = df[df["gender"].isin(["M", "F"])].copy()
        df["flagged"] = (df["ppw"] <= df["ppw"].quantile(0.05)).astype(int)
        df["is_male"] = (df["gender"] == "M").astype(int)

        for band in ["<40", "40-59", "60-79", "80+"]:
            sub = df[df["age_band"] == band]
            if sub["is_male"].nunique() < 2 or sub["flagged"].sum() < 3:
                rows.append(dict(pathology=pathology, age_band=band, n=len(sub), OR=np.nan, p_value=np.nan, note="too few flagged cases"))
                continue
            X = sm.add_constant(sub[["is_male"]])
            model = sm.Logit(sub["flagged"], X).fit(disp=0)
            rows.append(dict(pathology=pathology, age_band=band, n=len(sub),
                              OR=round(np.exp(model.params["is_male"]), 2), p_value=model.pvalues["is_male"], note=""))
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "part6_age_stratified.csv", index=False)
    return out


def main():
    print("Building demographic maps...")
    race_map = build_race_map()
    demo_map = build_demo_map()
    view_map = build_view_map()

    print("Part 1: chi-square disparity tests (race/sex/age, all pathologies x architectures x thresholds)...")
    chi2_df, _ = part1_chi_square(race_map, demo_map, view_map)
    print(f"  {chi2_df['significant_bh'].sum()}/{len(chi2_df)} tests significant after BH correction\n")

    print("Part 2: does the sex effect generalize across all metrics/thresholds?...")
    part2_all_metrics_thresholds(race_map, demo_map, view_map)

    print("Part 3: case-mix decomposition...")
    part3_case_mix_decomposition(race_map, demo_map, view_map)

    print("Part 4: patient-stratified bootstrap CI for the PPW sex OR...")
    part4_bootstrap(race_map, demo_map, view_map)

    print("Part 5: controlling for AP/PA acquisition view...")
    part5_view_position_control(race_map, demo_map, view_map)

    print("Part 6: age-stratified consistency check...")
    part6_age_stratified(race_map, demo_map, view_map)

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
