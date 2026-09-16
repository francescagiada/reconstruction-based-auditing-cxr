"""Label-update cases (post-Med-PaLM-2 revision) that fall inside each
pathology's own test split.

Input:
    your_chexpert_labels_pre_update.csv: subject_id, study_id, one column
        per pathology (1.0/0.0/-1.0/NaN)
    your_chexpert_labels_post_update.csv: same format, after label revision
    your_test_split_membership_<pathology>_{train,val,test}.csv: subject_id,
        study_id, one file per pathology per split
Output:
    output/label_updates_full_list.csv
    output/label_updates_split_summary.csv
"""
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRE_PATH = "your_chexpert_labels_pre_update.csv"
POST_PATH = "your_chexpert_labels_post_update.csv"
SPLIT_DIR = "your_test_split_membership_dir"
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

PATHOLOGY_COLS = ["Atelectasis", "Cardiomegaly", "Edema", "No Finding", "Pleural Effusion", "Pneumothorax"]
SPLIT_FOLDER = {
    "Atelectasis": "atelectasis", "Cardiomegaly": "cardiomegaly", "Edema": "edema",
    "Pleural Effusion": "pleuraleffusion", "Pneumothorax": "pneumothorax",
    # "No Finding" has no dedicated binary classifier/test split in this study.
}


def normalize(v):
    # -1.0 (uncertain) and NaN both read as "no positive label"; treating them
    # as the same normalized state means the dataset-wide -1.0 -> blank
    # cleanup never registers as a label "change" here.
    if pd.isna(v) or v == -1.0:
        return "none"
    return "positive" if v == 1.0 else "other"


def main():
    pre = pd.read_csv(PRE_PATH, dtype={"subject_id": "Int64", "study_id": "Int64"}).set_index(["subject_id", "study_id"])
    post = pd.read_csv(POST_PATH, dtype={"subject_id": "Int64", "study_id": "Int64"}).set_index(["subject_id", "study_id"])
    pre, post = pre.align(post, join="inner")

    update_rows = []
    for col in PATHOLOGY_COLS:
        old_n, new_n = pre[col].map(normalize), post[col].map(normalize)
        changed = old_n != new_n
        for (subject_id, study_id) in old_n.index[changed]:
            update_rows.append({
                "pathology_col": col, "subject_id": subject_id, "study_id": study_id,
                "old_value": pre.loc[(subject_id, study_id), col],
                "new_value": post.loc[(subject_id, study_id), col],
                "direction": "to_positive" if new_n.loc[(subject_id, study_id)] == "positive" else "to_negative",
            })

    df_updates = pd.DataFrame(update_rows)
    print("Per-pathology update counts:")
    print(df_updates.groupby("pathology_col").size().to_string())
    print(f"Total: {len(df_updates)}")

    membership_rows = []
    for _, r in df_updates.iterrows():
        folder = SPLIT_FOLDER.get(r["pathology_col"])
        r_out = r.to_dict()
        if folder is None:
            r_out["split"] = "no_binary_task"
        else:
            found = "not_in_this_pathology_dataset"
            for split_name in ["test", "train", "val"]:
                path = os.path.join(SPLIT_DIR, folder, f"split_{folder}_{split_name}.csv")
                if not os.path.exists(path):
                    continue
                sdf = pd.read_csv(path, usecols=["subject_id", "study_id"])
                if ((sdf["subject_id"] == r["subject_id"]) & (sdf["study_id"] == r["study_id"])).any():
                    found = split_name
                    break
            r_out["split"] = found
        membership_rows.append(r_out)

    df_membership = pd.DataFrame(membership_rows)
    df_membership.to_csv(os.path.join(OUT_DIR, "label_updates_full_list.csv"), index=False)

    summary = df_membership.groupby(["pathology_col", "split"]).size().unstack(fill_value=0)
    summary.to_csv(os.path.join(OUT_DIR, "label_updates_split_summary.csv"))
    print("\nSplit membership by pathology:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
