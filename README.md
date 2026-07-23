# Reconstruction metrics as a triage signal for chest X-ray classifier errors

Code accompanying:

> Antonaci FG, Tohyama T, Han A, Yamamoto R, Marullo G, Ulrich L, Moos S, Celi LA, Vezzetti E.
> **Beyond accuracy: using reconstruction metrics to identify high-risk predictions in chest x-ray AI classifiers.**
> *PLOS Digital Health* (in revision, manuscript PDIG-D-26-00700).

This repository contains the code used to fine-tune a chest X-ray classifier, train two autoencoder
architectures, compute reconstruction metrics, and evaluate whether those metrics can flag
likely classifier errors (both false negatives and false positives) for prioritized human review.

## Overview

The pipeline has five stages, one folder each, run in order:

```
01_classifier_training/    Fine-tunes a TorchXRayVision DenseNet-121 classifier,
                            one binary task per pathology (Atelectasis, Cardiomegaly,
                            Edema, Pleural Effusion, Pneumothorax).

02_autoencoder_training/   Trains two autoencoder architectures (a small deterministic
                            autoencoder, "TinyAE", and a variational autoencoder, "VAE")
                            to reconstruct chest X-ray images, per-pathology and
                            cross-pathology variants.

03_reconstruction_metrics/ Computes four reconstruction metrics per case from the
                            trained autoencoders: MSE, SSIM, Percent Pixels Within
                            tolerance (PPW), and Normalized Edge Difference (NED).

04_tail_analysis/          Splits each metric's distribution into percentile/SD-based
                            tails and reports the classifier's TN/TP/FP/FN composition
                            per tail, per pathology, per architecture, per metric.

05_figures_cost_benefit/   Turns the per-case CSVs into the paper's cost-benefit and
                            enrichment figures (false-negative capture rate vs.
                            percentage of cases flagged for review) and the baseline
                            classification performance summary (accuracy, sensitivity,
                            specificity, AUC per pathology/architecture).
```

Stages 01-02 require GPU training; 03-05 are analysis/plotting and run on CPU given the
per-case CSV outputs of stage 01-02.

## Data

This study uses the [MIMIC-CXR-JPEG](https://physionet.org/content/mimic-cxr-jpeg/) dataset
(chest radiographs) with the accompanying [MIMIC-CXR CheXpert](https://physionet.org/content/mimic-cxr-jpeg/)
and [Med-PaLM 2 labeler](https://physionet.org/content/medpalm-cxr-labels/) label sets, all
distributed via PhysioNet under a Data Use Agreement (DUA). **No patient data is included in
this repository.** Access to MIMIC-CXR requires completing PhysioNet's credentialing process
and DUA; once obtained, the scripts here expect the CSV/image layout described in each stage's
folder (see the placeholder paths in each script — replace with your local data location).

## Environment

```
pip install -r requirements.txt
```

Tested with Python 3.10+. `requirements.txt` documents known-compatible minimum versions for
the packages actually imported across all five stages — the exact environment used for the
original training run was not preserved (external HPC environment), so this is not a frozen
historical pin. If exact reproducibility down to numerical noise matters, re-pin after a first
successful run in a fresh environment.

## Reproducibility

- Stage 01 and 02 (model training) seed `random`, `numpy`, and `torch` from a single
  `GLOBAL_SEED = 42` constant at the top of each script/notebook.
- Stage 04's tail-direction convention (which tail of each metric's distribution is
  "higher-risk") follows the same `METRIC_DIRECTION` mapping used throughout the paper's
  statistical analysis: MSE and NED are error measures (higher = worse), SSIM and PPW are
  similarity measures (lower = worse). This is made explicit in stage 04's output via the
  `is_worst_reconstruction_tail` column, and in stage 05 via the `tradeoff_df` construction.
- Stage 05 ships as a single script, `cost_benefit_figures.py`, distilled from the
  original exploratory notebook down to the code path that actually produces the
  published figures and tables (the earlier notebook, with intermediate exploratory
  variants, is kept only in the authors' private working copy, not in this deposit).

## Citation

If you use this code, please cite the paper above. A DOI/citation record will be added here
once the deposit is minted on Zenodo.

## License

To be added at deposit time.

## Contact

Francesca Giada Antonaci — fga@mit.edu
