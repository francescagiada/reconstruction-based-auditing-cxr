# Reconstruction metrics as a triage signal for chest X-ray classifier errors

Code accompanying:

> Antonaci FG, Tohyama T, Han A, Yamamoto R, Marullo G, Ulrich L, Moos S, Celi LA, Vezzetti E.
> **Beyond accuracy: using reconstruction metrics to identify high-risk predictions in chest x-ray AI classifiers.**
> *PLOS Digital Health* (in revision, manuscript PDIG-D-26-00700).

This repository contains the code used to fine-tune a chest X-ray classifier, train two autoencoder
architectures, compute reconstruction metrics, and evaluate whether those metrics can flag
likely classifier errors (both false negatives and false positives) for prioritized human review.

## Overview

The pipeline has six stages, one folder each, run in order:

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

05_figures_cost_benefit/   Turns the per-case CSVs into cost-benefit and enrichment
                            figures (false-negative capture rate vs.
                            percentage of cases flagged for review) and the baseline
                            classification performance summary (accuracy, sensitivity,
                            specificity, AUC per pathology/architecture).

06_robustness_and_sensitivity_analyses/
                            Statistical robustness checks on the stage 03-05 results.
                            13 scripts, numbered in run order; see each script's own
                            docstring for what it does and what it depends on.
```

Stages 01-02 require GPU training. Stage 03 runs inference with a trained autoencoder
(GPU recommended, CPU also supported) to compute per-case reconstruction metrics from
images. Stages 04-06 are CPU-only statistical analysis on the resulting per-case CSVs.

## Data

This study uses the [MIMIC-CXR-JPEG](https://physionet.org/content/mimic-cxr-jpeg/) dataset
(chest radiographs) with the accompanying [MIMIC-CXR CheXpert](https://physionet.org/content/mimic-cxr-jpeg/)
and [Med-PaLM 2 labeler](https://physionet.org/content/medpalm-cxr-labels/) label sets, all
distributed via PhysioNet under a Data Use Agreement (DUA). **No patient data is included in
this repository.** Access to MIMIC-CXR requires completing PhysioNet's credentialing process
and DUA; once obtained, the scripts here expect the CSV/image layout described in each stage's
folder (see the placeholder paths in each script, replace with your local data location).

## Environment

```
pip install -r requirements.txt
```

Tested with Python 3.10+. `requirements.txt` lists known-compatible minimum versions, not the
exact environment used for the original training run (run on an external server and not
preserved).

Even with a fixed seed and matching package versions, exact numerical reproducibility can
still be affected by standard sources of GPU non-determinism (cuDNN algorithm selection,
floating-point operation ordering).

## Reproducibility

- Stages 01-02 seed `random`, `numpy`, and `torch` from `GLOBAL_SEED = 42`.
- Stage 05 is a single script distilled from the original exploratory notebook down
  to the code path that produces the output figures; intermediate exploratory
  variants are not included in this deposit.
- Stage 06's bootstrap/permutation procedures use a fixed seed (42) and patient-level,
  not row-level, resampling.

## Citation

If you use this code, please cite the paper above.

## License

This code is released under the [MIT License](LICENSE).
