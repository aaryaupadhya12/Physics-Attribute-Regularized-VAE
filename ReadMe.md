# Physics Attribute-Regularized VAE (PAR-VAE)

## A Physics-Constrained Generative Audit of CT Severity Classification (Accepted MIUA 2026)

![Model](https://img.shields.io/badge/Model-PAR--VAE-blue)
![Dataset](https://img.shields.io/badge/Dataset-MosMedData%20%2B%20COVID--CT--MD-orange)
![S3_AUC](https://img.shields.io/badge/S3%20AUC-99.3±1.0%25-brightgreen)
![S2_AUC](https://img.shields.io/badge/S2%20AUC-74.6±0.8%25-green)
![S1_AUC](https://img.shields.io/badge/S1%20AUC-67.4±1.4%25-yellowgreen)
![Physics_R2](https://img.shields.io/badge/Physics%20R²-0.972-purple)
![Features](https://img.shields.io/badge/Physics%20Features-14-blue)
![Reproducibility](https://img.shields.io/badge/Reproducibility-3%20Seeds-lightgrey)

---

## Core Question

When a physics-grounded model underperforms a black-box CNN on COVID-19 CT classification, is the gap a modeling failure — or a fundamental biological property of the disease?

**Our answer: The ceiling might be Biological at Lower severity instances**

---

## Key Findings

**Finding 1 — Biological Ceiling at Mild Disease**
Global physics features face an irreducible 84% class overlap at mild COVID-19 (S1). Both PAR-VAE and CNN baselines converge at ~67–70% AUC on S1, confirming the limit is data-intrinsic, not model-intrinsic.

**Finding 2 — Severity Gradient**
S1: 67% → S2: 75% → S3: 99.3% AUC — exactly as GGO biology predicts. As disease burden exceeds 50% lung involvement, whole-lung physics statistics shift enough for near-perfect separation.

**Finding 3 — CNN Catastrophic Failure at Severe Disease**
At S3, PAR-VAE achieves 99.3 ± 1.0% AUC and misses only 1.6% of cases. The CNN baseline achieves 66.0 ± 7.5% AUC and misses 46.9 ± 19.7% of severe cases across seeds — a 26× variance inflation revealing systematic shortcut learning rather than genuine pathological generalisation.

**Finding 4 — Patch Ablation Confirms the Biological Mechanism**
Finer spatial granularity (3×3 patches) increases class overlap rather than reducing it. This is biologically explained by GGO dilution within normal tissue even at the patch level in mild disease — the ceiling is irreducible at any aggregation scale.

**Finding 5 — Physics Makes Domain Shift Visible**
When evaluated on COVID-CT-MD (different scanner), PAR-VAE's physics alignment R² drops from 0.972 to 0.320 and ΔHU = 482 units — a quantifiable diagnostic signal. The CNN degrades silently with no internal warning.

---

## Methodology

### Datasets

- **MosMedData (Primary):** 1,110 patients, 5-level CT severity stratification, Centre for Diagnostics and Telemedicine, Moscow.
- **COVID-CT-MD (Transfer):** Independent multi-institutional DICOM cohort for cross-scanner evaluation.

### Cohort Construction

| Label | Severity | GGO Involvement | Slices |
|-------|----------|-----------------|--------|
| S0 | Normal (CT-0) | 0% | — |
| S1 | Mild (CT-1) | < 25% | 5,500 balanced |
| S2 | Moderate (CT-2) | 25–50% | 1,700 balanced |
| S3 | Severe (CT-3) | 50–75% | 1,760 balanced |

Volume-level 70/15/15 train/val/test split — all slices from one patient confined to one split. Chi-square split balance confirmed (p = 0.521).

### 14 Physics Features

Grounded in X-ray attenuation physics (HU scale):

| Category | Features |
|----------|----------|
| Tissue Density (7) | Mean HU, Std HU, p10, p25, p50, p75, p90 |
| Lung Geometry (2) | Mask area, fractional occupancy |
| Boundary Sharpness (2) | Sobel gradient mean, Sobel gradient std |
| Texture (3) | GLCM contrast, homogeneity, entropy |

### Architecture

PAR-VAE has three components:

- **Encoder:** 5-layer CNN mapping 512×512 CT slices to an 85-dimensional latent space (Leaky-ReLU, Batch Normalization).
- **Decoder:** Mirrored 5-layer CNN (Tanh output).
- **Physics Attribute Predictor:** 3-layer MLP predicting the 14 radiological attributes from latent means μz.

The 85-dimensional space is implicitly partitioned: 14 dimensions are regularized toward clinical attributes, 71 remain free for residual information not explained by global physics.

### Training Objective

The loss balances reconstruction, latent regularization, and physics alignment:

$$\mathcal{L} = \mathcal{L}_\text{recon} + \beta \cdot \mathcal{L}_\text{KL} + \lambda \cdot \mathcal{L}_\text{attr}$$

### 3-Phase Annealing Schedule

| Phase | Epochs | β | λ | Purpose |
|-------|--------|---|---|---------|
| Physics-First | 0–20 | 10⁻⁴ → 2·10⁻⁴ | 1.5 | Prevent posterior collapse |
| Gradual Balance | 20–40 | 2·10⁻⁴ → 5·10⁻⁴ | 1.5 → 3.0 | Tighten physics supervision |
| Fine-Tune | 40–50 | 5·10⁻⁴ | 3.0 | Maximise physics alignment |

Healthy KL ≈ 15–17 confirmed across all seeds and cohorts. Collapse threshold: KL < 5.

### 9-Stage Data Integrity Protocol

| Check | Result |
|-------|--------|
| File integrity | 0 missing files |
| HU range verification | Mean −614.9 ± 79.1 HU, zero outliers |
| Mask integrity | 0 non-diagnostic slices |
| Slice sampling consistency | 21.1/patient (S1), 29.3/patient (S2) |
| Physics feature validation | ΔHU ≈ 30 (S1 vs S0) |
| Outlier detection (IQR) | < 4% across all 14 features |
| Image quality audit | 5.0% flagged at 5th percentile |
| Split balance | Chi-square p = 0.521 |
| Severity gradient | Mann-Whitney p < 0.0001 all features |

---

## Results

### Physics Alignment R² (mean ± std, 3 seeds)

| Category | Feature | S1 R² | S2 R² | S3 R² |
|----------|---------|--------|--------|--------|
| Tissue Density | Mean HU | 0.864 | 0.910 | 0.976 |
| Tissue Density | HU Std Dev | 0.832 | 0.927 | 0.981 |
| Tissue Density | HU p10 | 0.364 | 0.569 | 0.963 |
| Tissue Density | HU p25 | 0.794 | 0.846 | 0.956 |
| Tissue Density | HU p50 | 0.704 | 0.671 | 0.943 |
| Tissue Density | HU p75 | 0.814 | 0.803 | 0.962 |
| Tissue Density | HU p90 | 0.770 | 0.773 | 0.962 |
| Lung Geometry | Mask Area | 0.902 | 0.922 | 0.981 |
| Lung Geometry | Fractional Occupancy | 0.909 | 0.927 | 0.984 |
| Boundary Sharpness | Gradient Mean | 0.791 | 0.820 | 0.989 |
| Boundary Sharpness | Gradient Std | 0.890 | 0.899 | 0.972 |
| Texture | GLCM Contrast | 0.819 | 0.805 | 0.979 |
| Texture | Homogeneity | 0.843 | 0.880 | 0.982 |
| Texture | Entropy | 0.875 | 0.915 | 0.980 |
| | **Mean** | **0.798** | **0.833** | **0.972** |

### Classification Performance (mean ± std, 3 seeds)

| Task | Model | Val Acc | Test Acc | Test F1 | Test AUC |
|------|-------|---------|----------|---------|----------|
| S1 vs S0 (Mild) | PAR-VAE (LogReg) | 62.0 ± 2.1 | 62.6 ± 2.8 | 65.8 ± 2.6 | 67.4 ± 1.4 |
| S1 vs S0 (Mild) | CNN Baseline | 66.8 ± 1.5 | 65.2 ± 0.9 | 68.2 ± 2.5 | **69.8 ± 1.1** |
| S2 vs S0 (Moderate) | PAR-VAE (LogReg) | 70.8 ± 1.2 | 66.5 ± 1.6 | 69.3 ± 2.0 | **74.6 ± 0.8** |
| S2 vs S0 (Moderate) | CNN Baseline | 68.3 ± 2.4 | 64.1 ± 2.1 | 66.7 ± 1.9 | 70.0 ± 1.5 |
| S3 vs S0 (Severe) | PAR-VAE (RBF-SVM) | 98.5 ± 1.1 | 97.3 ± 2.9 | 96.7 ± 3.8 | **99.3 ± 1.0** |
| S3 vs S0 (Severe) | CNN Baseline | 57.1 ± 6.7 | 60.6 ± 6.3 | 57.2 ± 11.6 | 66.0 ± 7.5 |

CNN leads narrowly on S1 (69.8% vs 67.4%). PAR-VAE leads decisively on S2 and S3. At S3, CNN misses 46.9 ± 19.7% of severe cases vs PAR-VAE's 1.6% — a 26× variance inflation.

### Class Overlap and Separability (mean ± std, 3 seeds)

| Task | Feature Type | Mean Overlap | Cohen's d |
|------|-------------|-------------|-----------|
| S1 vs S0 | Physics (14) | 0.845 ± 0.080 | 0.38 ± 0.21 |
| S1 vs S0 | Learned (top 15) | 0.845 ± 0.070 | 0.41 ± 0.19 |
| S2 vs S0 | Physics (14) | 0.841 ± 0.060 | 0.42 ± 0.15 |
| S2 vs S0 | Learned (top 15) | 0.841 ± 0.090 | 0.44 ± 0.18 |
| S3 vs S0 | Physics (14) | 0.776 ± 0.050 | 0.42 ± 0.15 |
| S3 vs S0 | Learned (top 15) | 0.783 ± 0.014 | 0.37 ± 0.17 |

Near-identical physics and learned overlap values confirm the ceiling is data-intrinsic, not feature-design-dependent.

### Cross-Scanner Transfer (COVID-CT-MD)

| Setup | R² | AUC | FN Rate |
|-------|-----|-----|---------|
| MosMedData S3 in-domain | 0.972 | 0.999 | 1.1% |
| Retrained from scratch on COVID-CT-MD | 0.322 | 0.710 | 49.3% |
| **Frozen encoder + fine-tuned predictor** | **0.417** | **0.745** | **33.0%** |

R² drop from 0.972 → 0.320 quantifies the scanner calibration gap (ΔHU = 482 units). CNN achieves AUC = 0.71 with no equivalent internal signal of degradation.

---

## Ablation Studies

### Annealing Schedule

| Strategy | KL at Epoch 50 | Outcome |
|----------|---------------|---------|
| No annealing | < 5 (collapse) | Degenerate latent space |
| High β early | < 5 (collapse) | No image structure encoded |
| High λ early | ≈ 15 | Attribute lock-in, poor reconstruction |
| **3-phase (ours)** | **≈ 15** | **Stable, generalisable alignment** |

---

## Acknowledgements

MosMedData provided by the Centre for Diagnostics and Telemedicine, Moscow. COVID-CT-MD provided under open access for research use.
