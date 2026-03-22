# Physics Attribute-Regularized VAE (PAR-VAE)
## Why Early COVID-19 Resists Automated Detection: A Radiological Audit of Physics-Based Feature Discriminability

![Model](https://img.shields.io/badge/Model-PAR--VAE-blue)
![Dataset](https://img.shields.io/badge/Dataset-MosMedData%20%2B%20COVID--CT--MD-orange)
![CT3 AUC](https://img.shields.io/badge/CT--3%20AUC-0.999±0.000-brightgreen)
![CT2 AUC](https://img.shields.io/badge/CT--2%20AUC-0.746±0.008-green)
![CT1 AUC](https://img.shields.io/badge/CT--1%20AUC-0.674±0.011-yellowgreen)
![R2](https://img.shields.io/badge/Physics%20R²-0.972±0.000-purple)
![Features](https://img.shields.io/badge/Physics%20Features-14-blue)
![Seeds](https://img.shields.io/badge/Reproducibility-3%20Seeds-lightgrey)
![Transfer](https://img.shields.io/badge/Transfer%20AUC-0.745-orange)
![Conference](https://img.shields.io/badge/MIUA-2026-red)

---

## Core Question

Why do interpretable AI systems plateau in performance while black-box CNNs achieve near-perfect accuracy on COVID-19 CT classification? Is this gap a modeling limitation — or a fundamental property of the disease itself?

**Our answer: The ceiling is biological, not computational.**

---

## Key Findings

**Finding 1 — Biological Ceiling**
Global physics features face an irreducible 84% class overlap in mild/moderate COVID-19. Both PAR-VAE and CNN baselines hit the same ~70% AUC ceiling on CT-1, confirming the limit is data-intrinsic not model-intrinsic.

**Finding 2 — Severity Gradient**
CT-1: 67% → CT-2: 75% → CT-3: 99.9% AUC — exactly as GGO biology predicts. As disease burden exceeds 50% lung involvement, whole-lung physics statistics shift substantially enough for near-perfect separation.

**Finding 3 — Physics Learns Genuine Pathology**
At severe disease (CT-3), PAR-VAE achieves AUC=0.999±0.000 while CNN collapses to 0.660±0.073 with 49–67% false negative rate across seeds. Physics-constrained representations scale with severity; CNN spatial shortcuts do not.

**Finding 4 — Patch Ablation Confirms Biological Mechanism**
Finer spatial granularity (3×3 patches) increases class overlap rather than reducing it — biologically explained by GGO dilution over normal tissue even within individual patches in mild disease.

**Finding 5 — Cross-Dataset Transfer**
Frozen MosMed encoder with lightweight predictor adaptation outperforms domain-specific retraining on an independent DICOM cohort (AUC=0.745 vs 0.710, FN 33% vs 49.3%).

---

## Results

### Physics Alignment R² (3 seeds, mean ± std)

| Category | Feature | CT-1 R² | CT-2 R² | CT-3 R² |
|----------|---------|---------|---------|---------|
| Tissue Density | Mean HU | 0.868±0.013 | 0.915±0.009 | 0.975±0.003 |
| Tissue Density | HU Std Dev | 0.836±0.004 | 0.927±0.001 | 0.980±0.002 |
| Tissue Density | HU p10 | 0.382±0.018 | 0.580±0.020 | 0.968±0.006 |
| Tissue Density | HU p25 | 0.792±0.007 | 0.845±0.012 | 0.959±0.002 |
| Tissue Density | HU p50 | 0.723±0.018 | 0.686±0.036 | 0.946±0.002 |
| Tissue Density | HU p75 | 0.828±0.014 | 0.800±0.005 | 0.958±0.007 |
| Tissue Density | HU p90 | 0.761±0.010 | 0.758±0.014 | 0.952±0.010 |
| Lung Geometry | Mask Area | 0.906±0.003 | 0.931±0.007 | 0.984±0.002 |
| Lung Geometry | Mask Fraction | 0.910±0.005 | 0.933±0.005 | 0.983±0.002 |
| Boundary Sharpness | Gradient Mean | 0.805±0.028 | 0.805±0.016 | 0.990±0.001 |
| Boundary Sharpness | Gradient Std | 0.885±0.005 | 0.904±0.004 | 0.973±0.003 |
| Texture | GLCM Contrast | 0.824±0.008 | 0.809±0.007 | 0.975±0.005 |
| Texture | Homogeneity | 0.862±0.020 | 0.882±0.002 | 0.982±0.001 |
| Texture | Entropy | 0.891±0.020 | 0.921±0.006 | 0.984±0.003 |
| | **Mean** | **0.81±0.07** | **0.84±0.07** | **0.972±0.000** |

### Classification Performance (3 seeds, mean ± std)

| Task | Model | Test AUC | Test F1 | FN Rate |
|------|-------|---------|---------|---------|
| CT-1 vs CT-0 (Mild) | PAR-VAE LogReg | 0.674±0.011 | 0.658±0.022 | — |
| CT-1 vs CT-0 (Mild) | PAR-VAE Linear SVM | 0.672±0.007 | 0.664±0.013 | — |
| CT-1 vs CT-0 (Mild) | CNN Baseline | 0.703±0.016 | 0.679±0.024 | — |
| CT-2 vs CT-0 (Moderate) | PAR-VAE LogReg | 0.746±0.008 | 0.693±0.020 | — |
| CT-2 vs CT-0 (Moderate) | PAR-VAE Linear SVM | 0.751±0.016 | 0.701±0.023 | — |
| CT-3 vs CT-0 (Severe) | **PAR-VAE RBF SVM** | **0.999±0.000** | **0.983±0.000** | **1.1%** |
| CT-3 vs CT-0 (Severe) | CNN Baseline | 0.660±0.073 | 0.572±0.107 | 49–67% |

*CT-2 CNN baseline excluded due to unstable training dynamics across seeds.*

### Cross-Dataset Transfer Learning (COVID-CT-MD)

| Setup | AUC | FN Rate |
|-------|-----|---------|
| MosMed CT-3 in-domain | 0.999 | 1.1% |
| Retrained from scratch on COVID-CT-MD | 0.710 | 49.3% |
| Frozen MosMed encoder (zero-shot) | 0.731 | 26.5% |
| **Frozen encoder + fine-tuned predictor** | **0.745** | **33.0%** |

### Patch Ablation

| Feature Level | Mean Overlap | vs Global |
|---------------|-------------|-----------|
| Global (whole lung) | 0.765 | baseline |
| 3×3 Patch (central) | 0.800 | +0.035 worse |
| 3×3 Patch (peripheral) | 0.870 | +0.105 worse |
| Mean patch-level | 0.824 | +0.059 worse |

### Class Overlap Analysis (3 seeds)

| Task | Feature Type | Mean Overlap | Cohen's d |
|------|-------------|-------------|-----------|
| CT-1 vs CT-0 | Physics (14-dim) | 0.845±0.080 | 0.38±0.21 |
| CT-1 vs CT-0 | Learned (top 15) | 0.845±0.070 | 0.41±0.19 |
| CT-2 vs CT-0 | Physics (14-dim) | 0.841±0.060 | 0.42±0.15 |
| CT-2 vs CT-0 | Learned (top 15) | 0.841±0.090 | 0.44±0.18 |

---

## Architecture

```
Input CT Slice (512×512, normalised [-1,1])
        ↓
   Encoder — 5-layer CNN (LeakyReLU + BatchNorm)
        ↓
  85-dimensional Latent Space
        ↓
┌─────────────────────────┐
│  Attribute Predictor    │ → 14 physics attributes
│  (3-layer ResNet MLP)   │   R² = 0.972±0.000
└─────────────────────────┘
        ↓
   Decoder — 5-layer CNN (Tanh output)
```

**Training objective:**
```
L = L_recon + β·L_KL + λ·L_attr
```

**3-Phase Annealing Schedule:**

| Phase | Epochs | β | λ | Purpose |
|-------|--------|---|---|---------|
| Physics-First | 0–20 | 10⁻⁴→2·10⁻⁴ | 1.5 | Prevent posterior collapse |
| Gradual Balance | 20–40 | 2·10⁻⁴→5·10⁻⁴ | 1.5→3.0 | Strengthen physics supervision |
| Fine-Tune | 40–50 | 5·10⁻⁴ | 3.0 | Maximise physics alignment |

Healthy KL ≈ 17 confirmed across all seeds and cohorts (collapse threshold KL < 5).

---

## Methodology

### Datasets
- **MosMedData (Primary):** 1,110 patients, 5-level CT severity stratification, Center for Diagnostics and Telemedicine, Moscow
- **COVID-CT-MD (Transfer):** 169 COVID-19 + 76 Normal patients, multi-institutional DICOM cohort

### Cohorts
- **Cohort A (CT-1 vs CT-0):** Mild COVID vs Normal — GGOs covering <25% lung
- **Cohort B (CT-2 vs CT-0):** Moderate COVID vs Normal — 25–50% involvement
- **Cohort C (CT-3 vs CT-0):** Severe COVID vs Normal — 50–75% involvement

### 14 Physics Features

Grounded in X-ray attenuation physics:
```
HU = 1000 × (μ_tissue − μ_water) / (μ_water − μ_air)
```

| Category | Features |
|----------|---------|
| Densitometric (7) | Mean HU, Std HU, p10, p25, p50, p75, p90 |
| Morphological (2) | Lung mask area, fractional occupancy |
| Gradient (2) | Sobel mean, Sobel std |
| Texture (3) | GLCM contrast, homogeneity, entropy |

### 9-Stage Validation Protocol

| Check | Result |
|-------|--------|
| File integrity | 0 missing files |
| HU range | Mean −614.9 ± 79.1 HU, zero outliers |
| Mask integrity | 0 empty masks |
| Slice sampling | Mean 21.1/patient (CT-1), 29.3 (CT-2) |
| Physics validation | ΔHU ≈ 30 (CT-1), ΔHU ≈ 50 (CT-2) |
| Outlier detection | <4% across all features |
| Image quality | 5.0% flagged at 5th percentile |
| Split balance | Chi-square p=0.521 |
| Severity gradient | Mann-Whitney p<0.0001 all features |

Volume-level splitting prevents patient-level leakage — all slices from one patient confined to one split.

---

## Ablation Studies

### Annealing Schedule

| Strategy | KL | Outcome |
|----------|----|---------|
| No annealing | <5 (collapse) | Degenerate latent space |
| High β early | Severe collapse | No image encoding |
| High λ early | ≈15 | Attribute lock-in |
| **3-phase (ours)** | **≈17** | **Stable physics alignment** |

### Latent Dimensionality

| Dimensions | Val R² | Test R² | Gap | Assessment |
|------------|--------|---------|-----|------------|
| 64 | 0.842 | 0.808 | 0.034 | Undercapacity |
| **85** | **0.969** | **0.972** | **0.003** | **Optimal** |
| 96 | 0.845 | 0.789 | 0.056 | Overfitting |

---

## Repository Structure

```
CT_1_Model/                    # Mild severity (CT-0 vs CT-1)
├── Seed_16/
├── Seed_42/
└── Seed_999/
    ├── classification_results/
    ├── latent_space_results/
    ├── overlap_analysis/
    ├── images/
    └── results.txt

CT_2_Model/                    # Moderate severity (CT-0 vs CT-2)
└── [same structure]

CT_3_Model/                    # Severe severity (CT-0 vs CT-3)
└── [same structure]

Transfer_Learning/             # COVID-CT-MD cross-dataset evaluation
├── preprocessing/             # DICOM → NPY pipeline
├── frozen_encoder/            # Zero-shot transfer results
└── finetuned_predictor/       # Adapted predictor results

Ablations/
├── annealing_schedule/
├── latent_dimensionality/
└── patch_ablation/

Docs/
├── paper_MIUA2026.pdf
└── Physics_based_papers.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

```
# python==3.11.13
numpy==1.26.4
pandas==2.2.3
matplotlib==3.7.2
seaborn==0.12.2
scipy==1.15.3
scikit-image==0.25.2
scikit-learn==1.2.2
torch==2.6.0+cu124
Pillow==11.3.0
opencv-python==4.12.0
nibabel==5.3.2
kagglehub==0.3.13
pydicom==2.4.4
pylibjpeg==2.0.1
pylibjpeg-libjpeg==2.1.0
gdcm==3.0.24
tqdm==4.67.1
SimpleITK==2.4.1
```

---

## Citation

```bibtex
@inproceedings{upadhya2026parvae,
  title     = {Why Early COVID-19 Resists Automated Detection:
               A Radiological Audit of Physics-Based Feature Discriminability},
  author    = {Upadhya, Aarya and Udyavar, Anshull M},
  booktitle = {Medical Image Understanding and Analysis (MIUA)},
  year      = {2026},
  publisher = {Springer},
  note      = {LNCS}
}
```

---

## Acknowledgements

MosMedData provided by the Center for Diagnostics and Telemedicine, Moscow.
COVID-CT-MD provided via Figshare (doi:10.6084/m9.figshare.12991592).
PES University, Bengaluru, India.