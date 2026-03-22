# Physics Attribute-Regularized VAE (PAR-VAE)
## Why Early COVID-19 Resists Automated Detection: A Radiological Audit of Physics-Based Feature Discriminability

![Model](https://img.shields.io/badge/Model-PAR--VAE-blue)
![Dataset](https://img.shields.io/badge/Dataset-MosMedData%20%2B%20COVID--CT--MD-orange)
![CT3 AUC](https://img.shields.io/badge/CT--3%20AUC-0.999-brightgreen)
![CT2 AUC](https://img.shields.io/badge/CT--2%20AUC-0.746-green)
![CT1 AUC](https://img.shields.io/badge/CT--1%20AUC-0.674-yellowgreen)
![R2](https://img.shields.io/badge/Physics%20R²-0.972-purple)
![Features](https://img.shields.io/badge/Physics%20Features-14-blue)
![Seeds](https://img.shields.io/badge/Reproducibility-3%20Seeds-lightgrey)
![Transfer](https://img.shields.io/badge/Transfer%20AUC-0.745-orange)

---

## Core Question

Why do interpretable AI systems plateau in performance while black-box CNNs achieve near-perfect accuracy on COVID-19 CT classification? Is this gap a modeling limitation — or a fundamental property of the disease itself?

---

## What We Found

Using PAR-VAE — a Variational Autoencoder constrained to encode 14 clinically interpretable radiological attributes grounded in X-ray attenuation physics — we demonstrate that **the performance ceiling of interpretable AI in early COVID-19 detection is biological, not computational.**

### Four Key Findings

**1. Biological Ceiling (CT-1/CT-2)**
Global physics features face an irreducible discriminative ceiling in mild/moderate disease — 84% class overlap confirmed across PAR-VAE, CNN, and ResNet baselines. No model exceeds ~75% AUC on early-stage COVID-19 regardless of architecture.

**2. Model-Independent Proof**
CNN baselines hit the same ceiling as PAR-VAE on CT-1/CT-2, ruling out model capacity as the explanation. At CT-2, PAR-VAE outperforms CNN on the test set (AUC=0.746 vs 0.700) — physics constraints act as implicit regularisation.

**3. Severity Gradient**
The same 14 physics features that fail to discriminate mild infection achieve 99.9% AUC on severe disease (CT-3) — exactly as GGO biology predicts. CNN achieves only 74.5% on CT-3, missing 49.3% of severe COVID cases vs PAR-VAE's 1.1%.

**4. Patch Ablation — Biological Mechanism**
Finer spatial granularity (3×3 patches) *increases* class overlap rather than reducing it — confirming the ceiling is biological. GGO signals are diluted even within individual patches in mild disease.

---

## Severity Gradient Results

| Task | Model | Test AUC | Test F1 | FN Rate |
|------|-------|---------|---------|---------|
| CT-1 vs CT-0 (Mild) | PAR-VAE LogReg | 0.674 ± 1.1% | 0.658 ± 2.2% | — |
| CT-1 vs CT-0 (Mild) | CNN Baseline | 0.717 | 0.683 | — |
| CT-2 vs CT-0 (Moderate) | PAR-VAE Linear SVM | **0.751 ± 1.6%** | 0.701 ± 2.3% | — |
| CT-2 vs CT-0 (Moderate) | CNN Baseline | 0.700 | 0.596 | — |
| CT-3 vs CT-0 (Severe) | PAR-VAE RBF SVM | **0.999** | **0.983** | **1.1%** |
| CT-3 vs CT-0 (Severe) | CNN Baseline | 0.745 | 0.634 | 49.3% |

*CT-1/CT-2 results reported as mean ± std over 3 seeds. CT-3 single seed (999).*

---

## Physics Alignment (R²)

| Category | Feature | CT-1 R² | CT-2 R² | CT-3 R² |
|----------|---------|---------|---------|---------|
| Tissue Density | Mean HU | 0.868±0.013 | 0.915±0.009 | 0.978 |
| Tissue Density | HU Std Dev | 0.836±0.004 | 0.927±0.001 | 0.972 |
| Tissue Density | HU p10–p90 | 0.38–0.83 | 0.58–0.93 | 0.94–0.97 |
| Lung Geometry | Mask Area | 0.906±0.003 | 0.931±0.007 | 0.985 |
| Lung Geometry | Mask Fraction | 0.910±0.005 | 0.933±0.005 | 0.984 |
| Boundary Sharpness | Gradient Mean | 0.805±0.028 | 0.805±0.016 | 0.989 |
| Boundary Sharpness | Gradient Std | 0.885±0.005 | 0.904±0.004 | 0.975 |
| Texture | GLCM Contrast | 0.824±0.008 | 0.809±0.007 | 0.973 |
| Texture | Homogeneity | 0.862±0.020 | 0.882±0.002 | 0.982 |
| Texture | Entropy | 0.891±0.020 | 0.921±0.006 | 0.985 |
| **Mean** | | **0.81±0.07** | **0.84±0.07** | **0.969** |

---

## Cross-Dataset Transfer Learning

Evaluated on COVID-CT-MD — an independent multi-institutional DICOM cohort (169 COVID-19, 76 Normal patients).

| Setup | AUC | FN Rate |
|-------|-----|---------|
| MosMed CT-3 in-domain | 0.999 | 1.1% |
| Retrained from scratch on COVID-CT-MD | 0.710 | 49.3% |
| Frozen MosMed encoder (zero-shot) | 0.731 | 26.5% |
| **Frozen encoder + fine-tuned predictor** | **0.745** | **33.0%** |

Lightweight adaptation — fine-tuning only the 14-dimensional attribute predictor — exceeds the domain-specific retrained baseline while reducing missed COVID cases by 16.3%.

---

## Patch Ablation

| Feature Level | Mean Overlap | vs Global | Interpretation |
|---------------|-------------|-----------|----------------|
| Global (whole lung) | 0.765 | baseline | Best discrimination |
| 3×3 Patch (central) | 0.800 | +0.035 worse | Perihilar — least affected |
| 3×3 Patch (peripheral) | 0.870 | +0.105 worse | GGO region but still diluted |
| Mean patch-level | 0.824 | +0.059 worse | Finer = worse, not better |

---

## Architecture

```
Input CT Slice (512×512)
        ↓
   Encoder (5-layer CNN)
        ↓
  85-dim Latent Space
  ├── 14-dim: physics-constrained
  └── 71-dim: residual spatial
        ↓
┌───────────────────────┐
│  Attribute Predictor  │  →  14 physics attributes (R²=0.97)
│  (3-layer MLP)        │
└───────────────────────┘
        ↓
   Decoder (5-layer CNN)
        ↓
  Reconstructed CT Slice
```

**3-Phase Annealing Schedule:**

| Phase | Epochs | β | λ | Purpose |
|-------|--------|---|---|---------|
| Physics-First | 0–20 | 10⁻⁴→2·10⁻⁴ | 1.5 | Prevent collapse, organise physics |
| Gradual Balance | 20–40 | 2·10⁻⁴→5·10⁻⁴ | 1.5→3.0 | Strengthen physics supervision |
| Fine-Tune | 40–50 | 5·10⁻⁴ | 3.0 | Maximise physics alignment |

---

## Methodology

### Datasets
- **MosMedData (Primary):** 1,110 patients, 5-level CT severity stratification, Moscow municipal hospitals
- **COVID-CT-MD (Transfer):** 169 COVID-19 + 76 Normal patients, multi-institutional DICOM

### Cohorts
- **Cohort A (CT-1 vs CT-0):** Mild COVID vs Normal — the hard clinical case
- **Cohort B (CT-2 vs CT-0):** Moderate COVID vs Normal — severity gradient test
- **Cohort C (CT-3 vs CT-0):** Severe COVID vs Normal — upper bound

### 9-Stage Validation Protocol
1. File integrity — 0 missing files
2. HU range verification — mean -614.9 ± 79.1 HU
3. Mask integrity — 0 empty masks
4. Slice sampling consistency — Middle-30 strategy
5. Physics feature validation — ∆HU ≈ 30 (CT-1), ∆HU ≈ 50 (CT-2)
6. Outlier detection (IQR) — <4% across all features
7. Image quality audit — 5% low-quality threshold
8. Split balance — Chi-square p=0.521
9. Severity gradient verification — Mann-Whitney p<0.0001

### Volume-Level Splitting
All slices from a single patient volume are confined to a single split — preventing anatomy-based data leakage confirmed by Chi-square test (p=0.521).

---

## Repository Structure

```
CT_1_Model/                  # Mild severity (CT-0 vs CT-1)
├── Seed_16/
├── Seed_42/
└── Seed_999/
    ├── classification_results/   # SVM, LogReg, CNN baselines
    ├── latent_space_results/     # 85-dim vs 14-dim analysis
    ├── overlap_analysis/         # Cohen's d, overlap coefficient
    ├── images/                   # Latent space, correlation, violin plots
    └── results.txt

CT_2_Model/                  # Moderate severity (CT-0 vs CT-2)
└── [same structure]

CT_3_Model/                  # Severe severity (CT-0 vs CT-3)
└── [same structure]

Transfer_Learning/           # COVID-CT-MD cross-dataset evaluation
├── preprocessing/           # DICOM → NPY pipeline
├── frozen_encoder/          # Zero-shot transfer results
└── finetuned_predictor/     # Adapted predictor results

Ablations/
├── annealing_schedule/      # 3-phase vs alternatives
├── latent_dimensionality/   # 64 vs 85 vs 96 dim
└── patch_ablation/          # 3×3 local vs global features

Docs/
├── paper_MIUA2026.pdf       # Conference paper (submitted)
└── Physics_based_papers.txt # Literature references
```

---

## Key Contributions

1. **Biological ceiling quantification** — 84% class overlap in mild COVID-19 confirmed model-independently across PAR-VAE, CNN, and ResNet baselines

2. **Severity gradient** — CT-1: 67% → CT-2: 75% → CT-3: 99% AUC, following GGO biology exactly

3. **Patch ablation mechanism** — Counterintuitive finding that finer spatial granularity increases class overlap, biologically explained by GGO dilution

4. **Cross-dataset transfer** — Frozen encoder + lightweight predictor adaptation outperforms domain-specific retraining (AUC=0.745 vs 0.710) on independent DICOM cohort

5. **3-phase annealing curriculum** — Novel training strategy preventing posterior collapse while enforcing physics alignment (KL≈17, R²=0.97)

---

## Clinical Implications

| Approach | CT-1 AUC | CT-2 AUC | CT-3 AUC | Interpretable |
|----------|---------|---------|---------|---------------|
| PAR-VAE (physics-constrained) | 0.674 | 0.751 | **0.999** | Yes — R²=0.97 |
| CNN (black-box) | 0.717 | 0.700 | 0.745 | No |
| ResNet-18 (black-box) | ~0.70 | ~0.72 | ~0.95 | No |

PAR-VAE matches or outperforms black-box baselines on test sets while providing full physics interpretability. The 1.1% false negative rate on severe COVID vs CNN's 49.3% demonstrates that physics-constrained representations learn genuine pathology rather than spurious correlations.

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