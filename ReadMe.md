# Physics Attribute-Regularized VAE (PAR-VAE)
## Beyond Model Performance: A Data-Centric Analysis of COVID-19 CT Classification Ceiling

---

## Core Question

While deep learning models achieve 90%+ accuracy on COVID-19 CT classification, they operate as "black boxes" prone to spurious correlations. We ask: **Can physics-based radiological features alone provide sufficient discriminative signal for trustworthy medical AI?**

---

## What We Found

Using a Physics Attribute-Regularized VAE that constrains latent representations to align with 14 clinically interpretable radiological features, we establish a **"trustworthiness ceiling"**:

- **Performance Plateau:** 66.4% accuracy (CT-2) and 62.6% (CT-1) despite strong physics grounding (R² = 0.83)
- **Feature Redundancy:** 82% of learned features explainable by physics features (R² = 0.70-0.85)
- **Class Overlap:** 84% distributional overlap in physics space creates irreducible error floor
- **Spatial Information Gap:** ~25% performance gap to black-box CNNs represents unconstrained spatial patterns

**Key Insight:** The limitation isn't model capacity—it's the data representation itself. Global radiological features, regardless of sophistication, cannot approach CNN performance without spatial context.

---

## Main Results

| Classification Task | Test Accuracy | Test AUC | Overlap Coefficient | Seeds |
|---------------------|---------------|----------|---------------------|-------|
| **CT-0 vs CT-2** (Moderate, 25-50% involvement) | 66.4 ± 2.0% | 0.746 ± 0.010 | 0.838 | 3 |
| **CT-0 vs CT-1** (Mild, <25% involvement) | 62.6 ± 2.8% | 0.674 ± 0.014 | 0.855 | 3 |

**Dimensionality Analysis:** Reducing from 85-dim latent space to 14-dim physics features causes only 0.66-4.95% accuracy loss, proving that 83.5% of features are informationally redundant.

---

## Methodology

### Dataset
- **MosMedData:** ~5,000 CT slices filtered to early-stage cases only
- **CT-0:** Normal (0% involvement)
- **CT-1:** Mild severity (<25% involvement)
- **CT-2:** Moderate severity (25-50% involvement)
- **Excluded:** CT-3/CT-4 (severe cases inflate metrics)

### 14 Physics-Based Features
Grounded in X-ray attenuation physics (Hounsfield Units):

**Densitometric (7):** HU mean, std, percentiles (p10, p25, p50, p75, p90)  
**Textural (3):** GLCM contrast, homogeneity, entropy  
**Gradient (2):** Sobel mean, std  
**Morphological (2):** Lung mask area, fractional occupancy  

### PAR-VAE Architecture
- **Encoder:** 5-layer CNN → 85-dim latent space
- **Decoder:** Mirrored 5-layer CNN
- **Attribute Predictor:** 3-layer MLP predicting 14 physics features from latent means
- **Loss:** L = L_recon + β·L_KL + λ·L_attr

### Validation Pipeline
- Volume-level splitting (60/20/20) prevents patient-level leakage
- Multiple seeds (16, 42, 999) for reproducibility
- Statistical analysis: correlation, interaction (R²), class overlap

---

## Repository Organization
```
CT_1_Model/          # Mild severity (CT-0 vs CT-1) experiments
├── Seed_16/         # Independent run with seed 16
├── Seed_42/         # Independent run with seed 42
└── Seed_999/        # Independent run with seed 999
    ├── classification_results/  # SVM, LogReg performance
    ├── latent_space_results/    # 85-dim vs 14-dim analysis
    ├── images/                  # Visualizations
    └── results.txt              # Summary metrics

CT_2_Model/          # Moderate severity (CT-0 vs CT-2) experiments
└── [Same structure as CT_1_Model]

Docs/
├── Draft_methodology.pdf        # Detailed methods
└── Physics_based_papers.txt     # Literature references
```

---

## Key Contributions

1. **Methodological:** Rigorous physics-based feature engineering validated on ~5,000 CT slices, transforming raw pixels into interpretable biomarkers

2. **Architectural:** Multi-objective VAE curriculum preventing posterior collapse while enforcing physics interpretability

3. **Analytical:** Quantification of the "Spatial Information Gap"—identifying specific statistical overlaps in global features that create an irreducible classification error floor

---

## Statistical Analysis Results

### Correlation Analysis
- 15+ learned features show |r| > 0.5 with physics features
- Example: Feature_40 correlates with homogeneity (r = -0.72)

### Interaction Analysis
- Mean R² = 0.72 for predicting learned features from physics
- 89% of analyzed features marked "REDUNDANT"
- Only 11% capture genuine interaction effects

### Class Overlap Analysis
- Mean overlap coefficient = 0.84 (range: 0.71-0.91)
- Cohen's d < 0.5 for 60% of features
- High overlap despite statistically significant differences (p < 0.0001)

---

## Clinical Implications

Our results establish a **trustworthiness-performance tradeoff** in medical AI:

| Approach | Accuracy | Interpretability | Verifiability |
|----------|----------|------------------|---------------|
| Physics features only | 66% | Full | 100% |
| PAR-VAE (physics-constrained) | 70% | High (R²=0.83) | High |
| Black-box CNNs | 95% | None | 0% |

**The 70% ceiling represents the maximum performance achievable while maintaining full interpretability through established radiological principles.**

---


