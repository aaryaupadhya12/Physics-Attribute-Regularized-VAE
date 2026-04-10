# Dataset Documentation

Complete guide to datasets used in PAR-VAE.

## Overview

| Dataset | Patients | Slices | Scanner Type | Purpose |
|---------|----------|--------|--------------|---------|
| **MosMedData** | 1,110 | ~26,000 | Siemens SOMATOM | Primary training & validation |
| **COVID-CT-MD** | N/A | ~2,000 | Multi-institutional | Cross-scanner transfer evaluation |

## MosMedData (Primary Dataset)

### Source
- **Institution:** Centre for Diagnostics and Telemedicine (CDT), Moscow
- **Type:** Non-commercial research license
- **Access:** https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1
- **Citation:** Goncharov et al., 2020

### Cohort Composition

| Severity Level | Category | GGO % | Patients | Slices (5-per-patient) | Label |
|---|---|---|---|---|---|
| Normal | CT-0 | 0% | 370 | 1,850 | S0 |
| Mild | CT-1 | <25% | 440 | 5,500 balanced | S1 |
| Moderate | CT-2 | 25-50% | 170 | 1,700 balanced | S2 |
| Severe | CT-3 | 50-75% | 130 | 1,760 balanced | S3 |
| **Total** | | | **1,110** | **~26,000** | |

### Data Split Strategy

**Volume-level stratified split (70/15/15):**
- All slices from one patient confined to single split
- Stratified by severity to maintain label distribution
- Chi-square balance test: p = 0.521 (balanced)

```
Training Set:   770 patients (70%)
Validation Set: 170 patients (15%)  
Test Set:       170 patients (15%)
```

### File Format & Naming Convention

**Directory structure:**
```
mosmeddata/
├── CT_000/
│   ├── 01_initial_image.nii.gz      # Original CT volume
│   ├── 01_segmentation.nii.gz       # Lung mask  
│   └── 01_data1.nii.gz             # Preprocessed volume
├── CT_001/
│   ├── 02_initial_image.nii.gz
│   └── ...
└── train.csv  # CSV with splits

train.csv format:
patient_id,ct_scan_path,lung_mask_path,label,fold,severity
CT_000,CT_000/01_data1.nii.gz,CT_000/01_segmentation.nii.gz,0,train,S0
CT_001,CT_001/02_data1.nii.gz,CT_001/02_segmentation.nii.gz,1,train,S1
...
```

### Image Specifications

| Property | Value |
|----------|-------|
| **Format** | NIfTI (.nii.gz) |
| **Modality** | CT (Hounsfield Units) |
| **Slice Thickness** | 1-5mm (variable) |
| **Typical Dimensions** | 512 × 512 × 70-150 slices |
| **Mean HU (All)** | -614.9 ± 79.1 |
| **HU Range** | -1000 to 400 |
| **Preprocessing** | Filtered back-projection (FBP) |

### HU Statistics by Severity

| Severity | Mean HU | Std HU | Min HU | Max HU |
|----------|---------|--------|--------|--------|
| S0 (Normal) | -651 | 54 | -850 | 150 |
| S1 (Mild) | -625 | 62 | -800 | 200 |
| S2 (Moderate) | -585 | 68 | -750 | 250 |
| S3 (Severe) | -540 | 71 | -720 | 300 |

## COVID-CT-MD (Transfer Dataset)

### Source
- **Licenses:** Open access for research
- **Type:** Multi-institutional multi-scanner cohort
- **Purpose:** Cross-scanner domain shift evaluation
- **Access:** https://github.com/ShahinSHH/COVID-CT-MD

### Characteristics

| Property | Value |
|----------|-------|
| **Patients** | 169 |
| **Slices** | ~2,000 |
| **Scanners** | Siemens, GE, Philips (mixed) |
| **Mean HU** | -132.9 ± 156.8 |
| **ΔHU vs MosMedData** | **482 units** |
| **Task** | Binary classification (COVID vs Normal) |

### Known Challenges

1. **Scanner calibration:** Different manufacturers use different HU scales
2. **Reconstruction kernel:** FBP vs iterative reconstruction
3. **Protocol variations:** kVp, mAs differ across institutions
4. **HU shift:** ~482 units from MosMedData baseline

**Expected performance drop:**
- In-domain (MosMedData): R² = 0.972
- Transfer (COVID-CT-MD): R² = 0.320
- This quantifiable gap indicates domain shift severity

## Usage: Downloading and Preparing Data

### 1. Download MosMedData

Use the official access request:

```bash
# Follow instructions at:
# https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1

# Once downloaded, organize as:
mkdir -p data/mosmeddata
unzip mosmeddata_release_full.zip -d data/mosmeddata
```

### 2. Download COVID-CT-MD

```bash
# Clone the repository
git clone https://github.com/ShahinSHH/COVID-CT-MD.git

# Copy data to project structure
cp -r COVID-CT-MD/data/* data/covid_ct_md/
```

### 3. Create CSV Index Files

Auto-generate train/val/test CSVs:

```bash
python scripts/create_dataset_index.py \
  --data-dir data/mosmeddata \
  --output-dir data/mosmeddata \
  --split-ratio 0.7,0.15,0.15 \
  --stratify-by severity
```

**Output:**
```
data/mosmeddata/
├── train.csv
├── val.csv
└── test.csv
```

### 4. Validate Data Integrity

```bash
python scripts/data_integrity_checks.py \
  --data-dir data/mosmeddata \
  --output-dir experiments/validation

python scripts/data_integrity_checks.py \
  --data-dir data/covid_ct_md \
  --output-dir experiments/validation
```

Expected output for valid data:
```
✓ File integrity: 0 missing files
✓ HU range verification: Mean -614.9 ± 79.1 HU, 0 outliers
✓ Mask integrity: 0 non-diagnostic slices
✓ Physics feature validation: All 14 features computed
✓ Split balance: Chi-square p = 0.521
```

## 14 Physics Features

All features extracted per CT slice:

### Tissue Density (7 features)
Computed from HU histogram within lung mask:
- `mean_hu`: Average HU value
- `std_hu`: Standard deviation
- `hu_p10`, `hu_p25`, `hu_p50`, `hu_p75`, `hu_p90`: Percentiles

```python
hu_values = ct_slice[lung_mask > 0]
mean_hu = np.mean(hu_values)
std_hu = np.std(hu_values)
```

### Lung Geometry (2 features)
Computed from segmentation mask:
- `mask_area`: Total lung area in mm²
- `fractional_occupancy`: (GGO area) / (total lung area)

```python
mask_area = np.sum(lung_mask > 0) * pixel_spacing ** 2
fractional_occupancy = np.sum(ggo_mask > 0) / np.sum(lung_mask > 0)
```

### Boundary Sharpness (2 features)
Computed with Sobel gradient operator:
- `gradient_mean`: Average absolute gradient magnitude
- `gradient_std`: Std dev of gradient magnitude

```python
gradient = np.hypot(sobel_x, sobel_y)
gradient_mean = np.mean(gradient[lung_mask > 0])
gradient_std = np.std(gradient[lung_mask > 0])
```

### Texture (3 features)
Gray-Level Co-occurrence Matrix (GLCM) features:
- `glcm_contrast`: Sum of squared differences
- `glcm_homogeneity`: Local consistency
- `glcm_entropy`: Randomness/disorder

```python
from skimage.feature import greycomatrix, greycoprops
glcm = greycomatrix(ct_slice, [1], [0], 256, symmetric=True)
contrast = greycoprops(glcm, 'contrast')[0, 0]
```

## Feature Extraction Code Example

```python
from src.utils import PhysicsFeatureExtractor

extractor = PhysicsFeatureExtractor()

# Extract all 14 features for a slice
ct_slice = np.load('CT_000_slice_20.npy')
lung_mask = np.load('mask_000_slice_20.npy')

features = extractor.extract(
    ct_slice=ct_slice,
    lung_mask=lung_mask,
    pixel_spacing=0.5  # mm/pixel
)

print(features)  # Output: dict with 14 keys
#  {'mean_hu': -620.5, 'std_hu': 58.2, ..., 'glcm_entropy': 6.8}
```

## Data Licensing and Citation

### MosMedData
```bibtex
@article{goncharov2020,
  title={CT-based COVID-19 triage},
  author={Goncharov, M and others},
  journal={medRxiv},
  year={2020}
}
```

### COVID-CT-MD
```bibtex
@article{shahin2020,
  title={COVID-CT-MD: Evaluation of Convolutional Neural Networks...},
  author={Shahin, Amir Hossein and others},
  journal={arXiv},
  year={2020}
}
```

## Data Privacy and Ethics

- **Consent:** All datasets obtained with appropriate institutional approvals
- **De-identification:** Patient identifiers removed
- **Usage:** Research purposes only; cannot be used for clinical deployment without appropriate regulatory approval
- **Sharing:** Follow respective dataset licensing terms

## Troubleshooting Data Loading

### Issue: "File not found" errors

```python
# Verify paths are absolute
cd data/mosmeddata
ls -la CT_*/01_data1.nii.gz | head -5

# Fix CSV paths if relative
python scripts/fix_dataset_paths.py --data-dir data/mosmeddata
```

### Issue: HU values out of expected range

```python
# Check if data is already normalized (should be -1000 to 400)
# If not, may need HU clamping:
ct_clipped = np.clip(ct_slice, -1000, 400)
```

### Issue: Memory errors loading full volumes

Use sliced loading:

```python
import nibabel as nib

img = nib.load('CT_000/01_data1.nii.gz')
# Load slice-by-slice instead of full volume
for i in range(img.shape[2]):
    slice_i = np.array(img.dataobj[:, :, i])
```

## Next Steps

- After data setup, run: [QUICKSTART.md](QUICKSTART.md)
- For detailed training: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- Check feature extraction: [API.md](API.md#PhysicsFeatureExtractor)
