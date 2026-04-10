# Quick Start Guide

Get PAR-VAE running in 15 minutes.

## Prerequisites

- Installation complete (see [INSTALLATION.md](INSTALLATION.md))
- Dataset downloaded (see [DATASET.md](DATASET.md))
- Virtual environment activated

## Step 1: Setup Project Directories

```bash
# Create necessary directories
mkdir -p data/{mosmeddata,covid_ct_md}
mkdir -p pretrained_models
mkdir -p experiments/{mosmeddata,covid_ct_md}
```

## Step 2: Prepare Data

Place your dataset CSV files in the appropriate directories:

```
data/
├── mosmeddata/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── covid_ct_md/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Each CSV should contain columns: `[patient_id, ct_slice_path, label, severity]`

## Step 3: Run Data Integrity Checks

Validate your dataset before training:

```bash
python scripts/data_integrity_checks.py \
  --data-dir data/mosmeddata \
  --output-dir experiments/data_validation \
  --n-workers 8
```

Expected output:
```
✓ File integrity: 0 missing files
✓ HU range verification: Mean -614.9 ± 79.1 HU
✓ Mask integrity: 0 non-diagnostic slices
✓ Physics features valid: All 14 features computed
```

## Step 4: Extract Physics Features

Pre-compute physics features for faster training:

```bash
python scripts/extract_features.py \
  --config configs/parvae_default.yaml \
  --data-dir data/mosmeddata \
  --output-dir experiments/feature_extraction \
  --batch-size 32 \
  --n-workers 8
```

Output: `features_extracted.hdf5` with all 14 physics features

## Step 5: Train PAR-VAE

### Train on S3 (Severe) Task - Recommended First

```bash
python scripts/train_parvae.py \
  --config configs/parvae_s3.yaml \
  --seed 42 \
  --data-dir data/mosmeddata \
  --output-dir experiments/mosmeddata/s3_analysis \
  --device cuda:0
```

Monitor training:
```bash
tensorboard --logdir experiments/mosmeddata/s3_analysis/logs
```

**Training time:** ~2 hours on single GPU (V100/A100)

### Model Checkpoints

Training saves checkpoints every 5 epochs:
```
experiments/mosmeddata/s3_analysis/checkpoints/
├── epoch_05.pth
├── epoch_10.pth
├── ...
└── best_model.pth  # Best validation model
```

Resume from checkpoint:
```bash
python scripts/train_parvae.py \
  --config configs/parvae_s3.yaml \
  --resume experiments/mosmeddata/s3_analysis/checkpoints/epoch_30.pth
```

## Step 6: Evaluate Model

### Extract Learned Latent Features

```bash
python scripts/evaluate_model.py \
  --model-path experiments/mosmeddata/s3_analysis/checkpoints/best_model.pth \
  --config configs/parvae_s3.yaml \
  --data-dir data/mosmeddata \
  --output-dir experiments/mosmeddata/s3_analysis/eval \
  --device cuda:0
```

Output:
```
experiments/mosmeddata/s3_analysis/eval/
├── test_latents.npy        # Latent representations (N × 85)
├── test_physics_attrs.npy  # Physics features (N × 14)
├── test_labels.npy         # Binary labels
└── metrics.json            # Classification metrics
```

### Train Classifiers (LogReg/SVM)

```bash
python scripts/train_classifiers.py \
  --features experiments/mosmeddata/s3_analysis/eval/test_latents.npy \
  --labels experiments/mosmeddata/s3_analysis/eval/test_labels.npy \
  --output-dir experiments/mosmeddata/s3_analysis/classifiers
```

Output table:
| Model | Accuracy | AUC | F1 |
|-------|----------|-----|----| 
| LogReg | 97.3% | 99.3% | 96.7% |
| RBF-SVM | 98.5% | 99.3% | 97.1% |

## Step 7: Validate Physics Alignment

Check if physics features are captured:

```bash
python scripts/physics_validation.py \
  --model-path experiments/mosmeddata/s3_analysis/checkpoints/best_model.pth \
  --data-dir data/mosmeddata \
  --output-dir experiments/mosmeddata/s3_analysis/physics
```

Expected R² values by severity:
- **S1 (Mild):** ~0.798
- **S2 (Moderate):** ~0.833  
- **S3 (Severe):** ~0.972

## Step 8: Cross-Scanner Transfer Evaluation

Transfer your trained model to COVID-CT-MD (different scanner):

```bash
python scripts/covid_ct_md_evaluation.py \
  --model-path experiments/mosmeddata/s3_analysis/checkpoints/best_model.pth \
  --config configs/parvae_s3.yaml \
  --data-dir data/covid_ct_md \
  --output-dir experiments/covid_ct_md/transfer_s3 \
  --device cuda:0
```

Expected results:
- In-domain (MosMedData): R² = 0.972, AUC = 99.9%
- Transfer (COVID-CT-MD): R² = 0.320, AUC = 71.0%
- Difference: ΔHU = 482 units (scanner calibration gap)

## Complete Workflow

Train all severity levels (S1, S2, S3):

```bash
for severity in s1 s2 s3; do
  echo "Training S${severity^^}..."
  python scripts/train_parvae.py \
    --config configs/parvae_${severity}.yaml \
    --seed 42 \
    --output-dir experiments/mosmeddata/${severity}_analysis
done
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size in config file
# Or run with:
python scripts/train_parvae.py --batch-size 16 ...
```

### Slow Training

```bash
# Increase workers for data loading
python scripts/train_parvae.py --n-workers 8 ...

# Or enable mixed precision
python scripts/train_parvae.py --mixed-precision ...
```

### No GPU Detected

```bash
python -c "import torch; print(torch.cuda.is_available())"
# If False, check INSTALLATION.md GPU section
```

## Next Steps

- **Multi-seed reproduction:** See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
- **Ablation studies:** See [configs/ablations/](../configs/ablations/)
- **API documentation:** See [API.md](API.md)
- **Dataset details:** See [DATASET.md](DATASET.md)
