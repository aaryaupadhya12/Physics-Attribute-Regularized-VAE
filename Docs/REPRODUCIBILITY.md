# Reproducibility Guide

Detailed instructions for reproducing all results in the paper.

## Overview

All results use **3 random seeds** (16, 42, 999) with fixed PyTorch seeds for deterministic behavior.

| Task | Seeds | Model | Runs Required |
|------|-------|-------|-----------------|
| S1 Classification | 3 | LogReg | 3 |
| S2 Classification | 3 | LogReg | 3 |
| S3 Classification | 3 | RBF-SVM | 3 |
| Ablation: Annealing | 1 | VAE | 5 strategies |
| Ablation: Latent Dim | 1 | VAE | 3 dimensions |
| Transfer Learning | 1 | VAE (transfer) | 1 |
| **Total Training Time** | | | ~72 GPU-hours |

## Reproducing Table 2 (Main Results)

### Automatic Reproduction (Recommended)

```bash
# Reproduce all results in one script
bash scripts/reproduce_all_results.sh
```

This will:
1. Train PAR-VAE for S1, S2, S3 with seeds 16, 42, 999
2. Extract physics features for all models
3. Train classifiers (LogReg/SVM)
4. Generate result tables (CSV and JSON)
5. Visualize comparisons

**Expected output:**
```
experiments/
├── mosmeddata/
│   ├── s1_analysis/
│   │   ├── seed_16/
│   │   ├── seed_42/
│   │   └── seed_999/
│   ├── s2_analysis/
│   │   ├── seed_16/
│   │   ├── seed_42/
│   │   └── seed_999/
│   └── s3_analysis/
│       ├── seed_16/
│       ├── seed_42/
│       └── seed_999/
└── results_summary.csv   # Final table
```

### Manual Step-by-Step

#### 1. Train PAR-VAE (All Tasks & Seeds)

```bash
# Create output directories
mkdir -p experiments/mosmeddata/{s1,s2,s3}_analysis/seed_{16,42,999}

# S1 (Mild) - All seeds
for seed in 16 42 999; do
  python scripts/train_parvae.py \
    --config configs/parvae_s1.yaml \
    --seed $seed \
    --output-dir experiments/mosmeddata/s1_analysis/seed_${seed}
done

# S2 (Moderate) - All seeds
for seed in 16 42 999; do
  python scripts/train_parvae.py \
    --config configs/parvae_s2.yaml \
    --seed $seed \
    --output-dir experiments/mosmeddata/s2_analysis/seed_${seed}
done

# S3 (Severe) - All seeds
for seed in 16 42 999; do
  python scripts/train_parvae.py \
    --config configs/parvae_s3.yaml \
    --seed $seed \
    --output-dir experiments/mosmeddata/s3_analysis/seed_${seed}
done
```

**Expected runtime:** ~48 hours on single V100 GPU

#### 2. Extract Physics Features

```bash
# For each trained model, extract alignment metrics
for severity in s1 s2 s3; do
  for seed in 16 42 999; do
    python scripts/extract_features.py \
      --model-path experiments/mosmeddata/${severity}_analysis/seed_${seed}/checkpoints/best_model.pth \
      --config configs/parvae_${severity}.yaml \
      --data-dir data/mosmeddata \
      --output-dir experiments/mosmeddata/${severity}_analysis/seed_${seed}/features
  done
done
```

#### 3. Train Classifiers

```bash
# LogReg/SVM on latent features
for severity in s1 s2 s3; do
  for seed in 16 42 999; do
    python scripts/train_classifiers.py \
      --latent-path experiments/mosmeddata/${severity}_analysis/seed_${seed}/eval/test_latents.npy \
      --labels-path experiments/mosmeddata/${severity}_analysis/seed_${seed}/eval/test_labels.npy \
      --output-dir experiments/mosmeddata/${severity}_analysis/seed_${seed}/classifiers
  done
done
```

#### 4. Generate Results Table

```bash
python scripts/aggregate_results.py \
  --results-dir experiments/mosmeddata \
  --output experiments/results_summary.csv
```

**Output: `results_summary.csv`**
```csv
Task,Model,Seed,Val_Acc,Test_Acc,Test_F1,Test_AUC,Physics_R2
S1,PAR-VAE,16,62.1,62.8,65.9,67.1,0.798
S1,PAR-VAE,42,61.9,62.4,65.7,67.5,0.808
S1,PAR-VAE,999,62.0,62.6,65.8,67.4,0.788
...
```

## Reproducing Table 3 (Ablation Studies)

### Annealing Schedule Ablation

```bash
mkdir -p experiments/ablations/annealing

# No annealing (baseline)
python scripts/train_parvae.py \
  --config configs/ablations/annealing_none.yaml \
  --output-dir experiments/ablations/annealing/no_annealing

# High β early
python scripts/train_parvae.py \
  --config configs/ablations/annealing_high_beta_early.yaml \
  --output-dir experiments/ablations/annealing/high_beta_early

# High λ early  
python scripts/train_parvae.py \
  --config configs/ablations/annealing_high_lambda_early.yaml \
  --output-dir experiments/ablations/annealing/high_lambda_early

# Our 3-phase schedule
python scripts/train_parvae.py \
  --config configs/parvae_s3.yaml \
  --output-dir experiments/ablations/annealing/three_phase
```

### Latent Dimensionality Ablation

```bash
mkdir -p experiments/ablations/latent_dim

for dim in 64 85 96; do
  python scripts/train_parvae.py \
    --config configs/ablations/latent_dim_${dim}.yaml \
    --output-dir experiments/ablations/latent_dim/dim_${dim}
done
```

## Reproducing Table 4 (Transfer Learning)

```bash
mkdir -p experiments/covid_ct_md

# In-domain performance (MosMedData)
python scripts/evaluate_model.py \
  --model-path experiments/mosmeddata/s3_analysis/seed_42/checkpoints/best_model.pth \
  --config configs/parvae_s3.yaml \
  --data-dir data/mosmeddata \
  --output-dir experiments/covid_ct_md/indomain

# Transfer to COVID-CT-MD (frozen encoder + fine-tune predictor)
python scripts/covid_ct_md_evaluation.py \
  --model-path experiments/mosmeddata/s3_analysis/seed_42/checkpoints/best_model.pth \
  --config configs/transfer_learning.yaml \
  --data-dir data/covid_ct_md \
  --output-dir experiments/covid_ct_md/transfer_frozen \
  --freeze-encoder

# Retrain from scratch on COVID-CT-MD
python scripts/train_parvae.py \
  --config configs/parvae_s3.yaml \
  --data-dir data/covid_ct_md \
  --output-dir experiments/covid_ct_md/retrain_scratch
```

## Critical Implementation Details

### 1. Random Seed Management

Ensure deterministic results across runs:

```python
# In config files:
seed: 42
torch_deterministic: true
torch_benchmark: false

# PyTorch settings
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 2. Data Split Consistency

All 3 seeds must use the **same train/val/test split**:

```bash
# Generate split once
python scripts/create_splits.py \
  --data-dir data/mosmeddata \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --output-dir data/mosmeddata/splits \
  --seed 0  # Fixed seed for split

# Use same split for all training runs
# (script automatically applies this)
```

### 3. Model Configuration Consistency

All S3 runs must use identical hyperparameters:

```yaml
# configs/parvae_s3.yaml (consistent across all seeds)
model:
  latent_dim: 85
  reconstruction_loss: mse
  optimizer: adam
  learning_rate: 1.0e-3

training:
  epochs: 50
  batch_size: 32
  annealing_schedule: three_phase
  beta_range: [1.0e-4, 5.0e-4]
  lambda_range: [1.5, 3.0]
```

### 4. Validation Metrics

Monitor during training to ensure convergence:

```python
# Expected KL divergence trajectory (S3)
Epoch 5:  KL ≈ 2-3   (physics lock-in phase)
Epoch 20: KL ≈ 12-14 (transition)
Epoch 50: KL ≈ 15-17 (final)
```

If KL < 5 at epoch 50 → **posterior collapse** → check annealing schedule

## Verification Checklist

After reproduction, verify:

- [ ] S1 AUC: 67.4 ± 1.4% (yours within 1%)
- [ ] S2 AUC: 74.6 ± 0.8% (yours within 1%)
- [ ] S3 AUC: 99.3 ± 1.0% (yours matches)
- [ ] Physics R² (S3): 0.972 (yours ≥ 0.96)
- [ ] Transfer R² (COVID-CT-MD): 0.320 (yours within 5%)
- [ ] 3 seeds show convergence: Std < 3%

## Expected Computational Resources

| Component | GPU VRAM | Time (V100) | Time (A100) |
|-----------|----------|------------|-----------|
| Training 1 model | 8GB | 2 hours | 45 min |
| 3 seeds × 3 tasks | 8GB | ~18 hours | ~6 hours |
| All experiments | 8GB | ~72 hours | ~24 hours |
| Batch evaluation | 8GB | 30 min | 10 min |

**Memory settings:** Code automatically adjusts batch_size if OOM detected

## Debugging Failed Reproduction

**If results differ by >5%:**

1. Check Python version: `python --version` (must be 3.11)
2. Verify PyTorch version: `pip show torch`
3. Check CUDA: Are you using same GPU type?
4. Verify data integrity: `python scripts/data_integrity_checks.py`
5. Check config files: Ensure they haven't been modified

**If training is slower than expected:**

1. Monitor GPU usage: `nvidia-smi` (should be near 100%)
2. Check mixed precision: Is it enabled correctly?
3. Verify DataLoader workers: `--n-workers 8`

## Next Steps

- Review [QUICKSTART.md](QUICKSTART.md) for simplified walkthrough
- Check [INSTALLATION.md](INSTALLATION.md) for setup troubleshooting
- See [API.md](API.md) for extending the code
