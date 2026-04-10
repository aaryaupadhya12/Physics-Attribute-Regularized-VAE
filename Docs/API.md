# API Documentation

Complete API reference for PAR-VAE source code.

## Table of Contents

1. [Models](#models)
2. [Data Handling](#data-handling)
3. [Utils](#utils)
4. [Evaluation](#evaluation)

---

## Models

### `src.models.VAE`

**Variational Autoencoder with physics attribute regularization.**

```python
from src.models import VAE

model = VAE(
    input_channels=1,
    latent_dim=85,
    reconstruction_loss='mse',
    device='cuda:0'
)
```

#### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_channels` | int | 1 | Number of input image channels |
| `latent_dim` | int | 85 | Dimensionality of latent space |
| `reconstruction_loss` | str | 'mse' | 'mse' or 'l1' |
| `device` | str | 'cuda:0' | Device placement |

#### Methods

**`forward(x)`** → tuple(reconstruction, mu, logvar)
```python
x = torch.randn(32, 1, 512, 512)  # Batch of CT slices
recon, mu, logvar = model(x)
```

**`encode(x)`** → tuple(mu, logvar)
```python
mu, logvar = model.encode(x)
latent = model.reparameterize(mu, logvar)
```

**`decode(z)`** → reconstruction
```python
z = torch.randn(32, 85)  # Latent codes
recon = model.decode(z)
```

**`sample(n_samples)`** → samples
```python
samples = model.sample(n_samples=16)
```

#### Properties
| Property | Type | Description |
|----------|------|-------------|
| `encoder` | nn.Module | Encoder network |
| `decoder` | nn.Module | Decoder network |
| `latent_dim` | int | Latent dimension |

---

### `src.models.PhysicsRegularizer`

**Physics-based loss term for attribute alignment.**

```python
from src.models import PhysicsRegularizer

regularizer = PhysicsRegularizer(
    num_physics_features=14,
    latent_dim=85,
    weight=3.0
)
```

#### Methods

**`compute_loss(latent, physics_features)`** → loss
```python
# latent: (batch_size, latent_dim)
# physics_features: (batch_size, 14)
loss = regularizer(latent, physics_features)
```

**Parameters:**
- `latent`: Tensor of shape (N, 85) - encoded features
- `physics_features`: Tensor of shape (N, 14) - ground truth physics features

**Returns:** Scalar loss tensor

---

### `src.models.AnnelingScheduler`

**3-phase annealing schedule for hyperparameters.**

```python
from src.models import AnnelingScheduler

scheduler = AnnelingScheduler(
    total_epochs=50,
    beta_range=(1e-4, 5e-4),
    lambda_range=(1.5, 3.0),
    strategy='three_phase'
)
```

#### Methods

**`get_params(current_epoch)`** → dict
```python
epoch = 25
params = scheduler.get_params(epoch)
# {'beta': 0.00035, 'lambda': 2.25, 'phase': 'balance'}
```

---

## Data Handling

### `src.data.CTPatchDataset`

**PyTorch Dataset for CT slices and physics features.**

```python
from src.data import CTPatchDataset
import torch.utils.data as data

# Initialize dataset
dataset = CTPatchDataset(
    csv_file='data/train.csv',
    data_dir='data/mosmeddata',
    image_size=512,
    patch_size=32,
    normalize_hu=True,
    augment=True
)

# Create DataLoader
loader = data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

# Iterate through batches
for batch in loader:
    images = batch['image']  # (32, 1, 512, 512)
    labels = batch['label']  # (32,)
    physics = batch['physics_features']  # (32, 14)
```

#### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_file` | str | - | Path to CSV index |
| `data_dir` | str | - | Root data directory |
| `image_size` | int | 512 | Resized height/width |
| `patch_size` | int | 32 | Patch size for extraction |
| `normalize_hu` | bool | True | Normalize to [-1, 1] |
| `augment` | bool | False | Random augmentation |

#### Methods

**`__getitem__(idx)`** → dict
```python
sample = dataset[0]
# {
#   'image': Tensor(1, 512, 512),
#   'label': 0 (int),
#   'physics_features': Tensor(14),
#   'patient_id': 'CT_000'
# }
```

**`__len__()`** → int
```python
n_samples = len(dataset)
```

---

### `src.data.DataLoader`

**Utilities for creating robust data loaders.**

```python
from src.data import get_data_loaders

train_loader, val_loader, test_loader = get_data_loaders(
    data_dir='data/mosmeddata',
    batch_size=32,
    num_workers=8,
    shuffle_train=True,
    drop_last=True
)
```

**Parameters:**
- `data_dir`: Root directory
- `batch_size`: Batch size
- `num_workers`: DataLoader workers
- `shuffle_train`: Shuffle training set
- `drop_last`: Drop incomplete batches

---

## Utils

### `src.utils.PhysicsFeatureExtractor`

**Extract 14 physics features from CT slices.**

```python
from src.utils import PhysicsFeatureExtractor
import numpy as np

extractor = PhysicsFeatureExtractor(
    pixel_spacing=0.5,  # mm per pixel
    glcm_distance=1
)

# Extract features for one slice
ct_slice = np.load('CT_000_slice.npy')  # (512, 512)
lung_mask = np.load('mask_000_slice.npy')  # (512, 512)

features = extractor.extract(
    ct_slice=ct_slice,
    lung_mask=lung_mask
)

print(features)
# {
#   'mean_hu': -625.3,
#   'std_hu': 58.2,
#   'hu_p10': -712.1,
#   'hu_p25': -670.5,
#   'hu_p50': -625.0,
#   'hu_p75': -580.2,
#   'hu_p90': -538.4,
#   'mask_area': 45230.5,
#   'fractional_occupancy': 0.285,
#   'gradient_mean': 12.3,
#   'gradient_std': 5.6,
#   'glcm_contrast': 285.2,
#   'glcm_homogeneity': 0.782,
#   'glcm_entropy': 6.8
# }
```

#### Methods

**`extract(ct_slice, lung_mask)`** → dict

Extract all 14 features for a single slice.

**Parameters:**
- `ct_slice`: numpy array (H, W) of HU values
- `lung_mask`: binary numpy array (H, W)

**Returns:** Dictionary with 14 feature keys

**`extract_batch(ct_slices, lung_masks)`** → numpy array

Extract features for batch of slices.

**Parameters:**
- `ct_slices`: numpy array (N, H, W)
- `lung_masks`: numpy array (N, H, W)

**Returns:** Numpy array of shape (N, 14)

---

### `src.utils.Metrics`

**Classification and physics alignment metrics.**

```python
from src.utils import Metrics

metrics = Metrics()

# Classification metrics
y_true = np.array([0, 1, 0, 1, 1])
y_pred = np.array([0, 1, 0, 0, 1])
y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8])

acc = metrics.accuracy(y_true, y_pred)
f1 = metrics.f1_score(y_true, y_pred)
auc = metrics.roc_auc_score(y_true, y_prob)
cm = metrics.confusion_matrix(y_true, y_pred)

# Physics alignment R²
pred_features = np.random.randn(100, 14)
true_features = np.random.randn(100, 14)
r2 = metrics.r2_score_per_feature(pred_features, true_features)
r2_mean = r2.mean()
```

#### Methods

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `accuracy(y_true, y_pred)` | 1D arrays | float | Classification accuracy |
| `f1_score(y_true, y_pred)` | 1D arrays | float | Weighted F1 score |
| `roc_auc_score(y_true, y_prob)` | 1D arrays | float | ROC-AUC |
| `confusion_matrix(y_true, y_pred)` | 1D arrays | array | 2×2 CM |
| `r2_score_per_feature(y_pred, y_true)` | 2D arrays | 1D array | R² per feature |
| `class_overlap(x_true, x_pred)` | 2D arrays | float | KL divergence |

---

### `src.utils.Visualization`

**Plotting utilities for results.**

```python
from src.utils import Visualization

viz = Visualization()

# Plot latent space
latents = np.random.randn(1000, 2)  # 2D for visualization
labels = np.random.randint(0, 3, 1000)

viz.plot_latent_space(latents, labels, 'latent_space.png')

# Plot physics alignment
true_phys = np.random.randn(100, 14)
pred_phys = true_phys + np.random.randn(100, 14) * 0.1

viz.plot_physics_alignment(true_phys, pred_phys, 'physics_r2.png')

# Plot ROC curve
y_true = np.random.randint(0, 2, 100)
y_prob = np.random.rand(100)

viz.plot_roc_curve(y_true, y_prob, 'roc_curve.png')
```

---

## Evaluation

### `src.evaluation.ClassifierEvaluator`

**Train and evaluate classifiers on VAE latent features.**

```python
from src.evaluation import ClassifierEvaluator

evaluator = ClassifierEvaluator()

# Load features
latents_train = np.load('train_latents.npy')
latents_test = np.load('test_latents.npy')
labels_train = np.load('train_labels.npy')
labels_test = np.load('test_labels.npy')

# Train LogReg classifier
lr_metrics = evaluator.train_logreg(
    X_train=latents_train,
    y_train=labels_train,
    X_test=latents_test,
    y_test=labels_test
)

print(lr_metrics)
# {
#   'accuracy': 0.973,
#   'auc': 0.993,
#   'f1': 0.967,
#   'specificity': 0.965,
#   'sensitivity': 0.981
# }

# Train SVM classifier
svm_metrics = evaluator.train_svm(
    X_train=latents_train,
    y_train=labels_train,
    X_test=latents_test,
    y_test=labels_test,
    kernel='rbf'
)
```

#### Methods

**`train_logreg(...)`** → dict

Train logistic regression classifier.

**`train_svm(..., kernel='rbf')`** → dict

Train SVM classifier with RBF kernel.

**`train_mlp(...)`** → dict

Train MLP classifier.

---

### `src.evaluation.TransferEvaluator`

**Evaluate cross-scanner transfer and domain shift.**

```python
from src.evaluation import TransferEvaluator

evaluator = TransferEvaluator()

# In-domain evaluation
indomain = evaluator.evaluate_transfer(
    model=model,
    dataloader=test_loader,
    device='cuda:0'
)

# Transfer evaluation (frozen encoder)
transfer = evaluator.evaluate_transfer(
    model=model,
    dataloader=transfer_test_loader,
    freeze_encoder=True,
    device='cuda:0'
)

# Quantify domain shift
shift_metrics = evaluator.compute_domain_shift(
    latents_source=indomain['latents'],
    latents_target=transfer['latents']
)
```

---

## Configuration Classes

### `src.utils.Config`

**YAML configuration loader.**

```python
from src.utils import Config

config = Config.from_yaml('configs/parvae_s3.yaml')

# Access config
print(config.model.latent_dim)  # 85
print(config.training.epochs)  # 50
print(config.training.beta)  # [1e-4, 5e-4]

# Update config
config.training.batch_size = 16
config.save('configs/parvae_s3_custom.yaml')
```

---

## Example: Full Training Pipeline

```python
import torch
from src.models import VAE, PhysicsRegularizer
from src.data import get_data_loaders
from src.utils import Config

# Load config
config = Config.from_yaml('configs/parvae_s3.yaml')

# Initialize model
model = VAE(
    latent_dim=config.model.latent_dim,
    device='cuda:0'
).to('cuda:0')

# Initialize optimizer & regularizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
regularizer = PhysicsRegularizer(num_physics_features=14)

# Load data
train_loader, val_loader, _ = get_data_loaders('data/mosmeddata')

# Training loop
for epoch in range(50):
    for batch in train_loader:
        x = batch['image'].to('cuda:0')
        physics = batch['physics_features'].to('cuda:0')
        
        # Forward pass
        recon, mu, logvar = model(x)
        
        # Compute losses
        recon_loss = torch.nn.functional.mse_loss(recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        physics_loss = regularizer(mu, physics)
        
        # Total loss
        total_loss = recon_loss + kl_loss + physics_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

---

## See Also

- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [REPRODUCIBILITY.md](REPRODUCIBILITY.md) - Reproducing paper results
- [DATASET.md](DATASET.md) - Dataset specifications
