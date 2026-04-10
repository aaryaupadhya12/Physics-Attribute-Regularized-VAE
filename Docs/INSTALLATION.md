# Installation Guide

Detailed setup instructions for PAR-VAE.

## System Requirements

- **OS:** Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10+
- **Python:** 3.11+
- **RAM:** 16GB minimum (32GB recommended)
- **GPU:** NVIDIA CUDA 12.4+ compatible (optional but recommended)
  - VRAM: 8GB minimum (12GB recommended for batch size 32)

## Step-by-Step Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/PAR-VAE.git
cd PAR-VAE
```

### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows
```

**Using conda (alternative):**
```bash
conda create -n parvae python=3.11
conda activate parvae
```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Verify Installation

Check Python packages:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import nibabel; print(f'NiBabel: {nibabel.__version__}')"
python -c "import skimage; print(f'scikit-image: {skimage.__version__}')"
```

Check GPU availability:
```bash
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'Current GPU: {torch.cuda.get_device_name(0)}')"
```

Check module imports:
```bash
python -c "from src.models import VAE; print('VAE imported successfully')"
python -c "from src.data import CTPatchDataset; print('Dataset imported successfully')"
python -c "from src.utils import PhysicsFeatureExtractor; print('PhysicsFeatureExtractor imported successfully')"
```

## Troubleshooting

### CUDA/GPU Issues

If GPU is not detected:

1. **Check CUDA installation:**
   ```bash
   nvcc --version
   ```

2. **Verify PyTorch can see GPU:**
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. **Reinstall PyTorch for your CUDA version:**
   ```bash
   # For CUDA 12.4
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

### Memory Issues

If you encounter out-of-memory errors:

1. Reduce batch size in config: `batch_size: 16` (default: 32)
2. Reduce latent dimension: `latent_dim: 64` (default: 85)
3. Use gradient accumulation: `gradient_accumulation_steps: 2`

### ImportError for nibabel or pydicom

```bash
pip install --upgrade nibabel pydicom SimpleITK
```

## Environment Variables (Optional)

Set for improved performance:

```bash
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1  # For multi-GPU
export PYTHONUNBUFFERED=1
```

## Docker Setup (Optional)

Create `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip git

WORKDIR /workspace

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python3.11"]
```

Build and run:

```bash
docker build -t parvae:v1.0 .
docker run --gpus all -v /path/to/data:/workspace/data parvae:v1.0 scripts/train_parvae.py --config configs/parvae_s3.yaml
```

## Next Steps

After successful installation:

1. Review [QUICKSTART.md](QUICKSTART.md) for basic usage
2. Download datasets (see [DATASET.md](DATASET.md))
3. Run integrity checks: `python scripts/data_integrity_checks.py`
4. Train your first model (see [QUICKSTART.md](QUICKSTART.md))
