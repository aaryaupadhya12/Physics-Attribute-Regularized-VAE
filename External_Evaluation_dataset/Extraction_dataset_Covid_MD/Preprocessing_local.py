"""
COVID-CT-MD Local CPU Preprocessing Script
==========================================
Runs entirely on CPU. Outputs:
  - .npy files (ct + mask) for PAR-VAE training on Kaggle
  - .png files (ct windowed + mask + overlay) for local visual QC
  - full_balanced.csv, train.csv, val.csv, test.csv

Folder structure expected:
    COVID-19 Cases/  P001/ P002/ ... (DICOM files inside)
    Normal Cases/    normal001/ normal002/ ...
    Cap Cases/       cap001/  (SKIPPED)

Install dependencies first:
    pip install pydicom opencv-python scipy scikit-learn matplotlib tqdm pandas numpy

Usage:
    python preprocess_covidctmd_local.py

Then zip ct_npy/ + mask_npy/ + CSVs and upload to Kaggle as private dataset.
PNGs stay local for visual verification only.
"""

import os
import numpy as np
import pandas as pd
import pydicom
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from sklearn.model_selection import train_test_split
import warnings
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

DATASET_ROOT = r"C:\Users\Aarya-2\Documents\ADOG\PAR-VAE\COVID-CT-MD-DATASET"
CSV_DIR      = r"C:\Users\Aarya-2\Documents\ADOG\PAR-VAE\COVID-CT-MD-DATASET"
OUTPUT_DIR   = r"C:\Users\Aarya-2\Documents\ADOG\PAR-VAE\dataset_COVID_MD_PROCESSED"

COVID_DIR    = os.path.join(DATASET_ROOT, "COVID-19 Cases")
NORMAL_DIR   = os.path.join(DATASET_ROOT, "Normal Cases")

SLICE_SIZE   = 512
MAX_PER_VOL  = 30
SEED         = 999
# ============================================================

# Output subdirectories
CT_NPY_DIR   = os.path.join(OUTPUT_DIR, 'ct_npy')
MASK_NPY_DIR = os.path.join(OUTPUT_DIR, 'mask_npy')
CT_PNG_DIR   = os.path.join(OUTPUT_DIR, 'ct_png')
MASK_PNG_DIR = os.path.join(OUTPUT_DIR, 'mask_png')
OVERLAY_DIR  = os.path.join(OUTPUT_DIR, 'overlays')

for d in [CT_NPY_DIR, MASK_NPY_DIR, CT_PNG_DIR, MASK_PNG_DIR, OVERLAY_DIR]:
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, 'preprocessing_log.txt'),
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)

np.random.seed(SEED)


# ============================================================
# DICOM LOADING
# ============================================================
def load_dicom_volume(patient_dir):
    """
    Load all DICOM slices from a patient folder.
    Sorts by SliceLocation or InstanceNumber.
    Returns HU volume shape (H, W, N_slices) and patient_id string.
    """
    all_files = sorted(Path(patient_dir).iterdir())
    slices = []

    for f in all_files:
        try:
            dcm = pydicom.dcmread(str(f), force=True)
            if not hasattr(dcm, 'pixel_array'):
                continue
            slices.append(dcm)
        except Exception:
            continue

    if len(slices) == 0:
        return None, None

    # Sort by SliceLocation → InstanceNumber → filename order
    try:
        slices.sort(key=lambda s: float(s.SliceLocation))
    except AttributeError:
        try:
            slices.sort(key=lambda s: float(s.InstanceNumber))
        except AttributeError:
            pass

    volume = []
    for dcm in slices:
        img       = dcm.pixel_array.astype(np.float32)
        slope     = float(getattr(dcm, 'RescaleSlope',     1))
        intercept = float(getattr(dcm, 'RescaleIntercept', -1024))
        hu        = img * slope + intercept
        volume.append(hu)

    volume     = np.stack(volume, axis=-1)  # (H, W, N)
    patient_id = Path(patient_dir).name
    return volume, patient_id


# ============================================================
# LUNG MASK  (identical to PAR-VAE pipeline)
# ============================================================
def simple_lung_mask(hu_slice):
    binary = (hu_slice >= -900) & (hu_slice <= -100)
    struct = np.ones((5, 5), dtype=bool)
    binary = ndimage.binary_closing(binary, structure=struct)
    binary = ndimage.binary_opening(binary, structure=struct)
    return binary.astype(np.float32)


def clean_lung_mask(mask):
    """Keep only two largest connected components (left + right lung)."""
    mask_bool = mask > 0.5
    labeled, n = ndimage.label(mask_bool)
    if n == 0:
        return mask
    if n <= 2:
        return mask
    sizes = ndimage.sum(mask_bool, labeled, range(1, n + 1))
    top2  = np.argsort(sizes)[-2:]
    clean = np.zeros_like(mask)
    for idx in top2:
        clean[labeled == (idx + 1)] = 1
    return clean.astype(np.float32)


def is_valid_mask(mask, min_pixels=1000, max_fraction=0.60):
    pixel_count = mask.sum()
    if pixel_count < min_pixels:
        return False
    if pixel_count / (mask.shape[0] * mask.shape[1]) > max_fraction:
        return False
    return True


# ============================================================
# NORMALIZATION  (matches PAR-VAE exactly)
# ============================================================
def normalize_for_model(hu_slice):
    """
    Clip to lung window [-1000, 400] then normalize to [-1, 1].
    Inverse of: ct_hu = (ct+1) * 1400 - 1000
    """
    clipped    = np.clip(hu_slice, -1000.0, 400.0)
    normalized = (clipped + 1000.0) / 1400.0 * 2.0 - 1.0
    return normalized.astype(np.float32)


# ============================================================
# PNG SAVING
# ============================================================
def save_ct_png(hu_slice, path, wc=-600, ww=1500):
    """Lung-windowed grayscale PNG."""
    lo       = wc - ww / 2
    hi       = wc + ww / 2
    windowed = np.clip(hu_slice, lo, hi)
    windowed = ((windowed - lo) / (hi - lo) * 255).astype(np.uint8)
    if windowed.shape[:2] != (SLICE_SIZE, SLICE_SIZE):
        windowed = cv2.resize(windowed, (SLICE_SIZE, SLICE_SIZE))
    cv2.imwrite(path, windowed)


def save_mask_png(mask, path):
    """Binary mask as PNG (white=lung)."""
    m = (mask * 255).astype(np.uint8)
    if m.shape[:2] != (SLICE_SIZE, SLICE_SIZE):
        m = cv2.resize(m, (SLICE_SIZE, SLICE_SIZE))
    cv2.imwrite(path, m)


def save_overlay_png(hu_slice, mask, path, wc=-600, ww=1500):
    """CT slice with green lung mask overlay."""
    lo       = wc - ww / 2
    hi       = wc + ww / 2
    windowed = np.clip(hu_slice, lo, hi)
    windowed = ((windowed - lo) / (hi - lo) * 255).astype(np.uint8)
    rgb      = cv2.cvtColor(windowed, cv2.COLOR_GRAY2BGR)
    overlay  = rgb.copy()
    overlay[mask > 0.5] = [0, 200, 0]
    combined = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)
    if combined.shape[:2] != (SLICE_SIZE, SLICE_SIZE):
        combined = cv2.resize(combined, (SLICE_SIZE, SLICE_SIZE))
    cv2.imwrite(path, combined)


# ============================================================
# PROCESS ONE CLASS
# ============================================================
def process_class(class_dir, label, label_name):
    patient_dirs = sorted([d for d in Path(class_dir).iterdir() if d.is_dir()])
    print(f"\nProcessing {label_name} ({len(patient_dirs)} patients)...")

    records = []
    skipped = 0

    for patient_dir in tqdm(patient_dirs, desc=label_name):
        try:
            volume, patient_id = load_dicom_volume(str(patient_dir))

            if volume is None or volume.shape[2] < 10:
                logging.warning(f"Skipped {patient_dir.name}: bad volume")
                skipped += 1
                continue

            n_slices  = volume.shape[2]
            start_idx = n_slices // 4
            end_idx   = 3 * n_slices // 4
            n_extract = min(MAX_PER_VOL, end_idx - start_idx)
            indices   = np.linspace(start_idx, end_idx - 1, n_extract, dtype=int)

            for slice_idx in indices:
                hu = volume[:, :, slice_idx].copy()

                if hu.shape[:2] != (SLICE_SIZE, SLICE_SIZE):
                    hu = cv2.resize(hu, (SLICE_SIZE, SLICE_SIZE),
                                    interpolation=cv2.INTER_LINEAR)

                mask = simple_lung_mask(hu)
                mask = clean_lung_mask(mask)

                if not is_valid_mask(mask):
                    skipped += 1
                    continue

                ct_norm = normalize_for_model(hu)
                stem    = f"{label_name}_{patient_id}_slice_{slice_idx:03d}"

                # NPY — for Kaggle training
                ct_npy_path   = os.path.join(CT_NPY_DIR,   f"{stem}_ct.npy")
                mask_npy_path = os.path.join(MASK_NPY_DIR, f"{stem}_mask.npy")
                np.save(ct_npy_path,   ct_norm)
                np.save(mask_npy_path, mask)

                # PNG — for local visual QC only
                ct_png_path   = os.path.join(CT_PNG_DIR,   f"{stem}_ct.png")
                mask_png_path = os.path.join(MASK_PNG_DIR, f"{stem}_mask.png")
                overlay_path  = os.path.join(OVERLAY_DIR,  f"{stem}_overlay.png")
                save_ct_png(hu, ct_png_path)
                save_mask_png(mask, mask_png_path)
                save_overlay_png(hu, mask, overlay_path)

                records.append({
                    'id':          stem,
                    'patient_id':  patient_id,
                    'ct_path':     ct_npy_path,
                    'mask_path':   mask_npy_path,
                    'ct_png':      ct_png_path,
                    'mask_png':    mask_png_path,
                    'overlay_png': overlay_path,
                    'label':       label,
                    'slice_idx':   int(slice_idx),
                    'n_slices':    n_slices
                })

        except Exception as e:
            logging.error(f"Error {patient_dir.name}: {str(e)[:200]}")
            skipped += 1
            continue

    print(f"  Done: {len(records)} slices from {len(patient_dirs)} patients, {skipped} skipped")
    return records


# ============================================================
# VERIFICATION GRID
# ============================================================
def save_verification_grid(records, label_name, n=16):
    sample = np.random.choice(records, size=min(n, len(records)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    for i, rec in enumerate(sample):
        img = cv2.imread(rec['overlay_png'])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
        axes[i].set_title(f"{rec['patient_id']} s{rec['slice_idx']}", fontsize=7)
        axes[i].axis('off')
    plt.suptitle(f"{label_name} — Verification Grid (green = lung mask)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"verification_grid_{label_name}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# LEAKAGE CHECK
# ============================================================
def check_leakage(train_df, val_df, test_df):
    train_p = set(train_df['patient_id'])
    val_p   = set(val_df['patient_id'])
    test_p  = set(test_df['patient_id'])
    assert len(train_p & val_p)  == 0, "LEAKAGE: Train/Val"
    assert len(train_p & test_p) == 0, "LEAKAGE: Train/Test"
    assert len(val_p   & test_p) == 0, "LEAKAGE: Val/Test"
    print("  No patient-level leakage confirmed.")


# ============================================================
# MAIN
# ============================================================
def main():

    covid_records  = process_class(COVID_DIR,  label=1, label_name='covid')
    normal_records = process_class(NORMAL_DIR, label=0, label_name='normal')

    # Balance
    n = min(len(covid_records), len(normal_records))
    np.random.seed(SEED)
    covid_bal  = list(np.random.choice(covid_records,  n, replace=False))
    normal_bal = list(np.random.choice(normal_records, n, replace=False))
    all_records = covid_bal + normal_bal
    np.random.shuffle(all_records)

    df = pd.DataFrame(all_records)
    print(f"\nBalanced: COVID={len(df[df['label']==1])} Normal={len(df[df['label']==0])} Total={len(df)}")

    # Verification grids
    print("\nSaving verification grids...")
    save_verification_grid(covid_records,  'covid')
    save_verification_grid(normal_records, 'normal')

    # Patient-level split 70/15/15
    patients      = df['patient_id'].unique()
    pat_labels    = df.groupby('patient_id')['label'].first().reindex(patients).values

    train_p, temp_p, _, temp_l = train_test_split(
        patients, pat_labels, test_size=0.30, random_state=SEED, stratify=pat_labels)
    val_p, test_p, _, _ = train_test_split(
        temp_p, temp_l, test_size=0.50, random_state=SEED, stratify=temp_l)

    train_df = df[df['patient_id'].isin(train_p)].reset_index(drop=True)
    val_df   = df[df['patient_id'].isin(val_p)].reset_index(drop=True)
    test_df  = df[df['patient_id'].isin(test_p)].reset_index(drop=True)

    check_leakage(train_df, val_df, test_df)

    print(f"\nSplit Summary:")
    for name, split in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"  {name:5s}: {len(split):4d} slices | "
              f"{split['patient_id'].nunique()} patients | "
              f"COVID={int((split['label']==1).sum())} "
              f"Normal={int((split['label']==0).sum())}")

    # Save CSVs
    df.to_csv(      os.path.join(OUTPUT_DIR, 'full_balanced.csv'), index=False)
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'),         index=False)
    val_df.to_csv(  os.path.join(OUTPUT_DIR, 'val.csv'),           index=False)
    test_df.to_csv( os.path.join(OUTPUT_DIR, 'test.csv'),          index=False)

    print(f"\n{'='*60}")
    print(f"DONE. Output: {OUTPUT_DIR}")
    print(f"\nUpload to Kaggle (zip these):")
    print(f"  ct_npy/           ← training data")
    print(f"  mask_npy/         ← training data")
    print(f"  full_balanced.csv")
    print(f"  train.csv / val.csv / test.csv")
    print(f"\nKeep local (verification only):")
    print(f"  ct_png/ mask_png/ overlays/")
    print(f"  verification_grid_covid.png")
    print(f"  verification_grid_normal.png")

    return df, train_df, val_df, test_df


if __name__ == '__main__':
    df, train_df, val_df, test_df = main()