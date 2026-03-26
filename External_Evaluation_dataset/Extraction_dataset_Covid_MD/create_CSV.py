import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 999
CT_NPY_DIR   = r"C:\Users\Aarya-2\Documents\ADOG\PAR-VAE\dataset_COVID_MD_PROCESSED\ct_npy"
MASK_NPY_DIR = r"C:\Users\Aarya-2\Documents\ADOG\PAR-VAE\dataset_COVID_MD_PROCESSED\mask_npy"
OUTPUT_DIR   = r"C:\Users\Aarya-2\Documents\ADOG\PAR-VAE\dataset_COVID_MD_PROCESSED"

np.random.seed(SEED)

# Rebuild records from existing files
records = []
for f in sorted(Path(CT_NPY_DIR).iterdir()):
    stem = f.stem.replace('_ct', '')
    mask_path = os.path.join(MASK_NPY_DIR, f"{stem}_mask.npy")
    
    if not os.path.exists(mask_path):
        continue
    
    # Extract info from filename
    # Format: covid_P001_slice_050_ct.npy or normal_normal001_slice_050_ct.npy
    parts = stem.split('_')
    label_name  = parts[0]                    # 'covid' or 'normal'
    label       = 1 if label_name == 'covid' else 0
    patient_id  = parts[1]                    # 'P001' or 'normal001'
    slice_idx   = int(parts[-1])              # last number

    records.append({
        'id':         stem,
        'patient_id': patient_id,
        'ct_path':    str(f),
        'mask_path':  mask_path,
        'label':      label,
    })

df = pd.DataFrame(records)
print(f"Total records: {len(df)}")
print(f"COVID:  {len(df[df['label']==1])}")
print(f"Normal: {len(df[df['label']==0])}")

# Balance
covid_df  = df[df['label'] == 1]
normal_df = df[df['label'] == 0]
n = min(len(covid_df), len(normal_df))
df = pd.concat([
    covid_df.sample(n, random_state=SEED),
    normal_df.sample(n, random_state=SEED)
]).sample(frac=1, random_state=SEED).reset_index(drop=True)

# Patient-level split
patients   = df['patient_id'].unique()
pat_labels = df.groupby('patient_id')['label'].first().reindex(patients).values

train_p, temp_p, _, temp_l = train_test_split(
    patients, pat_labels, test_size=0.30, random_state=SEED, stratify=pat_labels)
val_p, test_p, _, _ = train_test_split(
    temp_p, temp_l, test_size=0.50, random_state=SEED, stratify=temp_l)

train_df = df[df['patient_id'].isin(train_p)].reset_index(drop=True)
val_df   = df[df['patient_id'].isin(val_p)].reset_index(drop=True)
test_df  = df[df['patient_id'].isin(test_p)].reset_index(drop=True)

# Leakage check
assert len(set(train_p) & set(test_p)) == 0
assert len(set(val_p)   & set(test_p)) == 0
print("No leakage confirmed")

print(f"Train: {len(train_df)} slices, {len(train_p)} patients")
print(f"Val:   {len(val_df)} slices, {len(val_p)} patients")
print(f"Test:  {len(test_df)} slices, {len(test_p)} patients")

# Save
df.to_csv(      os.path.join(OUTPUT_DIR, 'full_balanced.csv'), index=False)
train_df.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'),         index=False)
val_df.to_csv(  os.path.join(OUTPUT_DIR, 'val.csv'),           index=False)
test_df.to_csv( os.path.join(OUTPUT_DIR, 'test.csv'),          index=False)
print("CSV creation ended")