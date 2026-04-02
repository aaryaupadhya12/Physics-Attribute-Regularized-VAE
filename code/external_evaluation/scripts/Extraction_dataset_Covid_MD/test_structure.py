import os
from pathlib import Path

DATASET_ROOT = r"C:\Users\Aarya-2\Documents\ADOG\PAR-VAE\COVID-CT-MD-DATASET"

print("Top level contents:")
for item in Path(DATASET_ROOT).iterdir():
    print(f"  {item.name} | is_dir: {item.is_dir()}")

print("\nFirst COVID patient contents:")
covid_dir = Path(DATASET_ROOT) / "COVID-19 Cases"
if covid_dir.exists():
    patients = sorted([d for d in covid_dir.iterdir() if d.is_dir()])
    if patients:
        p = patients[0]
        print(f"Inside {p.name}:")
        for f in sorted(p.iterdir())[:5]:
            print(f"  {f.name} | suffix: '{f.suffix}' | size: {f.stat().st_size}")
else:
    print("  'COVID-19 Cases' folder NOT found — listing what exists:")
    for item in Path(DATASET_ROOT).iterdir():
        print(f"  {item.name}")