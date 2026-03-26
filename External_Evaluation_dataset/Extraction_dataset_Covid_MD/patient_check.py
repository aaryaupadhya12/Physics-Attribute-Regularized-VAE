# Run this FIRST before the main script to diagnose the issue
import pydicom
from pathlib import Path
import os

# Check what's actually inside P001
patient_dir = r"C:\Users\Aarya-2\Documents\ADOG\PAR-VAE\COVID-CT-MD-DATASET\COVID-19 Cases\P001"

print("Files in P001:")
for f in sorted(Path(patient_dir).iterdir()):
    print(f"  {f.name} | suffix: '{f.suffix}' | size: {f.stat().st_size} bytes")

print("\nTrying to read first file:")
files = sorted(Path(patient_dir).iterdir())
for f in files[:3]:
    try:
        dcm = pydicom.dcmread(str(f), force=True)
        print(f"  {f.name} -> OK | has pixel_array: {hasattr(dcm, 'pixel_array')}")
        if hasattr(dcm, 'pixel_array'):
            print(f"    Shape: {dcm.pixel_array.shape}")
            print(f"    RescaleSlope: {getattr(dcm, 'RescaleSlope', 'NOT FOUND')}")
            print(f"    RescaleIntercept: {getattr(dcm, 'RescaleIntercept', 'NOT FOUND')}")
    except Exception as e:
        print(f"  {f.name} -> ERROR: {e}")