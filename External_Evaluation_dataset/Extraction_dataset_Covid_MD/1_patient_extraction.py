import pydicom
from pathlib import Path

patient_dir = r"C:\Users\Aarya-2\Documents\ADOG\PAR-VAE\COVID-CT-MD-DATASET\COVID-19 Cases\P001"

all_files = sorted(Path(patient_dir).iterdir())
print(f"Total files found: {len(all_files)}")

slices = []
for f in all_files[:3]:
    print(f"\nTrying: {f.name}")
    try:
        dcm = pydicom.dcmread(str(f), force=True)
        print(f"  Read OK")
        print(f"  has pixel_array: {hasattr(dcm, 'pixel_array')}")
        if hasattr(dcm, 'pixel_array'):
            print(f"  Shape: {dcm.pixel_array.shape}")
            slices.append(dcm)
        else:
            print(f"  Tags available: {[str(t) for t in list(dcm.keys())[:5]]}")
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nSlices loaded: {len(slices)}")
print(f"Volume would be None: {len(slices) == 0}")