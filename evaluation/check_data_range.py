from pathlib import Path
import numpy as np


ROOT = Path(
    r"C:\Users\apurva vivobook\Downloads\train\train"
)

GT_DIR = ROOT / "GT"
LR_DIR = ROOT / "NoisyLR"


def inspect(path):
    image = np.load(path).astype(np.float32)

    print(f"\nFile: {path.name}")
    print(f"Shape : {image.shape}")
    print(f"Dtype : {image.dtype}")
    print(f"Min   : {image.min()}")
    print(f"Max   : {image.max()}")
    print(f"Mean  : {image.mean()}")
    print(f"Std   : {image.std()}")


print("=" * 60)
print("SAF DATA RANGE CHECK")
print("=" * 60)

gt_files = sorted(GT_DIR.glob("*.npy"))
lr_files = sorted(LR_DIR.glob("*.npy"))

print("\nGROUND TRUTH")
inspect(gt_files[0])

print("\nNOISY LR")
inspect(lr_files[0])

print("=" * 60)