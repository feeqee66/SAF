from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


DATASET_ROOT = Path(r"C:\Users\apurva vivobook\Downloads\train\train")

GT_DIR = DATASET_ROOT / "GT"
NOISY_DIR = DATASET_ROOT / "NoisyLR"

# Choose samples
indices = [0, 100, 500, 1000, 2000, 3000]


for idx in indices:

    filename = f"{idx:06d}.npy"

    noisy = np.load(NOISY_DIR / filename)
    gt = np.load(GT_DIR / filename)

    # --------------------------------------------------------
    # DISPLAY NORMALIZATION ONLY
    # --------------------------------------------------------

    noisy_display = noisy.copy()

    noisy_min = noisy_display.min()
    noisy_max = noisy_display.max()

    noisy_display = (
        (noisy_display - noisy_min)
        / (noisy_max - noisy_min + 1e-12)
    )

    gt_display = gt.copy()

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # NoisyLR
    axes[0, 0].imshow(noisy_display, cmap="gray")
    axes[0, 0].set_title(f"NoisyLR {filename}")
    axes[0, 0].axis("off")

    # GT
    axes[0, 1].imshow(gt_display, cmap="gray")
    axes[0, 1].set_title(f"GT {filename}")
    axes[0, 1].axis("off")

    # NoisyLR histogram
    axes[0, 2].hist(noisy.flatten(), bins=100)
    axes[0, 2].set_title("NoisyLR Histogram")

    # GT histogram
    axes[1, 0].hist(gt.flatten(), bins=100)
    axes[1, 0].set_title("GT Histogram")

    # GT gradient
    gx = np.diff(gt, axis=1)
    gy = np.diff(gt, axis=0)

    gradient = np.sqrt(
        gx[:-1, :] ** 2 +
        gy[:, :-1] ** 2
    )

    axes[1, 1].imshow(gradient, cmap="gray")
    axes[1, 1].set_title("GT Edge / Gradient Map")
    axes[1, 1].axis("off")

    # Statistics
    axes[1, 2].axis("off")

    stats = (
        f"File: {filename}\n\n"
        f"NoisyLR\n"
        f"Shape: {noisy.shape}\n"
        f"Min: {noisy.min():.4f}\n"
        f"Max: {noisy.max():.4f}\n"
        f"Mean: {noisy.mean():.4f}\n"
        f"Std: {noisy.std():.4f}\n\n"
        f"GT\n"
        f"Shape: {gt.shape}\n"
        f"Min: {gt.min():.4f}\n"
        f"Max: {gt.max():.4f}\n"
        f"Mean: {gt.mean():.4f}\n"
        f"Std: {gt.std():.4f}"
    )

    axes[1, 2].text(
        0.05,
        0.95,
        stats,
        verticalalignment="top",
        fontsize=10
    )

    plt.suptitle(f"Dataset Forensics — {filename}")

    plt.tight_layout()
    plt.show()