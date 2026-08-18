from pathlib import Path

import numpy as np
from skimage.transform import resize

from metrics import calculate_metrics


# ============================================================
# SAF DATASET
# ============================================================

ROOT = Path(
    r"C:\Users\apurva vivobook\Downloads\train\train"
)

GT_DIR = ROOT / "GT"
LR_DIR = ROOT / "NoisyLR"


# ============================================================
# LOAD NPY IMAGE
# ============================================================

def load_npy_image(path):
    """
    Load SAF .npy image.

    SAF images are already stored as floating-point values
    approximately in the [0, 1] range.

    Noise can produce values slightly below 0 or above 1,
    so values are clipped to the valid image range.
    """

    image = np.load(path).astype(np.float32)

    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D image, got {image.shape} "
            f"for {path.name}"
        )

    # SAF data is already normalized.
    # Do NOT divide by 255.
    image = np.clip(image, 0.0, 1.0)

    return image


# ============================================================
# BICUBIC UPSAMPLING
# ============================================================

def bicubic_upscale(image, target_shape):
    """
    Upscale LR image to GT resolution using bicubic interpolation.
    """

    upscaled = resize(
        image,
        target_shape,
        order=3,
        mode="reflect",
        anti_aliasing=False,
        preserve_range=True
    )

    # Keep evaluation values in [0, 1]
    upscaled = np.clip(
        upscaled,
        0.0,
        1.0
    )

    return upscaled.astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("SAF DATASET BASELINE EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not GT_DIR.exists():
        raise FileNotFoundError(
            f"GT directory not found:\n{GT_DIR}"
        )

    if not LR_DIR.exists():
        raise FileNotFoundError(
            f"NoisyLR directory not found:\n{LR_DIR}"
        )

    # --------------------------------------------------------
    # Get files
    # --------------------------------------------------------

    gt_files = sorted(
        GT_DIR.glob("*.npy")
    )

    lr_files = sorted(
        LR_DIR.glob("*.npy")
    )

    print(f"GT images : {len(gt_files)}")
    print(f"LR images : {len(lr_files)}")

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    if len(gt_files) == 0:
        raise RuntimeError(
            "No GT .npy files found."
        )

    if len(lr_files) == 0:
        raise RuntimeError(
            "No NoisyLR .npy files found."
        )

    if len(gt_files) != len(lr_files):
        raise RuntimeError(
            f"GT/LR count mismatch:\n"
            f"GT = {len(gt_files)}\n"
            f"LR = {len(lr_files)}"
        )

    # --------------------------------------------------------
    # Verify corresponding filenames
    # --------------------------------------------------------

    for gt_path, lr_path in zip(
        gt_files,
        lr_files
    ):

        if gt_path.stem != lr_path.stem:

            raise RuntimeError(
                f"Filename mismatch:\n"
                f"GT = {gt_path.name}\n"
                f"LR = {lr_path.name}"
            )

    # --------------------------------------------------------
    # Metric accumulators
    # --------------------------------------------------------

    total_mae = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    valid_images = 0

    # ========================================================
    # PROCESS ALL 3200 PAIRS
    # ========================================================

    for index, (gt_path, lr_path) in enumerate(
        zip(gt_files, lr_files),
        start=1
    ):

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        gt = load_npy_image(gt_path)
        lr = load_npy_image(lr_path)

        # ----------------------------------------------------
        # First-image information
        # ----------------------------------------------------

        if index == 1:

            print()
            print("First image:")
            print(f"GT shape : {gt.shape}")
            print(f"LR shape : {lr.shape}")

            print()
            print("After normalization/clipping:")

            print(
                f"GT range : "
                f"{gt.min():.6f} → {gt.max():.6f}"
            )

            print(
                f"LR range : "
                f"{lr.min():.6f} → {lr.max():.6f}"
            )

        # ----------------------------------------------------
        # 2× Bicubic baseline
        # ----------------------------------------------------

        if lr.shape != gt.shape:

            lr = bicubic_upscale(
                lr,
                gt.shape
            )

        # ----------------------------------------------------
        # Final shape check
        # ----------------------------------------------------

        if lr.shape != gt.shape:

            raise RuntimeError(
                f"Shape mismatch after upsampling:\n"
                f"GT: {gt_path.name} {gt.shape}\n"
                f"LR: {lr_path.name} {lr.shape}"
            )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        result = calculate_metrics(
            lr,
            gt
        )

        total_mae += result["MAE"]
        total_mse += result["MSE"]
        total_psnr += result["PSNR"]
        total_ssim += result["SSIM"]

        valid_images += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if valid_images % 500 == 0:

            print(
                f"Processed: "
                f"{valid_images}/{len(gt_files)}"
            )

    # ========================================================
    # DATASET AVERAGES
    # ========================================================

    average_mae = (
        total_mae / valid_images
    )

    average_mse = (
        total_mse / valid_images
    )

    average_psnr = (
        total_psnr / valid_images
    )

    average_ssim = (
        total_ssim / valid_images
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("RAW LR BASELINE COMPLETE")
    print("=" * 60)

    print(f"Images: {valid_images}")

    print(
        f"PSNR : "
        f"{average_psnr:.4f} dB"
    )

    print(
        f"SSIM : "
        f"{average_ssim:.4f}"
    )

    print(
        f"MAE  : "
        f"{average_mae:.6f}"
    )

    print(
        f"MSE  : "
        f"{average_mse:.6f}"
    )

    print("=" * 60)
    print("Baseline: 2× Bicubic Upsampling")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()