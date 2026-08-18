from pathlib import Path
import argparse
import csv

import numpy as np
from PIL import Image

from metrics import calculate_metrics
from edge_metrics import calculate_edge_metrics


# ============================================================
# IMAGE LOADING
# ============================================================

def load_npy_image(path):
    """
    Load a grayscale .npy image and normalize it to [0, 1].
    """

    image = np.load(path).astype(np.float32)
    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D image, got {image.shape} "
            f"for {path.name}"
        )

    image = np.clip(image, 0.0, 1.0)

    return image


# ============================================================
# BICUBIC 2x UPSAMPLING
# ============================================================

def bicubic_upscale(image, target_shape):
    """
    Upscale a grayscale image using bicubic interpolation.

    Parameters
    ----------
    image : np.ndarray
        2D image in [0, 1].

    target_shape : tuple
        (height, width) of GT image.

    Returns
    -------
    np.ndarray
        Upscaled image in [0, 1].
    """

    target_h, target_w = target_shape

    # Convert [0,1] float image to uint8 for PIL
    image_uint8 = np.clip(
        image * 255.0,
        0,
        255
    ).astype(np.uint8)

    pil_image = Image.fromarray(
        image_uint8,
        mode="L"
    )

    upscaled = pil_image.resize(
        (target_w, target_h),
        resample=Image.Resampling.BICUBIC
    )

    # Convert back to float32 [0,1]
    upscaled = np.asarray(
        upscaled,
        dtype=np.float32
    ) / 255.0

    return upscaled


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_bicubic_subset(
    lr_dir,
    gt_dir,
    subset_dir,
    output_csv=None
):

    lr_dir = Path(lr_dir)
    gt_dir = Path(gt_dir)
    subset_dir = Path(subset_dir)

    # --------------------------------------------------------
    # Find restored/model-output filenames
    # --------------------------------------------------------

    subset_files = sorted(
        subset_dir.glob("*.npy")
    )

    if len(subset_files) == 0:
        raise RuntimeError(
            f"No .npy files found in:\n{subset_dir}"
        )

    # --------------------------------------------------------
    # Create lookup dictionaries
    # --------------------------------------------------------

    lr_files = {
        path.stem: path
        for path in lr_dir.glob("*.npy")
    }

    gt_files = {
        path.stem: path
        for path in gt_dir.glob("*.npy")
    }

    # --------------------------------------------------------
    # Find files available in all required locations
    # --------------------------------------------------------

    subset_ids = {
        path.stem
        for path in subset_files
    }

    common_ids = sorted(
        subset_ids
        &
        set(lr_files.keys())
        &
        set(gt_files.keys())
    )

    missing_lr = sorted(
        subset_ids - set(lr_files.keys())
    )

    missing_gt = sorted(
        subset_ids - set(gt_files.keys())
    )

    # ========================================================
    # HEADER
    # ========================================================

    print("=" * 70)
    print("BICUBIC 2x SUBSET EVALUATION")
    print("=" * 70)

    print(f"Model-output files : {len(subset_ids)}")
    print(f"Matching LR files  : {len(common_ids) - len(missing_gt)}")
    print(f"Matching GT files  : {len(common_ids) - len(missing_lr)}")
    print(f"Evaluation pairs   : {len(common_ids)}")

    if missing_lr:
        print(
            f"Missing LR files  : {len(missing_lr)}"
        )

    if missing_gt:
        print(
            f"Missing GT files  : {len(missing_gt)}"
        )

    if len(common_ids) == 0:
        raise RuntimeError(
            "No common filenames found."
        )

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------

    total = {
        "MAE": 0.0,
        "MSE": 0.0,
        "PSNR": 0.0,
        "SSIM": 0.0,
        "Edge_Error": 0.0,
        "Edge_Preservation": 0.0
    }

    per_image_results = []

    valid = 0

    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for index, image_id in enumerate(
        common_ids,
        start=1
    ):

        lr_path = lr_files[image_id]
        gt_path = gt_files[image_id]

        # ----------------------------------------------------
        # Load LR and GT
        # ----------------------------------------------------

        lr = load_npy_image(
            lr_path
        )

        gt = load_npy_image(
            gt_path
        )

        # ----------------------------------------------------
        # Bicubic upscale to GT resolution
        # ----------------------------------------------------

        bicubic = bicubic_upscale(
            lr,
            gt.shape
        )

        # ----------------------------------------------------
        # Shape check
        # ----------------------------------------------------

        if bicubic.shape != gt.shape:
            raise RuntimeError(
                f"Shape mismatch for {image_id}:\n"
                f"Bicubic: {bicubic.shape}\n"
                f"GT     : {gt.shape}"
            )

        # ----------------------------------------------------
        # Basic metrics
        # ----------------------------------------------------

        basic = calculate_metrics(
            bicubic,
            gt
        )

        # ----------------------------------------------------
        # Edge metrics
        # ----------------------------------------------------

        edge = calculate_edge_metrics(
            bicubic,
            gt
        )

        # ----------------------------------------------------
        # Combine
        # ----------------------------------------------------

        result = {
            **basic,
            **edge
        }

        # ----------------------------------------------------
        # Accumulate
        # ----------------------------------------------------

        for key in total:
            total[key] += result[key]

        # ----------------------------------------------------
        # Per-image result
        # ----------------------------------------------------

        per_image_results.append({
            "Image": image_id,
            "PSNR": result["PSNR"],
            "SSIM": result["SSIM"],
            "MAE": result["MAE"],
            "MSE": result["MSE"],
            "Edge_Error": result["Edge_Error"],
            "Edge_Preservation":
                result["Edge_Preservation"]
        })

        valid += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if valid % 500 == 0:
            print(
                f"Processed: "
                f"{valid}/{len(common_ids)}"
            )

    # ========================================================
    # AVERAGE METRICS
    # ========================================================

    average = {
        key: total[key] / valid
        for key in total
    }

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("BICUBIC 2x EVALUATION COMPLETE")
    print("=" * 70)

    print(f"Evaluated pairs: {valid}")

    print()
    print("IMAGE QUALITY")
    print("-" * 70)

    print(
        f"PSNR : "
        f"{average['PSNR']:.4f} dB"
    )

    print(
        f"SSIM : "
        f"{average['SSIM']:.4f}"
    )

    print(
        f"MAE  : "
        f"{average['MAE']:.6f}"
    )

    print(
        f"MSE  : "
        f"{average['MSE']:.6f}"
    )

    print()
    print("STRUCTURAL / EDGE QUALITY")
    print("-" * 70)

    print(
        f"Edge Error        : "
        f"{average['Edge_Error']:.6f}"
    )

    print(
        f"Edge Preservation : "
        f"{average['Edge_Preservation']:.6f}"
    )

    print("=" * 70)

    # ========================================================
    # SAVE CSV
    # ========================================================

    if output_csv is not None:

        output_csv = Path(output_csv)

        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output_csv,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "Image",
                    "PSNR",
                    "SSIM",
                    "MAE",
                    "MSE",
                    "Edge_Error",
                    "Edge_Preservation"
                ]
            )

            writer.writeheader()

            writer.writerows(
                per_image_results
            )

        print()
        print(
            f"Per-image results saved to:\n"
            f"{output_csv}"
        )

    return average


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate bicubic 2x baseline on the "
            "exact same filenames as a restored model."
        )
    )

    parser.add_argument(
        "--lr",
        default=(
            r"C:\Users\apurva vivobook"
            r"\Downloads\train\train\NoisyLR"
        ),
        help="Directory containing 128x128 LR .npy images."
    )

    parser.add_argument(
        "--gt",
        default=(
            r"C:\Users\apurva vivobook"
            r"\Downloads\train\train\GT"
        ),
        help="Directory containing 256x256 GT .npy images."
    )

    parser.add_argument(
        "--subset",
        required=True,
        help=(
            "Directory containing the restored-model "
            ".npy files. Only these filenames are evaluated."
        )
    )

    parser.add_argument(
        "--output",
        default=(
            "evaluation/results/"
            "bicubic_subset_per_image.csv"
        ),
        help="Output CSV file."
    )

    args = parser.parse_args()

    evaluate_bicubic_subset(
        lr_dir=args.lr,
        gt_dir=args.gt,
        subset_dir=args.subset,
        output_csv=args.output
    )