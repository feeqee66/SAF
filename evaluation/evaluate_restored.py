from pathlib import Path
import argparse
import csv
import numpy as np

from metrics import calculate_metrics
from edge_metrics import calculate_edge_metrics


# ============================================================
# IMAGE LOADING
# ============================================================

def load_npy_image(path):
    """
    Load a SAF .npy grayscale image.

    Images are already stored approximately in [0, 1].
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
# EVALUATION
# ============================================================

def evaluate_restored(restored_dir, gt_dir, output_csv=None):

    restored_dir = Path(restored_dir)
    gt_dir = Path(gt_dir)

    # --------------------------------------------------------
    # Find files
    # --------------------------------------------------------

    restored_files = {
        path.stem: path
        for path in restored_dir.glob("*.npy")
    }

    gt_files = {
        path.stem: path
        for path in gt_dir.glob("*.npy")
    }

    print("=" * 70)
    print("RESTORED MODEL EVALUATION")
    print("=" * 70)

    print(f"Restored images : {len(restored_files)}")
    print(f"GT images       : {len(gt_files)}")

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not restored_files:
        raise RuntimeError(
            f"No .npy files found in:\n{restored_dir}"
        )

    if not gt_files:
        raise RuntimeError(
            f"No .npy files found in:\n{gt_dir}"
        )

    # --------------------------------------------------------
    # Find common filenames
    # --------------------------------------------------------

    common_ids = sorted(
        set(restored_files.keys())
        &
        set(gt_files.keys())
    )

    missing_gt = sorted(
        set(restored_files.keys())
        -
        set(gt_files.keys())
    )

    missing_restored = sorted(
        set(gt_files.keys())
        -
        set(restored_files.keys())
    )

    print()
    print(f"Matching pairs  : {len(common_ids)}")
    print(f"Missing GT      : {len(missing_gt)}")
    print(f"Missing restored: {len(missing_restored)}")

    if len(common_ids) == 0:
        raise RuntimeError(
            "No matching filenames were found."
        )

    # --------------------------------------------------------
    # Metric accumulators
    # --------------------------------------------------------

    total = {
        "MAE": 0.0,
        "MSE": 0.0,
        "PSNR": 0.0,
        "SSIM": 0.0,
        "Edge_Error": 0.0,
        "Edge_Preservation": 0.0
    }

    valid = 0

    # Per-image results
    per_image_results = []

    # ========================================================
    # PROCESS MATCHING PAIRS
    # ========================================================

    for index, image_id in enumerate(
        common_ids,
        start=1
    ):

        restored_path = restored_files[image_id]
        gt_path = gt_files[image_id]

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        restored = load_npy_image(
            restored_path
        )

        gt = load_npy_image(
            gt_path
        )

        # ----------------------------------------------------
        # Shape check
        # ----------------------------------------------------

        if restored.shape != gt.shape:

            raise RuntimeError(
                f"Shape mismatch for {image_id}:\n"
                f"Restored: {restored.shape}\n"
                f"GT      : {gt.shape}"
            )

        # ----------------------------------------------------
        # Basic metrics
        # ----------------------------------------------------

        basic = calculate_metrics(
            restored,
            gt
        )

        # ----------------------------------------------------
        # Edge metrics
        # ----------------------------------------------------

        edge = calculate_edge_metrics(
            restored,
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
        # Save per-image result
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
    # DATASET AVERAGES
    # ========================================================

    average = {
        key: total[key] / valid
        for key in total
    }

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("RESTORATION EVALUATION COMPLETE")
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

    print()
    print("DATA COVERAGE")
    print("-" * 70)

    print(
        f"Restored available : "
        f"{len(restored_files)}"
    )

    print(
        f"GT available       : "
        f"{len(gt_files)}"
    )

    print(
        f"Evaluated pairs    : "
        f"{len(common_ids)}"
    )

    print(
        f"Missing GT         : "
        f"{len(missing_gt)}"
    )

    print(
        f"Missing restored   : "
        f"{len(missing_restored)}"
    )

    print("=" * 70)

    # ========================================================
    # SAVE PER-IMAGE CSV
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
            "Evaluate restored SAF images against GT "
            "using matching filenames."
        )
    )

    parser.add_argument(
        "--restored",
        required=True,
        help="Directory containing restored .npy images."
    )

    parser.add_argument(
        "--gt",
        default=(
            r"C:\Users\apurva vivobook"
            r"\Downloads\train\train\GT"
        ),
        help="Directory containing GT .npy images."
    )

    parser.add_argument(
        "--output",
        default=(
            "evaluation/results/"
            "model1_per_image.csv"
        ),
        help="Output CSV for per-image metrics."
    )

    args = parser.parse_args()

    evaluate_restored(
        restored_dir=args.restored,
        gt_dir=args.gt,
        output_csv=args.output
    )