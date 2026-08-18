from pathlib import Path
import argparse
import csv
import numpy as np

from metrics import calculate_metrics


# ============================================================
# IMAGE LOADING
# ============================================================

def load_npy_image(path):
    """
    Load normalized SAF .npy image.
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
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model_name, restored_dir, gt_dir):

    restored_dir = Path(restored_dir)
    gt_dir = Path(gt_dir)

    restored_files = sorted(
        restored_dir.glob("*.npy")
    )

    gt_files = sorted(
        gt_dir.glob("*.npy")
    )

    if len(restored_files) == 0:
        raise RuntimeError(
            f"No .npy files found for {model_name}:\n"
            f"{restored_dir}"
        )

    if len(restored_files) != len(gt_files):
        raise RuntimeError(
            f"{model_name}: image count mismatch:\n"
            f"Restored = {len(restored_files)}\n"
            f"GT       = {len(gt_files)}"
        )

    total = {
        "MAE": 0.0,
        "MSE": 0.0,
        "PSNR": 0.0,
        "SSIM": 0.0
    }

    valid = 0

    for restored_path, gt_path in zip(
        restored_files,
        gt_files
    ):

        if restored_path.stem != gt_path.stem:
            raise RuntimeError(
                f"{model_name}: filename mismatch:\n"
                f"Restored = {restored_path.name}\n"
                f"GT       = {gt_path.name}"
            )

        restored = load_npy_image(restored_path)
        gt = load_npy_image(gt_path)

        if restored.shape != gt.shape:
            raise RuntimeError(
                f"{model_name}: shape mismatch:\n"
                f"{restored_path.name}\n"
                f"Restored = {restored.shape}\n"
                f"GT       = {gt.shape}"
            )

        result = calculate_metrics(
            restored,
            gt
        )

        for key in total:
            total[key] += result[key]

        valid += 1

    average = {
        key: total[key] / valid
        for key in total
    }

    return average


# ============================================================
# SAVE CSV
# ============================================================

def save_results(results, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "Model",
        "PSNR",
        "SSIM",
        "MAE",
        "MSE"
    ]

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for model_name, metrics in results.items():

            writer.writerow({
                "Model": model_name,
                "PSNR": metrics["PSNR"],
                "SSIM": metrics["SSIM"],
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"]
            })


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_results(results):

    print()
    print("=" * 75)
    print("RESTORATION MODEL COMPARISON")
    print("=" * 75)

    print(
        f"{'Model':<20}"
        f"{'PSNR':>12}"
        f"{'SSIM':>12}"
        f"{'MAE':>15}"
        f"{'MSE':>15}"
    )

    print("-" * 75)

    for model_name, metrics in results.items():

        print(
            f"{model_name:<20}"
            f"{metrics['PSNR']:>12.4f}"
            f"{metrics['SSIM']:>12.4f}"
            f"{metrics['MAE']:>15.6f}"
            f"{metrics['MSE']:>15.6f}"
        )

    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Compare restoration models."
    )

    parser.add_argument(
        "--gt",
        required=True,
        help="Ground-truth directory."
    )

    parser.add_argument(
        "--model_a",
        required=True,
        help="Restored images from Model A."
    )

    parser.add_argument(
        "--model_b",
        required=True,
        help="Restored images from Model B."
    )

    parser.add_argument(
        "--output",
        default="evaluation/results/model_comparison.csv",
        help="Output CSV path."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results = {}

    print()
    print("Evaluating Model A...")

    results["Model_A"] = evaluate_model(
        "Model_A",
        args.model_a,
        args.gt
    )

    print("Model A complete.")

    print()
    print("Evaluating Model B...")

    results["Model_B"] = evaluate_model(
        "Model_B",
        args.model_b,
        args.gt
    )

    print("Model B complete.")

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_results(results)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        args.output
    )

    print()
    print(
        f"Results saved to:\n"
        f"{args.output}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()