from pathlib import Path
import argparse
import csv

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


# ============================================================
# LOAD NPY
# ============================================================

def load_npy_image(path):
    image = np.load(path).astype(np.float32)
    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D image, got {image.shape} "
            f"for {path}"
        )

    return np.clip(image, 0.0, 1.0)


# ============================================================
# BICUBIC 2x
# ============================================================

def bicubic_upscale(image, target_shape):

    target_h, target_w = target_shape

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

    return (
        np.asarray(
            upscaled,
            dtype=np.float32
        )
        / 255.0
    )


# ============================================================
# HIGH FREQUENCY
# ============================================================

def high_frequency(image, sigma=2.0):

    blurred = gaussian_filter(
        image,
        sigma=sigma
    )

    return image - blurred


# ============================================================
# FREQUENCY ENERGY
# ============================================================

def frequency_energy(image):

    hf = high_frequency(image)

    return float(
        np.mean(hf ** 2)
    )


# ============================================================
# FREQUENCY METRICS
# ============================================================

def frequency_preservation(
    prediction,
    gt
):

    prediction_energy = frequency_energy(
        prediction
    )

    gt_energy = frequency_energy(
        gt
    )

    if gt_energy == 0:
        return 1.0

    return float(
        min(
            prediction_energy / gt_energy,
            1.0
        )
    )


def frequency_error(
    prediction,
    gt
):

    prediction_energy = frequency_energy(
        prediction
    )

    gt_energy = frequency_energy(
        gt
    )

    if gt_energy == 0:
        return 0.0

    return float(
        abs(
            prediction_energy
            -
            gt_energy
        )
        /
        gt_energy
    )


# ============================================================
# MAIN
# ============================================================

def evaluate_bicubic_frequency(
    lr_dir,
    gt_dir,
    subset_dir,
    output_csv
):

    lr_dir = Path(lr_dir)
    gt_dir = Path(gt_dir)
    subset_dir = Path(subset_dir)
    output_csv = Path(output_csv)

    # --------------------------------------------------------
    # Model 1 filenames define the exact evaluation subset
    # --------------------------------------------------------

    subset_files = {
        p.stem: p
        for p in subset_dir.glob("*.npy")
    }

    lr_files = {
        p.stem: p
        for p in lr_dir.glob("*.npy")
    }

    gt_files = {
        p.stem: p
        for p in gt_dir.glob("*.npy")
    }

    common_ids = sorted(
        set(subset_files.keys())
        &
        set(lr_files.keys())
        &
        set(gt_files.keys())
    )

    if not common_ids:
        raise RuntimeError(
            "No matching files found."
        )

    print("=" * 70)
    print("BICUBIC FREQUENCY PRESERVATION")
    print("=" * 70)

    print(
        f"Evaluation images : "
        f"{len(common_ids)}"
    )

    total_preservation = 0.0
    total_error = 0.0

    rows = []

    # ========================================================
    # PROCESS
    # ========================================================

    for index, image_id in enumerate(
        common_ids,
        start=1
    ):

        lr = load_npy_image(
            lr_files[image_id]
        )

        gt = load_npy_image(
            gt_files[image_id]
        )

        # ----------------------------------------------------
        # Bicubic 2x
        # ----------------------------------------------------

        bicubic = bicubic_upscale(
            lr,
            gt.shape
        )

        # ----------------------------------------------------
        # Check shape
        # ----------------------------------------------------

        if bicubic.shape != gt.shape:
            raise RuntimeError(
                f"Shape mismatch for {image_id}: "
                f"{bicubic.shape} vs {gt.shape}"
            )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        preservation = frequency_preservation(
            bicubic,
            gt
        )

        error = frequency_error(
            bicubic,
            gt
        )

        total_preservation += preservation
        total_error += error

        rows.append({
            "Image": image_id,
            "Frequency_Preservation":
                preservation,
            "Frequency_Error":
                error
        })

        if index % 500 == 0:
            print(
                f"Processed: "
                f"{index}/{len(common_ids)}"
            )

    # ========================================================
    # AVERAGE
    # ========================================================

    avg_preservation = (
        total_preservation
        /
        len(common_ids)
    )

    avg_error = (
        total_error
        /
        len(common_ids)
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 70)
    print("BICUBIC FREQUENCY EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Images evaluated       : "
        f"{len(common_ids)}"
    )

    print(
        f"Frequency preservation : "
        f"{avg_preservation:.6f}"
    )

    print(
        f"Frequency error        : "
        f"{avg_error:.6f}"
    )

    print("=" * 70)

    # ========================================================
    # SAVE
    # ========================================================

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
                "Frequency_Preservation",
                "Frequency_Error"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Results saved to:\n"
        f"{output_csv}"
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lr",
        default=(
            r"C:\Users\apurva vivobook"
            r"\Downloads\train\train\NoisyLR"
        )
    )

    parser.add_argument(
        "--gt",
        default=(
            r"C:\Users\apurva vivobook"
            r"\Downloads\train\train\GT"
        )
    )

    parser.add_argument(
        "--subset",
        required=True,
        help=(
            "Model 1 output folder. "
            "Its filenames define the evaluation subset."
        )
    )

    parser.add_argument(
        "--output",
        default=(
            "evaluation/results/"
            "bicubic_frequency.csv"
        )
    )

    args = parser.parse_args()

    evaluate_bicubic_frequency(
        lr_dir=args.lr,
        gt_dir=args.gt,
        subset_dir=args.subset,
        output_csv=args.output
    )