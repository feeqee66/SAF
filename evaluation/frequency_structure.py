from pathlib import Path
import argparse
import csv

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


# ============================================================
# LOAD IMAGE
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
# HIGH-FREQUENCY MAP
# ============================================================

def high_frequency(image, sigma=2.0):
    """
    Extract fine/high-frequency structures.

    HF = image - Gaussian blurred image
    """

    blurred = gaussian_filter(
        image,
        sigma=sigma
    )

    return image - blurred


# ============================================================
# NORMALIZED CORRELATION
# ============================================================

def frequency_correlation(pred, gt, sigma=2.0):
    """
    Pearson correlation between predicted and GT
    high-frequency maps.

    +1 = highly similar high-frequency structure
     0 = little linear relationship
    -1 = opposite structure
    """

    pred_hf = high_frequency(
        pred,
        sigma=sigma
    ).flatten()

    gt_hf = high_frequency(
        gt,
        sigma=sigma
    ).flatten()

    pred_hf = pred_hf - np.mean(pred_hf)
    gt_hf = gt_hf - np.mean(gt_hf)

    denominator = (
        np.sqrt(np.sum(pred_hf ** 2))
        *
        np.sqrt(np.sum(gt_hf ** 2))
    )

    if denominator == 0:
        return 1.0

    return float(
        np.sum(pred_hf * gt_hf)
        /
        denominator
    )


# ============================================================
# HIGH-FREQUENCY MAE
# ============================================================

def frequency_mae(pred, gt, sigma=2.0):
    """
    Mean absolute error between predicted and GT
    high-frequency maps.
    """

    pred_hf = high_frequency(
        pred,
        sigma=sigma
    )

    gt_hf = high_frequency(
        gt,
        sigma=sigma
    )

    return float(
        np.mean(
            np.abs(
                pred_hf - gt_hf
            )
        )
    )


# ============================================================
# HIGH-FREQUENCY RMSE
# ============================================================

def frequency_rmse(pred, gt, sigma=2.0):
    """
    Root mean squared error between predicted
    and GT high-frequency maps.
    """

    pred_hf = high_frequency(
        pred,
        sigma=sigma
    )

    gt_hf = high_frequency(
        gt,
        sigma=sigma
    )

    return float(
        np.sqrt(
            np.mean(
                (pred_hf - gt_hf) ** 2
            )
        )
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_prediction(
    prediction_dir,
    gt_dir,
    output_csv
):

    prediction_dir = Path(
        prediction_dir
    )

    gt_dir = Path(
        gt_dir
    )

    output_csv = Path(
        output_csv
    )

    prediction_files = {
        p.stem: p
        for p in prediction_dir.glob("*.npy")
    }

    gt_files = {
        p.stem: p
        for p in gt_dir.glob("*.npy")
    }

    common_ids = sorted(
        set(prediction_files)
        &
        set(gt_files)
    )

    if not common_ids:
        raise RuntimeError(
            "No matching prediction/GT files."
        )

    total_corr = 0.0
    total_mae = 0.0
    total_rmse = 0.0

    rows = []

    print("=" * 70)
    print("FREQUENCY STRUCTURE EVALUATION")
    print("=" * 70)

    print(
        f"Images evaluated : "
        f"{len(common_ids)}"
    )

    # ========================================================
    # PROCESS
    # ========================================================

    for index, image_id in enumerate(
        common_ids,
        start=1
    ):

        prediction = load_npy_image(
            prediction_files[image_id]
        )

        gt = load_npy_image(
            gt_files[image_id]
        )

        if prediction.shape != gt.shape:
            raise RuntimeError(
                f"Shape mismatch for {image_id}: "
                f"{prediction.shape} vs {gt.shape}"
            )

        corr = frequency_correlation(
            prediction,
            gt
        )

        hf_mae = frequency_mae(
            prediction,
            gt
        )

        hf_rmse = frequency_rmse(
            prediction,
            gt
        )

        total_corr += corr
        total_mae += hf_mae
        total_rmse += hf_rmse

        rows.append({
            "Image": image_id,
            "Frequency_Correlation": corr,
            "Frequency_MAE": hf_mae,
            "Frequency_RMSE": hf_rmse
        })

        if index % 500 == 0:
            print(
                f"Processed: "
                f"{index}/{len(common_ids)}"
            )

    # ========================================================
    # AVERAGES
    # ========================================================

    avg_corr = (
        total_corr
        /
        len(common_ids)
    )

    avg_mae = (
        total_mae
        /
        len(common_ids)
    )

    avg_rmse = (
        total_rmse
        /
        len(common_ids)
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FREQUENCY STRUCTURE RESULTS")
    print("=" * 70)

    print(
        f"Images evaluated       : "
        f"{len(common_ids)}"
    )

    print(
        f"Frequency correlation  : "
        f"{avg_corr:.6f}"
    )

    print(
        f"Frequency MAE          : "
        f"{avg_mae:.6f}"
    )

    print(
        f"Frequency RMSE         : "
        f"{avg_rmse:.6f}"
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
                "Frequency_Correlation",
                "Frequency_MAE",
                "Frequency_RMSE"
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
# BICUBIC EVALUATION
# ============================================================

def evaluate_bicubic(
    lr_dir,
    gt_dir,
    subset_dir,
    output_csv
):

    lr_dir = Path(lr_dir)
    gt_dir = Path(gt_dir)
    subset_dir = Path(subset_dir)
    output_csv = Path(output_csv)

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
        set(subset_files)
        &
        set(lr_files)
        &
        set(gt_files)
    )

    if not common_ids:
        raise RuntimeError(
            "No matching LR/GT files."
        )

    total_corr = 0.0
    total_mae = 0.0
    total_rmse = 0.0

    rows = []

    print("=" * 70)
    print("BICUBIC FREQUENCY STRUCTURE EVALUATION")
    print("=" * 70)

    print(
        f"Images evaluated : "
        f"{len(common_ids)}"
    )

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

        bicubic = bicubic_upscale(
            lr,
            gt.shape
        )

        corr = frequency_correlation(
            bicubic,
            gt
        )

        hf_mae = frequency_mae(
            bicubic,
            gt
        )

        hf_rmse = frequency_rmse(
            bicubic,
            gt
        )

        total_corr += corr
        total_mae += hf_mae
        total_rmse += hf_rmse

        rows.append({
            "Image": image_id,
            "Frequency_Correlation": corr,
            "Frequency_MAE": hf_mae,
            "Frequency_RMSE": hf_rmse
        })

        if index % 500 == 0:
            print(
                f"Processed: "
                f"{index}/{len(common_ids)}"
            )

    # ========================================================
    # AVERAGES
    # ========================================================

    avg_corr = (
        total_corr
        /
        len(common_ids)
    )

    avg_mae = (
        total_mae
        /
        len(common_ids)
    )

    avg_rmse = (
        total_rmse
        /
        len(common_ids)
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("BICUBIC FREQUENCY STRUCTURE RESULTS")
    print("=" * 70)

    print(
        f"Images evaluated       : "
        f"{len(common_ids)}"
    )

    print(
        f"Frequency correlation  : "
        f"{avg_corr:.6f}"
    )

    print(
        f"Frequency MAE          : "
        f"{avg_mae:.6f}"
    )

    print(
        f"Frequency RMSE         : "
        f"{avg_rmse:.6f}"
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
                "Frequency_Correlation",
                "Frequency_MAE",
                "Frequency_RMSE"
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

    parser = argparse.ArgumentParser(
        description=(
            "Frequency-structure similarity "
            "evaluation."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "model",
            "bicubic"
        ],
        required=True
    )

    parser.add_argument(
        "--prediction",
        help=(
            "Restored/model output directory "
            "when mode=model."
        )
    )

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
            "Its filenames define the 2469-image subset."
        )
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    if args.mode == "model":

        if args.prediction is None:
            raise RuntimeError(
                "--prediction is required "
                "when mode=model."
            )

        evaluate_prediction(
            prediction_dir=args.prediction,
            gt_dir=args.gt,
            output_csv=args.output
        )

    else:

        evaluate_bicubic(
            lr_dir=args.lr,
            gt_dir=args.gt,
            subset_dir=args.subset,
            output_csv=args.output
        )