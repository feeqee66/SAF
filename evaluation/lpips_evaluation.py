from pathlib import Path
import argparse
import csv

import numpy as np
import torch
import lpips
from PIL import Image


# ============================================================
# LOAD NPY
# ============================================================

def load_npy(path):
    image = np.load(path).astype(np.float32)
    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D image, got {image.shape}"
        )

    return np.clip(image, 0.0, 1.0)


# ============================================================
# NUMPY -> LPIPS TENSOR
# ============================================================

def to_tensor(image):
    """
    LPIPS expects:
        [N, 3, H, W]
    with values in [-1, 1].

    SAF images are grayscale, so we replicate
    grayscale into 3 channels.
    """

    tensor = torch.from_numpy(
        image
    ).float()

    tensor = tensor.unsqueeze(0)
    tensor = tensor.repeat(3, 1, 1)

    tensor = tensor.unsqueeze(0)

    tensor = tensor * 2.0 - 1.0

    return tensor


# ============================================================
# BICUBIC
# ============================================================

def bicubic_upscale(image, target_shape):

    target_h, target_w = target_shape

    image_uint8 = (
        image * 255.0
    ).astype(np.uint8)

    pil_image = Image.fromarray(
        image_uint8,
        mode="L"
    )

    resized = pil_image.resize(
        (target_w, target_h),
        resample=Image.Resampling.BICUBIC
    )

    return (
        np.asarray(
            resized,
            dtype=np.float32
        )
        / 255.0
    )


# ============================================================
# GET COMMON FILES
# ============================================================

def get_common_ids(
    prediction_dir,
    gt_dir
):

    prediction_dir = Path(
        prediction_dir
    )

    gt_dir = Path(
        gt_dir
    )

    prediction_files = {
        p.stem: p
        for p in prediction_dir.glob("*.npy")
    }

    gt_files = {
        p.stem: p
        for p in gt_dir.glob("*.npy")
    }

    common = sorted(
        set(prediction_files)
        &
        set(gt_files)
    )

    return (
        prediction_files,
        gt_files,
        common
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    prediction_dir,
    gt_dir,
    output_csv
):

    prediction_files, gt_files, common = (
        get_common_ids(
            prediction_dir,
            gt_dir
        )
    )

    if not common:
        raise RuntimeError(
            "No matching files."
        )

    total = 0.0
    rows = []

    print("=" * 70)
    print("LPIPS MODEL EVALUATION")
    print("=" * 70)

    print(
        f"Images evaluated : {len(common)}"
    )

    with torch.no_grad():

        for index, image_id in enumerate(
            common,
            start=1
        ):

            pred = load_npy(
                prediction_files[image_id]
            )

            gt = load_npy(
                gt_files[image_id]
            )

            if pred.shape != gt.shape:
                raise RuntimeError(
                    f"Shape mismatch: "
                    f"{image_id}"
                )

            pred_tensor = to_tensor(
                pred
            )

            gt_tensor = to_tensor(
                gt
            )

            value = model(
                pred_tensor,
                gt_tensor
            )

            value = float(
                value.item()
            )

            total += value

            rows.append({
                "Image": image_id,
                "LPIPS": value
            })

            if index % 500 == 0:
                print(
                    f"Processed: "
                    f"{index}/{len(common)}"
                )

    average = (
        total / len(common)
    )

    print()
    print("=" * 70)
    print("MODEL LPIPS RESULT")
    print("=" * 70)

    print(
        f"LPIPS : {average:.6f}"
    )

    print(
        "Lower is better."
    )

    print("=" * 70)

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
                "LPIPS"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Results saved to:\n"
        f"{output_csv}"
    )


# ============================================================
# EVALUATE BICUBIC
# ============================================================

def evaluate_bicubic(
    model,
    lr_dir,
    gt_dir,
    subset_dir,
    output_csv
):

    lr_dir = Path(lr_dir)
    gt_dir = Path(gt_dir)
    subset_dir = Path(subset_dir)

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

    common = sorted(
        set(subset_files)
        &
        set(lr_files)
        &
        set(gt_files)
    )

    if not common:
        raise RuntimeError(
            "No matching files."
        )

    total = 0.0
    rows = []

    print("=" * 70)
    print("LPIPS BICUBIC EVALUATION")
    print("=" * 70)

    print(
        f"Images evaluated : {len(common)}"
    )

    with torch.no_grad():

        for index, image_id in enumerate(
            common,
            start=1
        ):

            lr = load_npy(
                lr_files[image_id]
            )

            gt = load_npy(
                gt_files[image_id]
            )

            bicubic = bicubic_upscale(
                lr,
                gt.shape
            )

            bicubic_tensor = to_tensor(
                bicubic
            )

            gt_tensor = to_tensor(
                gt
            )

            value = model(
                bicubic_tensor,
                gt_tensor
            )

            value = float(
                value.item()
            )

            total += value

            rows.append({
                "Image": image_id,
                "LPIPS": value
            })

            if index % 500 == 0:
                print(
                    f"Processed: "
                    f"{index}/{len(common)}"
                )

    average = (
        total / len(common)
    )

    print()
    print("=" * 70)
    print("BICUBIC LPIPS RESULT")
    print("=" * 70)

    print(
        f"LPIPS : {average:.6f}"
    )

    print(
        "Lower is better."
    )

    print("=" * 70)

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
                "LPIPS"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"Results saved to:\n"
        f"{output_csv}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "model",
            "bicubic"
        ],
        required=True
    )

    parser.add_argument(
        "--prediction"
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
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    print(
        "Loading LPIPS AlexNet model..."
    )

    loss_fn = lpips.LPIPS(
        net="alex"
    )

    loss_fn = loss_fn.cpu()

    if args.mode == "model":

        if not args.prediction:
            raise RuntimeError(
                "--prediction is required."
            )

        evaluate_model(
            model=loss_fn,
            prediction_dir=args.prediction,
            gt_dir=args.gt,
            output_csv=Path(args.output)
        )

    else:

        evaluate_bicubic(
            model=loss_fn,
            lr_dir=args.lr,
            gt_dir=args.gt,
            subset_dir=args.subset,
            output_csv=Path(args.output)
        )