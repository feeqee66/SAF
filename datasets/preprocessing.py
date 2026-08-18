from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(r"C:\Users\apurva vivobook\Downloads\train\train")

GT_DIR = ROOT / "GT"
LR_DIR = ROOT / "NoisyLR"

OUTPUT = ROOT / "degradation_analysis.csv"


def resize_gt_to_lr(gt):
    """
    Create a synthetic 128x128 LR image from
    the 256x256 GT image.

    This is ONLY for degradation analysis.
    It does not modify the original dataset.
    """

    gt_uint8 = np.clip(
        gt * 255.0,
        0,
        255
    ).astype(np.uint8)

    image = Image.fromarray(gt_uint8)

    image = image.resize(
        (128, 128),
        Image.Resampling.BICUBIC
    )

    return np.asarray(
        image,
        dtype=np.float32
    ) / 255.0


def analyze_pair(filename):

    lr = np.load(LR_DIR / filename)
    gt = np.load(GT_DIR / filename)

    synthetic_lr = resize_gt_to_lr(gt)

    residual = lr - synthetic_lr

    # Correlation
    correlation = np.corrcoef(
        lr.flatten(),
        synthetic_lr.flatten()
    )[0, 1]

    signal_residual_corr = np.corrcoef(
        synthetic_lr.flatten(),
        residual.flatten()
    )[0, 1]

    return {

        "image": filename,

        # LR statistics
        "lr_min": float(lr.min()),
        "lr_max": float(lr.max()),
        "lr_mean": float(lr.mean()),
        "lr_std": float(lr.std()),

        # GT statistics
        "gt_min": float(gt.min()),
        "gt_max": float(gt.max()),
        "gt_mean": float(gt.mean()),
        "gt_std": float(gt.std()),

        # Synthetic LR statistics
        "synthetic_lr_mean": float(
            synthetic_lr.mean()
        ),

        "synthetic_lr_std": float(
            synthetic_lr.std()
        ),

        # LR vs synthetic LR
        "lr_synthetic_mae": float(
            np.mean(np.abs(
                lr - synthetic_lr
            ))
        ),

        "lr_synthetic_mse": float(
            np.mean(
                (lr - synthetic_lr) ** 2
            )
        ),

        "lr_synthetic_correlation": float(
            correlation
        ),

        # Residual
        "residual_mean": float(
            residual.mean()
        ),

        "residual_std": float(
            residual.std()
        ),

        "residual_mae": float(
            np.mean(np.abs(residual))
        ),

        "residual_min": float(
            residual.min()
        ),

        "residual_max": float(
            residual.max()
        ),

        "signal_residual_correlation": float(
            signal_residual_corr
        ),

        # Out-of-range LR fraction
        "lr_below_zero_fraction": float(
            np.mean(lr < 0)
        ),

        "lr_above_one_fraction": float(
            np.mean(lr > 1)
        )
    }


def main():

    gt_files = sorted(
        GT_DIR.glob("*.npy")
    )

    lr_files = sorted(
        LR_DIR.glob("*.npy")
    )

    print("GT files :", len(gt_files))
    print("LR files :", len(lr_files))

    if len(gt_files) != len(lr_files):
        raise ValueError(
            "GT and LR file counts do not match."
        )

    results = []

    for i, gt_file in enumerate(gt_files):

        filename = gt_file.name
        lr_file = LR_DIR / filename

        if not lr_file.exists():
            raise FileNotFoundError(
                f"Missing LR pair: {filename}"
            )

        result = analyze_pair(filename)

        results.append(result)

        if (i + 1) % 100 == 0:
            print(
                f"Processed {i + 1}/{len(gt_files)}"
            )

    df = pd.DataFrame(results)

    df.to_csv(
        OUTPUT,
        index=False
    )

    print("\n==============================")
    print("DATASET FORENSICS COMPLETE")
    print("==============================")

    print("Samples:", len(df))

    print("\nCorrelation")
    print(
        df["lr_synthetic_correlation"]
        .describe()
    )

    print("\nResidual STD")
    print(
        df["residual_std"]
        .describe()
    )

    print("\nResidual MAE")
    print(
        df["residual_mae"]
        .describe()
    )

    print("\nSignal-residual correlation")
    print(
        df["signal_residual_correlation"]
        .describe()
    )

    print("\nSaved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()