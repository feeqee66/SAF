import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity


ROOT = Path(r"C:\Users\apurva vivobook\Downloads\semic\SAF")

GT_DIR = Path(r"C:\Users\apurva vivobook\Downloads\train\train\GT")
LR_DIR = Path(r"C:\Users\apurva vivobook\Downloads\train\train\NoisyLR")

OUTPUT = ROOT / "results" / "advanced_metrics.csv"


def normalize(image):
    image = image.astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def resize_bicubic(image, target_size=(256, 256)):
    """
    Bicubic upsampling of the actual NoisyLR image.
    """

    from PIL import Image

    image = normalize(image)

    image_uint8 = (
        image * 255.0
    ).astype(np.uint8)

    pil_image = Image.fromarray(image_uint8)

    resized = pil_image.resize(
        target_size,
        Image.Resampling.BICUBIC
    )

    resized = np.asarray(
        resized,
        dtype=np.float32
    ) / 255.0

    return resized


def gradient_magnitude(image):
    """
    Calculate gradient magnitude using Sobel filters.
    """

    gx = sobel(
        image,
        axis=1,
        mode="reflect"
    )

    gy = sobel(
        image,
        axis=0,
        mode="reflect"
    )

    magnitude = np.sqrt(
        gx ** 2 + gy ** 2
    )

    return magnitude


def edge_mae(pred, gt):
    """
    Difference between predicted and GT edge maps.
    Lower is better.
    """

    pred_edges = gradient_magnitude(pred)
    gt_edges = gradient_magnitude(gt)

    return float(
        np.mean(
            np.abs(
                pred_edges - gt_edges
            )
        )
    )


def edge_correlation(pred, gt):
    """
    Correlation between predicted and GT edge maps.
    Higher is better.
    """

    pred_edges = gradient_magnitude(pred)
    gt_edges = gradient_magnitude(gt)

    correlation = np.corrcoef(
        pred_edges.flatten(),
        gt_edges.flatten()
    )[0, 1]

    return float(correlation)


def high_frequency_energy(image):
    """
    Estimate high-frequency energy using
    the Laplacian operator.
    """

    from scipy.ndimage import laplace

    high_freq = laplace(image)

    energy = np.mean(
        high_freq ** 2
    )

    return float(energy)


def analyze_pair(filename):

    gt = np.load(
        GT_DIR / filename
    )

    lr = np.load(
        LR_DIR / filename
    )

    gt = normalize(gt)
    lr = normalize(lr)

    prediction = resize_bicubic(
        lr,
        target_size=gt.shape[::-1]
    )

    # ------------------------------------------------
    # EDGE METRICS
    # ------------------------------------------------

    edge_error = edge_mae(
        prediction,
        gt
    )

    edge_corr = edge_correlation(
        prediction,
        gt
    )

    # ------------------------------------------------
    # HIGH FREQUENCY
    # ------------------------------------------------

    gt_hf = high_frequency_energy(
        gt
    )

    pred_hf = high_frequency_energy(
        prediction
    )

    hf_ratio = (
        pred_hf / gt_hf
        if gt_hf > 1e-12
        else 0.0
    )

    # ------------------------------------------------
    # SSIM
    # ------------------------------------------------

    ssim = structural_similarity(
        gt,
        prediction,
        data_range=1.0
    )

    # ------------------------------------------------
    # BASIC ERROR
    # ------------------------------------------------

    mae = np.mean(
        np.abs(
            gt - prediction
        )
    )

    mse = np.mean(
        (gt - prediction) ** 2
    )

    return {

        "image": filename,

        "mae": float(mae),

        "mse": float(mse),

        "ssim": float(ssim),

        "edge_mae": float(
            edge_error
        ),

        "edge_correlation": float(
            edge_corr
        ),

        "gt_high_frequency_energy": float(
            gt_hf
        ),

        "pred_high_frequency_energy": float(
            pred_hf
        ),

        "high_frequency_ratio": float(
            hf_ratio
        )
    }


def main():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    gt_files = sorted(
        GT_DIR.glob("*.npy")
    )

    print(
        "Number of GT images:",
        len(gt_files)
    )

    results = []

    for i, gt_file in enumerate(gt_files):

        filename = gt_file.name

        lr_file = LR_DIR / filename

        if not lr_file.exists():
            print(
                "Skipping missing:",
                filename
            )
            continue

        result = analyze_pair(
            filename
        )

        results.append(result)

        if (i + 1) % 100 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(gt_files)}"
            )

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        OUTPUT,
        index=False
    )

    print()
    print("==============================")
    print("ADVANCED METRICS COMPLETE")
    print("==============================")

    print(
        "Images:",
        len(df)
    )

    print()
    print("Average MAE:")
    print(
        df["mae"].mean()
    )

    print()
    print("Average MSE:")
    print(
        df["mse"].mean()
    )

    print()
    print("Average SSIM:")
    print(
        df["ssim"].mean()
    )

    print()
    print("Average Edge MAE:")
    print(
        df["edge_mae"].mean()
    )

    print()
    print("Average Edge Correlation:")
    print(
        df["edge_correlation"].mean()
    )

    print()
    print("Average High Frequency Ratio:")
    print(
        df["high_frequency_ratio"].mean()
    )

    print()
    print("Saved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()