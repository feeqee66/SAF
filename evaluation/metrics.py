import numpy as np

from skimage.metrics import structural_similarity


def mae(pred, gt):
    """
    Mean Absolute Error.
    Lower is better.
    """

    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    return float(
        np.mean(np.abs(pred - gt))
    )


def mse(pred, gt):
    """
    Mean Squared Error.
    Lower is better.
    """

    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    return float(
        np.mean((pred - gt) ** 2)
    )


def psnr(pred, gt, data_range=1.0):
    """
    Peak Signal-to-Noise Ratio.
    Higher is better.
    """

    error = mse(pred, gt)

    if error == 0:
        return float("inf")

    return float(
        10.0 * np.log10(
            (data_range ** 2) / error
        )
    )


def ssim(pred, gt):
    """
    Structural Similarity Index.
    Higher is better.
    """

    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)

    return float(
        structural_similarity(
            pred,
            gt,
            data_range=1.0
        )
    )


def calculate_metrics(pred, gt):
    """
    Calculate all basic image-quality metrics.
    """

    return {
        "MAE": mae(pred, gt),
        "MSE": mse(pred, gt),
        "PSNR": psnr(pred, gt),
        "SSIM": ssim(pred, gt)
    }