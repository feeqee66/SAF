"""
Model-agnostic edge-based evaluation metrics.

Used to evaluate whether a restored image preserves
the structural edges present in the ground-truth image.

Input:
    pred : restored image
    gt   : ground-truth image

Expected:
    2D grayscale NumPy arrays
    approximately in [0, 1]
"""

import numpy as np
from scipy.ndimage import sobel


# ============================================================
# INPUT VALIDATION
# ============================================================

def _prepare_image(image):
    """
    Convert image to float32 2D array in [0, 1].
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, "
            f"got shape {image.shape}"
        )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image


# ============================================================
# EDGE EXTRACTION
# ============================================================

def extract_edges(image):
    """
    Extract Sobel edge magnitude from an image.

    Parameters
    ----------
    image : numpy.ndarray
        2D grayscale image in [0, 1].

    Returns
    -------
    numpy.ndarray
        Normalized edge magnitude map.
    """

    image = _prepare_image(image)

    # Horizontal gradient
    gx = sobel(
        image,
        axis=1,
        mode="reflect"
    )

    # Vertical gradient
    gy = sobel(
        image,
        axis=0,
        mode="reflect"
    )

    # Gradient magnitude
    magnitude = np.sqrt(
        gx ** 2 +
        gy ** 2
    )

    # Normalize for stable comparison
    max_value = magnitude.max()

    if max_value > 0:
        magnitude = magnitude / max_value

    return magnitude.astype(
        np.float32
    )


# ============================================================
# EDGE ERROR
# ============================================================

def edge_error(pred, gt):
    """
    Calculate mean absolute difference between
    predicted and ground-truth edge maps.

    Lower is better.

    Returns
    -------
    float
        Edge error.
    """

    pred = _prepare_image(pred)
    gt = _prepare_image(gt)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"pred={pred.shape}, gt={gt.shape}"
        )

    pred_edges = extract_edges(
        pred
    )

    gt_edges = extract_edges(
        gt
    )

    error = np.mean(
        np.abs(
            pred_edges -
            gt_edges
        )
    )

    return float(error)


# ============================================================
# EDGE PRESERVATION
# ============================================================

def edge_preservation(pred, gt):
    """
    Calculate normalized edge preservation.

    A value close to 1 means strong structural
    edge preservation.

    Higher is better.

    Returns
    -------
    float
        Edge preservation score.
    """

    pred = _prepare_image(pred)
    gt = _prepare_image(gt)

    if pred.shape != gt.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"pred={pred.shape}, gt={gt.shape}"
        )

    pred_edges = extract_edges(
        pred
    )

    gt_edges = extract_edges(
        gt
    )

    difference = np.mean(
        np.abs(
            pred_edges -
            gt_edges
        )
    )

    # Convert error to preservation score
    preservation = 1.0 - difference

    # Keep score in [0, 1]
    preservation = np.clip(
        preservation,
        0.0,
        1.0
    )

    return float(
        preservation
    )


# ============================================================
# ALL EDGE METRICS
# ============================================================

def calculate_edge_metrics(pred, gt):
    """
    Calculate all edge-based metrics.

    Parameters
    ----------
    pred : numpy.ndarray
        Restored image.

    gt : numpy.ndarray
        Ground-truth image.

    Returns
    -------
    dict
        Edge evaluation results.
    """

    return {
        "Edge_Error": edge_error(
            pred,
            gt
        ),

        "Edge_Preservation": edge_preservation(
            pred,
            gt
        )
    }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EDGE METRICS TEST")
    print("=" * 60)

    # Ground truth
    gt = np.zeros(
        (64, 64),
        dtype=np.float32
    )

    gt[20:45, 20:45] = 1.0

    # Slightly different prediction
    pred = gt.copy()

    pred[22:43, 22:43] = 1.0

    results = calculate_edge_metrics(
        pred,
        gt
    )

    print(
        f"Edge Error       : "
        f"{results['Edge_Error']:.6f}"
    )

    print(
        f"Edge Preservation: "
        f"{results['Edge_Preservation']:.6f}"
    )

    print("=" * 60)