import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# SAF VISUAL ANALYSIS
# ============================================================
#
# Uses the already generated deployment_results.
#
# Generates:
#
#   1. Representative visual comparisons
#      Input / Restored / Ground Truth / Error Heatmap
#
#   2. Spatial restoration error maps
#
#   3. Highest-error region crops
#
#   4. Visual-analysis CSV
#
# IMPORTANT:
# The error heatmap is NOT a confidence map.
#
# It represents:
#
#       E(x,y) = |Restored(x,y) - GT(x,y)|
#
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

# evaluation/visual_analysis.py
# Parent of evaluation = SAF
# evaluation/visual_analysis.py
BASE_DIR = Path(__file__).resolve().parent.parent

# Directly under SAF
DEPLOYMENT_DIR = BASE_DIR / "deployment_results"

# Restored .npy files are directly here
RESTORED_DIR = DEPLOYMENT_DIR

# Visual analysis outputs
VISUAL_DIR = DEPLOYMENT_DIR / "visual_analysis"

HEATMAP_DIR = VISUAL_DIR / "error_heatmaps"
# ============================================================
# LOAD IMAGE


# ============================================================

def load_image(path):

    image = np.load(path).astype(np.float32)

    image = np.squeeze(image)

    if image.ndim != 2:

        raise RuntimeError(
            f"Expected grayscale 2D image:\n"
            f"{path}\n"
            f"Shape = {image.shape}"
        )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image


# ============================================================
# RESIZE INPUT FOR DISPLAY
# ============================================================

def resize_for_display(
    image,
    target_shape
):

    target_h, target_w = target_shape

    h, w = image.shape

    # --------------------------------------------------------
    # Normal SAF case: 128 -> 256
    # --------------------------------------------------------

    if (
        target_h % h == 0
        and target_w % w == 0
    ):

        sy = target_h // h
        sx = target_w // w

        return np.repeat(
            np.repeat(
                image,
                sy,
                axis=0
            ),
            sx,
            axis=1
        )

    # --------------------------------------------------------
    # Generic nearest-neighbour
    # --------------------------------------------------------

    y = np.linspace(
        0,
        h - 1,
        target_h
    ).astype(int)

    x = np.linspace(
        0,
        w - 1,
        target_w
    ).astype(int)

    return image[
        np.ix_(y, x)
    ]


# ============================================================
# FIND HIGHEST ERROR REGION
# ============================================================

def find_high_error_region(
    error,
    crop_size=64
):

    h, w = error.shape

    crop_size = min(
        crop_size,
        h,
        w
    )

    # --------------------------------------------------------
    # Use sliding local average through integral image
    # --------------------------------------------------------

    integral = np.pad(
        error,
        (
            (1, 0),
            (1, 0)
        ),
        mode="constant"
    ).cumsum(
        axis=0
    ).cumsum(
        axis=1
    )

    best_value = -1
    best_y = 0
    best_x = 0

    step = max(
        1,
        crop_size // 4
    )

    for y in range(
        0,
        h - crop_size + 1,
        step
    ):

        y2 = y + crop_size

        for x in range(
            0,
            w - crop_size + 1,
            step
        ):

            x2 = x + crop_size

            total = (
                integral[y2, x2]
                - integral[y, x2]
                - integral[y2, x]
                + integral[y, x]
            )

            mean_error = (
                total
                /
                (crop_size * crop_size)
            )

            if mean_error > best_value:

                best_value = mean_error
                best_y = y
                best_x = x

    return (
        best_y,
        best_y + crop_size,
        best_x,
        best_x + crop_size
    )


# ============================================================
# CREATE VISUALIZATION
# ============================================================

def create_visualization(
    input_image,
    restored,
    gt,
    error,
    output_path,
    filename
):

    input_display = resize_for_display(
        input_image,
        restored.shape
    )

    # --------------------------------------------------------
    # Error scale
    # --------------------------------------------------------

    error_scale = max(
        float(np.percentile(error, 99)),
        1e-6
    )

    # --------------------------------------------------------
    # Highest-error crop
    # --------------------------------------------------------

    (
        y1,
        y2,
        x1,
        x2
    ) = find_high_error_region(
        error,
        crop_size=64
    )

    mean_error = float(
        np.mean(error)
    )

    max_error = float(
        np.max(error)
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    # ========================================================
    # TOP ROW
    # ========================================================

    axes[0, 0].imshow(
        input_display,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0, 0].set_title(
        "Input / NoisyLR\n128×128 → display"
    )

    axes[0, 0].axis("off")

    # --------------------------------------------------------

    axes[0, 1].imshow(
        restored,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0, 1].set_title(
        "CNN Restored\n256×256"
    )

    axes[0, 1].axis("off")

    # --------------------------------------------------------

    axes[0, 2].imshow(
        gt,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[0, 2].set_title(
        "Ground Truth\n256×256"
    )

    axes[0, 2].axis("off")

    # ========================================================
    # BOTTOM ROW
    # ========================================================

    heatmap = axes[1, 0].imshow(
        error,
        cmap="hot",
        vmin=0,
        vmax=error_scale
    )

    axes[1, 0].set_title(
        "Spatial Restoration Error\n"
        "|Restored − GT|"
    )

    axes[1, 0].axis("off")

    fig.colorbar(
        heatmap,
        ax=axes[1, 0],
        fraction=0.046,
        pad=0.04
    )

    # --------------------------------------------------------
    # Restored highest-error region
    # --------------------------------------------------------

    axes[1, 1].imshow(
        restored[
            y1:y2,
            x1:x2
        ],
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[1, 1].set_title(
        "Highest-Error Region\nRestored"
    )

    axes[1, 1].axis("off")

    # --------------------------------------------------------
    # Error highest-error region
    # --------------------------------------------------------

    axes[1, 2].imshow(
        error[
            y1:y2,
            x1:x2
        ],
        cmap="hot",
        vmin=0,
        vmax=error_scale
    )

    axes[1, 2].set_title(
        "Highest-Error Region\nError"
    )

    axes[1, 2].axis("off")

    # ========================================================
    # TITLE
    # ========================================================

    fig.suptitle(
        (
            f"SAF Spatial Restoration Analysis — {filename}\n"
            f"Mean Error = {mean_error:.6f}    "
            f"Maximum Error = {max_error:.6f}"
        ),
        fontsize=14
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# CREATE SIMPLE VISUAL GALLERY
# ============================================================

def create_gallery(
    samples,
    input_map,
    restored_map,
    gt_map,
    output_path
):

    fig, axes = plt.subplots(
        len(samples),
        3,
        figsize=(12, 4 * len(samples))
    )

    if len(samples) == 1:

        axes = np.expand_dims(
            axes,
            axis=0
        )

    for row, name in enumerate(samples):

        input_img = resize_for_display(
            input_map[name],
            restored_map[name].shape
        )

        axes[row, 0].imshow(
            input_img,
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row, 0].set_title(
            f"{name} — Input"
        )

        axes[row, 0].axis("off")

        # ----------------------------------------------------

        axes[row, 1].imshow(
            restored_map[name],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row, 1].set_title(
            "Restored"
        )

        axes[row, 1].axis("off")

        # ----------------------------------------------------

        axes[row, 2].imshow(
            gt_map[name],
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row, 2].set_title(
            "Ground Truth"
        )

        axes[row, 2].axis("off")

    fig.suptitle(
        "SAF Representative Restoration Results",
        fontsize=16
    )

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SAF VISUAL ANALYSIS")
    print("=" * 70)

    # ========================================================
    # CHECK DEPLOYMENT RESULTS
    # ========================================================

    if not DEPLOYMENT_DIR.exists():

        raise RuntimeError(
            "deployment_results directory not found:\n"
            f"{DEPLOYMENT_DIR}\n\n"
            "Run python run.py first."
        )

    if not RESTORED_DIR.exists():

        raise RuntimeError(
            "Restored-image directory not found:\n"
            f"{RESTORED_DIR}\n\n"
            "Run python run.py first."
        )

    # ========================================================
    # ASK USER FOR VALIDATION DATASET
    # ========================================================
    #
    # Restored images are automatically taken from:
    #
    #     SAF/deployment_results/*.npy
    #
    # The user only provides the validation dataset root.
    # It must contain:
    #
    #     NoisyLR/
    #     GT/
    #
    # ========================================================

    print()
    print("Deployment results:")
    print(DEPLOYMENT_DIR)

    print()
    print("=" * 70)
    print("VALIDATION DATASET")
    print("=" * 70)

    validation_root = Path(
        input(
            "Enter validation dataset folder: "
        ).strip().strip('"')
    )

    if not validation_root.exists():
        raise RuntimeError(
            "Validation dataset folder not found:\n"
            f"{validation_root}"
        )

    if not validation_root.is_dir():
        raise RuntimeError(
            "The supplied validation dataset path is not a folder:\n"
            f"{validation_root}"
        )

    input_dir = validation_root / "NoisyLR"
    gt_dir = validation_root / "GT"

    # Common naming variants.
    if not input_dir.is_dir():
        for name in ["noisylr", "LR", "lr", "Input", "input"]:
            candidate = validation_root / name
            if candidate.is_dir():
                input_dir = candidate
                break

    if not gt_dir.is_dir():
        for name in ["gt", "GroundTruth", "ground_truth", "groundtruth"]:
            candidate = validation_root / name
            if candidate.is_dir():
                gt_dir = candidate
                break

    print()
    print("Input folder:")
    print(input_dir if input_dir.is_dir() else "NOT FOUND")

    print()
    print("Ground-truth folder:")
    print(gt_dir if gt_dir.is_dir() else "NOT FOUND")

    if not input_dir.is_dir():
        raise RuntimeError(
            "NoisyLR/Input folder not found inside the validation dataset:\n"
            f"{validation_root}\n\n"
            "Expected a folder such as:\n"
            f"{validation_root / 'NoisyLR'}"
        )

    if not gt_dir.is_dir():
        raise RuntimeError(
            "GT folder not found inside the validation dataset:\n"
            f"{validation_root}\n\n"
            "Expected a folder such as:\n"
            f"{validation_root / 'GT'}"
        )

    # ========================================================
    # FIND FILES
    # ========================================================

    restored_files = sorted(
        RESTORED_DIR.glob("*.npy")
    )

    input_files = sorted(
        input_dir.glob("*.npy")
    ) if input_dir else []

    gt_files = sorted(
        gt_dir.glob("*.npy")
    )

    print()
    print("=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)

    print(
        f"Restored images : {len(restored_files)}"
    )

    print(
        f"Input images    : {len(input_files)}"
    )

    print(
        f"GT images       : {len(gt_files)}"
    )

    # ========================================================
    # MAP FILES
    # ========================================================

    restored_map = {
        f.stem: f
        for f in restored_files
    }

    gt_map = {
        f.stem: f
        for f in gt_files
    }

    input_map_paths = {
        f.stem: f
        for f in input_files
    }

    common = sorted(
        set(restored_map)
        &
        set(gt_map)
        &
        set(input_map_paths)
    )

    if not common:

        raise RuntimeError(
            "No common Input + Restored + GT filenames found."
        )

    print(
        f"Complete visual pairs : {len(common)}"
    )

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    VISUAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HEATMAP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SELECT 3 REPRESENTATIVE SAMPLES
    # ========================================================

    if len(common) <= 3:

        samples = common

    else:

        indices = np.linspace(
            0,
            len(common) - 1,
            3
        ).astype(int)

        samples = [
            common[i]
            for i in indices
        ]

    print()
    print(
        "Representative samples:"
    )

    for name in samples:

        print(
            f"  {name}.npy"
        )

    # ========================================================
    # CSV
    # ========================================================

    csv_path = (
        VISUAL_DIR
        /
        "spatial_error_analysis.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow([
            "filename",
            "mean_error",
            "max_error",
            "std_error",
            "pixels_error_gt_0.05_percent",
            "pixels_error_gt_0.10_percent"
        ])

        # ====================================================
        # PROCESS SAMPLES
        # ====================================================

        for index, name in enumerate(
            samples,
            start=1
        ):

            print()
            print(
                f"Processing "
                f"{index}/{len(samples)}: "
                f"{name}.npy"
            )

            input_img = load_image(
                input_map_paths[name]
            )

            restored_img = load_image(
                restored_map[name]
            )

            gt_img = load_image(
                gt_map[name]
            )

            if restored_img.shape != gt_img.shape:

                raise RuntimeError(
                    f"Shape mismatch:\n"
                    f"{name}\n"
                    f"Restored = {restored_img.shape}\n"
                    f"GT       = {gt_img.shape}"
                )

            # ----------------------------------------------
            # Error
            # ----------------------------------------------

            error = np.abs(
                restored_img - gt_img
            )

            mean_error = float(
                np.mean(error)
            )

            max_error = float(
                np.max(error)
            )

            std_error = float(
                np.std(error)
            )

            error_005 = float(
                np.mean(
                    error > 0.05
                )
                *
                100
            )

            error_010 = float(
                np.mean(
                    error > 0.10
                )
                *
                100
            )

            writer.writerow([
                name,
                f"{mean_error:.8f}",
                f"{max_error:.8f}",
                f"{std_error:.8f}",
                f"{error_005:.4f}",
                f"{error_010:.4f}"
            ])

            # ----------------------------------------------
            # Heatmap
            # ----------------------------------------------

            heatmap_path = (
                HEATMAP_DIR
                /
                f"{name}_error_heatmap.png"
            )

            create_visualization(
                input_image=input_img,
                restored=restored_img,
                gt=gt_img,
                error=error,
                output_path=heatmap_path,
                filename=name
            )

            print(
                f"  Mean error : {mean_error:.6f}"
            )

            print(
                f"  Max error  : {max_error:.6f}"
            )

            print(
                f"  Saved      : {heatmap_path}"
            )

    # ========================================================
    # CREATE GALLERY
    # ========================================================

    gallery_input = {
        name: load_image(
            input_map_paths[name]
        )
        for name in samples
    }

    gallery_restored = {
        name: load_image(
            restored_map[name]
        )
        for name in samples
    }

    gallery_gt = {
        name: load_image(
            gt_map[name]
        )
        for name in samples
    }

    gallery_path = (
        VISUAL_DIR
        /
        "representative_gallery.png"
    )

    create_gallery(
        samples=samples,
        input_map=gallery_input,
        restored_map=gallery_restored,
        gt_map=gallery_gt,
        output_path=gallery_path
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_path = (
        VISUAL_DIR
        /
        "visual_analysis_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SAF VISUAL ANALYSIS SUMMARY\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )

        f.write(
            f"Restored images : "
            f"{len(restored_files)}\n"
        )

        f.write(
            f"GT images       : "
            f"{len(gt_files)}\n"
        )

        f.write(
            f"Complete pairs  : "
            f"{len(common)}\n\n"
        )

        f.write(
            "Representative samples:\n"
        )

        for name in samples:

            f.write(
                f"  {name}.npy\n"
            )

        f.write(
            "\nGenerated visualizations:\n"
        )

        f.write(
            "  - Representative restoration gallery\n"
        )

        f.write(
            "  - Spatial restoration error heatmaps\n"
        )

        f.write(
            "  - Highest-error region crops\n"
        )

        f.write(
            "\n"
        )

        f.write(
            "Error definition:\n"
        )

        f.write(
            "E(x,y) = |Restored(x,y) - GT(x,y)|\n"
        )

        f.write(
            "\nIMPORTANT:\n"
        )

        f.write(
            "The heatmap is a spatial reconstruction "
            "error visualization.\n"
        )

        f.write(
            "It is NOT a calibrated model confidence map.\n"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print("SAF VISUAL ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        "Representative gallery:"
    )

    print(
        gallery_path
    )

    print()
    print(
        "Error heatmaps:"
    )

    print(
        HEATMAP_DIR
    )

    print()
    print(
        "CSV:"
    )

    print(
        csv_path
    )

    print()
    print(
        "Summary:"
    )

    print(
        summary_path
    )

    print()
    print(
        "NOTE:"
    )

    print(
        "Heatmaps represent spatial reconstruction error "
        "against GT."
    )

    print(
        "They are not confidence maps."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()