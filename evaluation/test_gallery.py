from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = Path(
    r"C:\Users\apurva vivobook\Downloads"
    r"\SAF_test_outputs-20260817T134739Z-1-001"
    r"\SAF_test_outputs"
)

OUTPUT_DIR = Path(
    "evaluation/results/test_gallery"
)

# Number of images per gallery
IMAGES_PER_GALLERY = 25

# Display size of each image
IMAGE_SIZE = 256


# ============================================================
# LOAD IMAGE
# ============================================================

def load_npy(path):

    image = np.load(path).astype(
        np.float32
    )

    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D image, got "
            f"{image.shape} for {path.name}"
        )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image


# ============================================================
# CONVERT TO PIL
# ============================================================

def to_pil(image):

    image_uint8 = (
        image * 255.0
    ).astype(np.uint8)

    return Image.fromarray(
        image_uint8,
        mode="L"
    )


# ============================================================
# CREATE GALLERY
# ============================================================

def create_gallery(
    image_paths,
    output_path,
    gallery_number
):

    columns = 5
    rows = math.ceil(
        len(image_paths) / columns
    )

    label_height = 25

    canvas = Image.new(
        "L",
        (
            columns * IMAGE_SIZE,
            rows * (IMAGE_SIZE + label_height)
        ),
        color=255
    )

    draw = ImageDraw.Draw(
        canvas
    )

    for index, image_path in enumerate(
        image_paths
    ):

        image = load_npy(
            image_path
        )

        image = to_pil(
            image
        )

        image = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        )

        row = index // columns
        col = index % columns

        x = col * IMAGE_SIZE
        y = row * (
            IMAGE_SIZE
            +
            label_height
        )

        canvas.paste(
            image,
            (x, y)
        )

        draw.text(
            (
                x + 5,
                y + IMAGE_SIZE + 4
            ),
            image_path.stem,
            fill=0
        )

    canvas.save(
        output_path
    )

    print(
        f"Saved gallery {gallery_number}: "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SAF TEST OUTPUT GALLERY")
    print("=" * 70)

    files = sorted(
        TEST_DIR.glob("*.npy")
    )

    print(
        f"Test images found : "
        f"{len(files)}"
    )

    if len(files) == 0:
        raise RuntimeError(
            f"No .npy files found in:\n"
            f"{TEST_DIR}"
        )

    # --------------------------------------------------------
    # Check images
    # --------------------------------------------------------

    shapes = set()

    minimum = float("inf")
    maximum = float("-inf")

    for path in files:

        image = load_npy(
            path
        )

        shapes.add(
            image.shape
        )

        minimum = min(
            minimum,
            float(image.min())
        )

        maximum = max(
            maximum,
            float(image.max())
        )

    print(
        f"Image shapes      : "
        f"{sorted(shapes)}"
    )

    print(
        f"Global min        : "
        f"{minimum:.6f}"
    )

    print(
        f"Global max        : "
        f"{maximum:.6f}"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create galleries
    # --------------------------------------------------------

    total = len(files)

    gallery_number = 1

    for start in range(
        0,
        total,
        IMAGES_PER_GALLERY
    ):

        subset = files[
            start:
            start + IMAGES_PER_GALLERY
        ]

        output_path = (
            OUTPUT_DIR
            /
            f"test_gallery_{gallery_number:02d}.png"
        )

        create_gallery(
            subset,
            output_path,
            gallery_number
        )

        gallery_number += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST GALLERY COMPLETE")
    print("=" * 70)

    print(
        f"Images processed : "
        f"{total}"
    )

    print(
        f"Galleries created : "
        f"{gallery_number - 1}"
    )

    print()
    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()