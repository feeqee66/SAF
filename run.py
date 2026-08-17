from pathlib import Path
import sys

import numpy as np
import torch

from model.cnn_baseline import SAFBaseline


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "cnn_l1_best.pt"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(device):

    model = SAFBaseline().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def restore_image(
    model,
    input_path,
    output_path,
    device
):

    image = np.load(
        input_path
    ).astype(np.float32)

    # Expected input: H x W
    if image.ndim == 3:
        if image.shape[-1] == 1:
            image = image[..., 0]
        elif image.shape[0] == 1:
            image = image[0]
        else:
            raise ValueError(
                f"Unsupported input shape: {image.shape}"
            )

    if image.ndim != 2:
        raise ValueError(
            f"Expected 2D grayscale array, got {image.shape}"
        )

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(
        device
    )

    with torch.no_grad():

        restored = model(
            tensor
        )

        restored = torch.clamp(
            restored,
            0.0,
            1.0
        )

    restored = (
        restored
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Final safety checks
    if not np.isfinite(
        restored
    ).all():

        raise ValueError(
            f"Model produced NaN/Inf for "
            f"{input_path.name}"
        )

    if restored.shape != (256, 256):

        raise ValueError(
            f"Unexpected output shape "
            f"{restored.shape} for "
            f"{input_path.name}"
        )

    np.save(
        output_path,
        restored
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 3:

        print(
            "Usage: python run.py "
            "<input-dir> <output-dir>"
        )

        sys.exit(1)

    input_dir = Path(
        sys.argv[1]
    )

    output_dir = Path(
        sys.argv[2]
    )

    if not input_dir.exists():

        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found: "
            f"{MODEL_PATH}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    model = load_model(
        device
    )

    input_files = sorted(
        input_dir.glob("*.npy"),
        key=lambda p: p.name
    )

    if not input_files:

        raise RuntimeError(
            f"No .npy files found in "
            f"{input_dir}"
        )

    print(
        "Input files:",
        len(input_files)
    )

    for index, input_path in enumerate(
        input_files,
        start=1
    ):

        output_path = (
            output_dir
            / input_path.name
        )

        restore_image(
            model,
            input_path,
            output_path,
            device
        )

        if (
            index <= 5
            or index % 100 == 0
            or index == len(input_files)
        ):

            print(
                f"Processed: "
                f"{index}/{len(input_files)}"
            )

    print()
    print(
        "Inference complete."
    )

    print(
        "Outputs:",
        len(input_files)
    )

    print(
        "Output directory:",
        output_dir
    )


if __name__ == "__main__":
    main()