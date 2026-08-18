from pathlib import Path
import csv
import time
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import structural_similarity
from visual_analysis import generate_structural_analysis


# ============================================================
# SAF DEPLOYMENT CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

# run.py is inside SAF/evaluation/
# Therefore SAF_ROOT is one level above this file.

SAF_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = SAF_ROOT / "models"

OUTPUT_DIR = SAF_ROOT / "deployment_results"

DEPLOYED_MODELS = sorted(
    MODEL_DIR.glob("*_deployed.pt")
)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

    def forward(self, x):
        return x + self.block(x)


class SAFBaseline(nn.Module):

    def __init__(
        self,
        channels=64,
        num_blocks=8
    ):
        super().__init__()

        self.head = nn.Conv2d(
            1,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        self.refine = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.upsample = nn.Sequential(

            nn.Conv2d(
                channels,
                channels * 4,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv2d(
            channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):

        x = self.head(x)

        residual = self.body(x)

        x = x + residual

        x = self.refine(x)

        x = self.upsample(x)

        x = self.tail(x)

        return x


# ============================================================
# FIND DEPLOYED MODEL
# ============================================================

def find_model():

    if not MODEL_DIR.exists():

        raise RuntimeError(
            f"Model directory not found:\n"
            f"{MODEL_DIR}"
        )

    if len(DEPLOYED_MODELS) == 0:

        raise RuntimeError(
            "\nNo deployed model found.\n\n"
            "Expected a file like:\n"
            "SAF/models/cnn_l1_best_deployed.pt"
        )

    if len(DEPLOYED_MODELS) == 1:

        return DEPLOYED_MODELS[0]

    # Multiple deployed models
    print()
    print("=" * 70)
    print("AVAILABLE DEPLOYED MODELS")
    print("=" * 70)

    for i, model in enumerate(
        DEPLOYED_MODELS,
        start=1
    ):
        print(
            f"{i}. {model.name}"
        )

    while True:

        try:

            choice = int(
                input(
                    "\nSelect model number: "
                )
            )

            if 1 <= choice <= len(
                DEPLOYED_MODELS
            ):

                return DEPLOYED_MODELS[
                    choice - 1
                ]

        except ValueError:
            pass

        print(
            "Invalid selection."
        )


# ============================================================
# LOAD DEPLOYED MODEL
# ============================================================

def load_model(model_path):

    print()
    print("=" * 70)
    print("LOADING DEPLOYED MODEL")
    print("=" * 70)

    print(
        "Model:",
        model_path
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    required = [
        "model_type",
        "model_state_dict",
        "channels",
        "num_residual_blocks",
        "scale_factor"
    ]

    for key in required:

        if key not in checkpoint:

            raise RuntimeError(
                f"Invalid deployed model.\n"
                f"Missing: {key}"
            )

    model = SAFBaseline(
        channels=checkpoint[
            "channels"
        ],
        num_blocks=checkpoint[
            "num_residual_blocks"
        ]
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(DEVICE)

    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Architecture:",
        checkpoint["architecture"]
    )

    print(
        "Parameters:",
        f"{parameter_count:,}"
    )

    print(
        "Scale:",
        f"{checkpoint['scale_factor']}x"
    )

    print(
        "Input:",
        checkpoint.get(
            "input_size",
            "N/A"
        )
    )

    print(
        "Output:",
        checkpoint.get(
            "output_size",
            "N/A"
        )
    )

    print("=" * 70)

    return model


# ============================================================
# LOAD NPY IMAGE
# ============================================================

def load_npy(path):

    image = np.load(
        path
    ).astype(np.float32)

    image = np.squeeze(image)

    if image.ndim != 2:

        raise ValueError(
            f"{path.name}: expected 2D "
            f"image, got {image.shape}"
        )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image


# ============================================================
# RESTORE SINGLE IMAGE
# ============================================================

def restore_image(
    model,
    image
):

    tensor = torch.from_numpy(
        image
    ).float()

    tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        output = model(
            tensor
        )

    output = torch.clamp(
        output,
        0.0,
        1.0
    )

    output = (
        output
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return output


# ============================================================
# METRICS
# ============================================================

def calculate_mse(
    prediction,
    ground_truth
):

    return float(
        np.mean(
            (
                prediction -
                ground_truth
            ) ** 2
        )
    )


def calculate_mae(
    prediction,
    ground_truth
):

    return float(
        np.mean(
            np.abs(
                prediction -
                ground_truth
            )
        )
    )


def calculate_psnr(
    prediction,
    ground_truth
):

    value = calculate_mse(
        prediction,
        ground_truth
    )

    if value == 0:

        return float("inf")

    return float(
        10.0 *
        np.log10(
            1.0 / value
        )
    )


def calculate_ssim(
    prediction,
    ground_truth
):

    return float(
        structural_similarity(
            ground_truth,
            prediction,
            data_range=1.0
        )
    )


# ============================================================
# EDGE METRIC
# ============================================================

def edge_map(image):

    gx = np.diff(
        image,
        axis=1
    )

    gy = np.diff(
        image,
        axis=0
    )

    gx = np.pad(
        gx,
        ((0, 0), (0, 1))
    )

    gy = np.pad(
        gy,
        ((0, 1), (0, 0))
    )

    return np.sqrt(
        gx ** 2 +
        gy ** 2
    )


def calculate_edge_error(
    prediction,
    ground_truth
):

    prediction_edge = edge_map(
        prediction
    )

    ground_truth_edge = edge_map(
        ground_truth
    )

    return float(
        np.mean(
            np.abs(
                prediction_edge -
                ground_truth_edge
            )
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("SAF OFFLINE DEPLOYMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # AUTOMATIC MODEL
    # --------------------------------------------------------

    model_path = find_model()

    model = load_model(
        model_path
    )

    # --------------------------------------------------------
    # ASK INPUT DIRECTORY
    # --------------------------------------------------------

    print()

    if len(sys.argv) != 3:
        print(
            "Usage: python run.py <input-dir> <output-dir>"
        )
        sys.exit(1)

    input_dir = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()

    if not input_dir.exists():

        raise FileNotFoundError(
            f"\nInput folder not found:\n"
            f"{input_dir}"
        )
    # --------------------------------------------------------
    # OPTIONAL GT — NO USER INTERACTION
    # --------------------------------------------------------

    gt_dir = None

    # If a GT directory is supplied through the environment,
    # use it. Otherwise run normally without GT.
    gt_env = os.environ.get("SAF_GT_DIR")

    if gt_env:
        candidate_gt = Path(gt_env).resolve()

        if candidate_gt.exists():
            gt_dir = candidate_gt

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # INPUT FILES
    # --------------------------------------------------------

    input_files = sorted(
        input_dir.glob("*.npy")
    )

    if len(input_files) == 0:

        raise RuntimeError(
            "\nNo .npy files found in:\n"
            f"{input_dir}"
        )

    print()
    print("=" * 70)
    print("INFERENCE")
    print("=" * 70)

    print(
        "Input images:",
        len(input_files)
    )

    print(
        "Output:",
        output_dir
    )

    # --------------------------------------------------------
    # METRIC STORAGE
    # --------------------------------------------------------

    results = []

    psnr_values = []
    ssim_values = []
    mae_values = []
    mse_values = []
    edge_values = []

    evaluated = 0

    # --------------------------------------------------------
    # INFERENCE TIMER
    # --------------------------------------------------------

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    for index, input_path in enumerate(
        input_files,
        start=1
    ):

        image = load_npy(
            input_path
        )

        start = time.perf_counter()

        restored = restore_image(
            model,
            image
        )

        inference_time = (
            time.perf_counter()
            - start
        )

        # ----------------------------------------------------
        # SAVE RESTORED IMAGE
        # ----------------------------------------------------

        output_path = (
            output_dir /
            input_path.name
        )

        np.save(
            output_path,
            restored
        )

        row = {

            "filename":
                input_path.name,

            "input_height":
                image.shape[0],

            "input_width":
                image.shape[1],

            "output_height":
                restored.shape[0],

            "output_width":
                restored.shape[1],

            "inference_time_ms":
                inference_time * 1000.0
        }

        # ----------------------------------------------------
        # GT METRICS
        # ----------------------------------------------------

        if gt_dir is not None:

            gt_path = (
                gt_dir /
                input_path.name
            )

            if gt_path.exists():

                gt = load_npy(
                    gt_path
                )

                if restored.shape != gt.shape:

                    raise RuntimeError(
                        f"\nShape mismatch:\n"
                        f"Image: {input_path.name}\n"
                        f"Restored: "
                        f"{restored.shape}\n"
                        f"GT: {gt.shape}"
                    )

                image_mse = calculate_mse(
                    restored,
                    gt
                )

                image_mae = calculate_mae(
                    restored,
                    gt
                )

                image_psnr = calculate_psnr(
                    restored,
                    gt
                )

                image_ssim = calculate_ssim(
                    restored,
                    gt
                )

                image_edge = calculate_edge_error(
                    restored,
                    gt
                )

                row.update({

                    "PSNR_dB":
                        image_psnr,

                    "SSIM":
                        image_ssim,

                    "MAE":
                        image_mae,

                    "MSE":
                        image_mse,

                    "Edge_Error":
                        image_edge
                })

                psnr_values.append(
                    image_psnr
                )

                ssim_values.append(
                    image_ssim
                )

                mae_values.append(
                    image_mae
                )

                mse_values.append(
                    image_mse
                )

                edge_values.append(
                    image_edge
                )

                evaluated += 1

        results.append(row)

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            index % 100 == 0
            or index == len(input_files)
        ):

            print(
                f"Processed: "
                f"{index}/{len(input_files)}"
            )

    # --------------------------------------------------------
    # TOTAL TIME
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = (
        output_dir /
        "metrics.csv"
    )

    if results:

        fieldnames = list(
            results[0].keys()
        )

        with open(
            csv_path,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                results
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_path = (
        output_dir /
        "summary.txt"
    )

    with open(
        summary_path,
        "w"
    ) as file:

        file.write(
            "SAF OFFLINE DEPLOYMENT RESULTS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Model: {model_path.name}\n"
        )

        file.write(
            f"Input folder: {input_dir}\n"
        )

        file.write(
            f"Images processed: "
            f"{len(input_files)}\n"
        )

        file.write(
            f"Total inference time: "
            f"{total_time:.4f} seconds\n"
        )

        file.write(
            f"Average images/sec: "
            f"{len(input_files) / total_time:.4f}\n"
        )

        if evaluated > 0:

            file.write(
                f"\nEvaluated pairs: "
                f"{evaluated}\n\n"
            )

            file.write(
                f"PSNR: "
                f"{np.mean(psnr_values):.4f} dB\n"
            )

            file.write(
                f"SSIM: "
                f"{np.mean(ssim_values):.4f}\n"
            )

            file.write(
                f"MAE: "
                f"{np.mean(mae_values):.6f}\n"
            )

            file.write(
                f"MSE: "
                f"{np.mean(mse_values):.6f}\n"
            )

            file.write(
                f"Edge Error: "
                f"{np.mean(edge_values):.6f}\n"
            )

        else:

            file.write(
                "\nNo ground truth supplied.\n"
            )

            file.write(
                "Quality metrics were not calculated.\n"
            )

    # ========================================================
    # FINAL TERMINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("SAF DEPLOYMENT COMPLETE")
    print("=" * 70)

    print(
        f"Images processed : "
        f"{len(input_files)}"
    )

    print(
        f"Restored images  : "
        f"{output_dir}"
    )

    print(
        f"Total time       : "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Images/sec       : "
        f"{len(input_files) / total_time:.2f}"
    )

    if evaluated > 0:

        print()
        print("QUALITY METRICS")
        print("-" * 70)

        print(
            f"PSNR : "
            f"{np.mean(psnr_values):.4f} dB"
        )

        print(
            f"SSIM : "
            f"{np.mean(ssim_values):.4f}"
        )

        print(
            f"MAE  : "
            f"{np.mean(mae_values):.6f}"
        )

        print(
            f"MSE  : "
            f"{np.mean(mse_values):.6f}"
        )

        print(
            f"Edge Error : "
            f"{np.mean(edge_values):.6f}"
        )

        print(
            f"\nEvaluated pairs : "
            f"{evaluated}"
        )

    else:

        print()
        print(
            "No ground truth supplied."
        )

        print(
            "Quality metrics skipped."
        )

    print()
    print(
        "Per-image results:"
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
        # ========================================================
    # STRUCTURAL EXPLAINABILITY
    # ========================================================

    if evaluated == 0:

        try:

            visual_output_dir = (
                output_dir /
                "visual_analysis"
            )

            generate_structural_analysis(
                input_dir,
                output_dir,
                visual_output_dir
            )

        except Exception as error:

            print()
            print(
                "WARNING: Structural analysis failed:"
            )

            print(
                error
            )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()