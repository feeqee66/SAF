from pathlib import Path
import csv


# ============================================================
# PATHS
# ============================================================

MODEL_CSV = Path(
    "evaluation/results/model1_per_image.csv"
)

BASELINE_CSV = Path(
    "evaluation/results/bicubic_subset_per_image.csv"
)

OUTPUT_CSV = Path(
    "evaluation/results/final_comparison.csv"
)


# ============================================================
# LOAD CSV
# ============================================================

def load_results(path):
    """
    Load per-image metric CSV into a dictionary.

    Returns:
        {
            image_id: {
                metric: value,
                ...
            }
        }
    """

    results = {}

    with open(
        path,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            image_id = row["Image"]

            results[image_id] = {
                "PSNR": float(row["PSNR"]),
                "SSIM": float(row["SSIM"]),
                "MAE": float(row["MAE"]),
                "MSE": float(row["MSE"]),
                "Edge_Error":
                    float(row["Edge_Error"]),
                "Edge_Preservation":
                    float(row["Edge_Preservation"])
            }

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL MODEL 1 vs BICUBIC COMPARISON")
    print("=" * 70)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not MODEL_CSV.exists():
        raise FileNotFoundError(
            f"Model CSV not found:\n{MODEL_CSV}"
        )

    if not BASELINE_CSV.exists():
        raise FileNotFoundError(
            f"Baseline CSV not found:\n{BASELINE_CSV}"
        )

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    model = load_results(
        MODEL_CSV
    )

    baseline = load_results(
        BASELINE_CSV
    )

    print(
        f"Model 1 images : {len(model)}"
    )

    print(
        f"Bicubic images : {len(baseline)}"
    )

    # --------------------------------------------------------
    # Find common images
    # --------------------------------------------------------

    common_ids = sorted(
        set(model.keys())
        &
        set(baseline.keys())
    )

    print(
        f"Common images  : {len(common_ids)}"
    )

    if len(common_ids) == 0:
        raise RuntimeError(
            "No common image IDs found."
        )

    # ========================================================
    # DATASET AVERAGES
    # ========================================================

    metrics = [
        "PSNR",
        "SSIM",
        "MAE",
        "MSE",
        "Edge_Error",
        "Edge_Preservation"
    ]

    model_avg = {}
    baseline_avg = {}

    for metric in metrics:

        model_avg[metric] = (
            sum(
                model[i][metric]
                for i in common_ids
            )
            / len(common_ids)
        )

        baseline_avg[metric] = (
            sum(
                baseline[i][metric]
                for i in common_ids
            )
            / len(common_ids)
        )

    # ========================================================
    # IMPROVEMENTS
    # ========================================================

    # Higher is better
    psnr_gain = (
        model_avg["PSNR"]
        -
        baseline_avg["PSNR"]
    )

    ssim_gain = (
        model_avg["SSIM"]
        -
        baseline_avg["SSIM"]
    )

    edge_preservation_gain = (
        model_avg["Edge_Preservation"]
        -
        baseline_avg["Edge_Preservation"]
    )

    # Lower is better
    mae_reduction = (
        (
            baseline_avg["MAE"]
            -
            model_avg["MAE"]
        )
        /
        baseline_avg["MAE"]
    ) * 100.0

    mse_reduction = (
        (
            baseline_avg["MSE"]
            -
            model_avg["MSE"]
        )
        /
        baseline_avg["MSE"]
    ) * 100.0

    edge_error_reduction = (
        (
            baseline_avg["Edge_Error"]
            -
            model_avg["Edge_Error"]
        )
        /
        baseline_avg["Edge_Error"]
    ) * 100.0

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print()
    print(
        f"{'Metric':<25}"
        f"{'Bicubic':>15}"
        f"{'Model 1':>15}"
        f"{'Improvement':>15}"
    )

    print("-" * 70)

    print(
        f"{'PSNR (dB)':<25}"
        f"{baseline_avg['PSNR']:>15.4f}"
        f"{model_avg['PSNR']:>15.4f}"
        f"{psnr_gain:>+15.4f}"
    )

    print(
        f"{'SSIM':<25}"
        f"{baseline_avg['SSIM']:>15.4f}"
        f"{model_avg['SSIM']:>15.4f}"
        f"{ssim_gain:>+15.4f}"
    )

    print(
        f"{'MAE':<25}"
        f"{baseline_avg['MAE']:>15.6f}"
        f"{model_avg['MAE']:>15.6f}"
        f"{mae_reduction:>14.2f}%"
    )

    print(
        f"{'MSE':<25}"
        f"{baseline_avg['MSE']:>15.6f}"
        f"{model_avg['MSE']:>15.6f}"
        f"{mse_reduction:>14.2f}%"
    )

    print(
        f"{'Edge Error':<25}"
        f"{baseline_avg['Edge_Error']:>15.6f}"
        f"{model_avg['Edge_Error']:>15.6f}"
        f"{edge_error_reduction:>14.2f}%"
    )

    print(
        f"{'Edge Preservation':<25}"
        f"{baseline_avg['Edge_Preservation']:>15.6f}"
        f"{model_avg['Edge_Preservation']:>15.6f}"
        f"{edge_preservation_gain:>+15.6f}"
    )

    print("=" * 70)

    # ========================================================
    # SAVE SUMMARY CSV
    # ========================================================

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = [
        {
            "Metric": "PSNR",
            "Bicubic": baseline_avg["PSNR"],
            "Model1": model_avg["PSNR"],
            "Improvement": psnr_gain,
            "Improvement_Type": "Absolute gain (dB)"
        },
        {
            "Metric": "SSIM",
            "Bicubic": baseline_avg["SSIM"],
            "Model1": model_avg["SSIM"],
            "Improvement": ssim_gain,
            "Improvement_Type": "Absolute gain"
        },
        {
            "Metric": "MAE",
            "Bicubic": baseline_avg["MAE"],
            "Model1": model_avg["MAE"],
            "Improvement": mae_reduction,
            "Improvement_Type": "Percent reduction"
        },
        {
            "Metric": "MSE",
            "Bicubic": baseline_avg["MSE"],
            "Model1": model_avg["MSE"],
            "Improvement": mse_reduction,
            "Improvement_Type": "Percent reduction"
        },
        {
            "Metric": "Edge_Error",
            "Bicubic": baseline_avg["Edge_Error"],
            "Model1": model_avg["Edge_Error"],
            "Improvement": edge_error_reduction,
            "Improvement_Type": "Percent reduction"
        },
        {
            "Metric": "Edge_Preservation",
            "Bicubic":
                baseline_avg["Edge_Preservation"],
            "Model1":
                model_avg["Edge_Preservation"],
            "Improvement":
                edge_preservation_gain,
            "Improvement_Type": "Absolute gain"
        }
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Metric",
                "Bicubic",
                "Model1",
                "Improvement",
                "Improvement_Type"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Final comparison saved to:\n"
        f"{OUTPUT_CSV}"
    )

    # ========================================================
    # PPT-READY SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("PPT-READY SUMMARY")
    print("=" * 70)

    print(
        f"+{psnr_gain:.2f} dB PSNR improvement"
    )

    print(
        f"+{ssim_gain:.3f} SSIM improvement"
    )

    print(
        f"{mae_reduction:.1f}% MAE reduction"
    )

    print(
        f"{mse_reduction:.1f}% MSE reduction"
    )

    print(
        f"{edge_error_reduction:.1f}% edge-error reduction"
    )

    print(
        f"+{edge_preservation_gain:.4f} "
        f"edge preservation"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()