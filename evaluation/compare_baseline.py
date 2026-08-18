from pathlib import Path
import csv


# ============================================================
# BASELINE
# ============================================================

BASELINE = {
    "PSNR": 22.7973,
    "SSIM": 0.5306,
    "MAE": 0.056936,
    "MSE": 0.006911
}


# ============================================================
# CALCULATE IMPROVEMENT
# ============================================================

def calculate_improvement(model):

    result = {}

    # Higher is better
    result["PSNR_gain"] = (
        model["PSNR"] - BASELINE["PSNR"]
    )

    result["SSIM_gain"] = (
        model["SSIM"] - BASELINE["SSIM"]
    )

    # Lower is better
    result["MAE_reduction_percent"] = (
        (BASELINE["MAE"] - model["MAE"])
        / BASELINE["MAE"]
        * 100
    )

    result["MSE_reduction_percent"] = (
        (BASELINE["MSE"] - model["MSE"])
        / BASELINE["MSE"]
        * 100
    )

    return result


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

def load_results(csv_path):

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{csv_path}"
        )

    models = {}

    with open(
        csv_path,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            model_name = row["Model"]

            models[model_name] = {
                "PSNR": float(row["PSNR"]),
                "SSIM": float(row["SSIM"]),
                "MAE": float(row["MAE"]),
                "MSE": float(row["MSE"])
            }

    return models


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(models):

    print()
    print("=" * 85)
    print("BASELINE vs RESTORATION MODEL")
    print("=" * 85)

    print(
        f"{'Model':<18}"
        f"{'PSNR':>12}"
        f"{'SSIM':>12}"
        f"{'MAE':>14}"
        f"{'MSE':>14}"
    )

    print("-" * 85)

    print(
        f"{'Bicubic 2x':<18}"
        f"{BASELINE['PSNR']:>12.4f}"
        f"{BASELINE['SSIM']:>12.4f}"
        f"{BASELINE['MAE']:>14.6f}"
        f"{BASELINE['MSE']:>14.6f}"
    )

    for name, metrics in models.items():

        print(
            f"{name:<18}"
            f"{metrics['PSNR']:>12.4f}"
            f"{metrics['SSIM']:>12.4f}"
            f"{metrics['MAE']:>14.6f}"
            f"{metrics['MSE']:>14.6f}"
        )

    print("-" * 85)

    for name, metrics in models.items():

        improvement = calculate_improvement(
            metrics
        )

        print()
        print(f"{name} IMPROVEMENT")
        print("-" * 40)

        print(
            f"PSNR gain          : "
            f"{improvement['PSNR_gain']:+.4f} dB"
        )

        print(
            f"SSIM gain          : "
            f"{improvement['SSIM_gain']:+.4f}"
        )

        print(
            f"MAE reduction      : "
            f"{improvement['MAE_reduction_percent']:+.2f}%"
        )

        print(
            f"MSE reduction      : "
            f"{improvement['MSE_reduction_percent']:+.2f}%"
        )

    print()
    print("=" * 85)


# ============================================================
# SAVE IMPROVEMENT CSV
# ============================================================

def save_improvements(models, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fields = [
        "Model",
        "PSNR",
        "SSIM",
        "MAE",
        "MSE",
        "PSNR_gain",
        "SSIM_gain",
        "MAE_reduction_percent",
        "MSE_reduction_percent"
    ]

    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for name, metrics in models.items():

            improvement = calculate_improvement(
                metrics
            )

            writer.writerow({
                "Model": name,
                "PSNR": metrics["PSNR"],
                "SSIM": metrics["SSIM"],
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "PSNR_gain": improvement["PSNR_gain"],
                "SSIM_gain": improvement["SSIM_gain"],
                "MAE_reduction_percent":
                    improvement["MAE_reduction_percent"],
                "MSE_reduction_percent":
                    improvement["MSE_reduction_percent"]
            })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    results_file = (
        "evaluation/results/model_comparison.csv"
    )

    output_file = (
        "evaluation/results/baseline_comparison.csv"
    )

    models = load_results(
        results_file
    )

    print_results(
        models
    )

    save_improvements(
        models,
        output_file
    )

    print()
    print(
        f"Saved comparison to:\n"
        f"{output_file}"
    )