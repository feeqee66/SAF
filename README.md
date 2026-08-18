# SAF
### KLA — AI-Based Restoration of Degraded Images

SAF is an offline grayscale image restoration system. It reconstructs restored images at 2× resolution from degraded low-resolution `.npy` inputs using a residual CNN with PixelShuffle upsampling.

| | |
|---|---|
| **Task** | KLA — AI-Based Restoration of Degraded Images |
| **Model** | Residual CNN + PixelShuffle ×2 |
| **Parameters** | 776,705 |
| **Input** | 128 × 128 grayscale |
| **Output** | 256 × 256 grayscale |
| **Framework** | PyTorch |
| **Execution** | Offline, single command |

---

## 1. Overview

SAF accepts a directory of degraded `.npy` grayscale images and produces restored `.npy` outputs at double the input resolution. The deployed checkpoint (`models/cnn_l1_best_deployed.pt`) is bundled with the repository, and the entire pipeline runs offline with no API keys, no internet access, and no external model downloads.

Judge-facing execution is a single command:

```bash
python run.py <input-dir> <output-dir>
```

No ground-truth data is required to run the submission. When ground truth is available, SAF additionally computes quantitative image-quality metrics. When it is not, SAF produces a structural explainability visualization instead of any correctness claim.

---

## 2. Key Capabilities

- One-command offline restoration of `.npy` inputs to `.npy` outputs
- 2× resolution restoration via a residual CNN + PixelShuffle architecture
- Output validation: grayscale, `[0, 1]` range, NaN/Inf handling
- Filename-preserving batch processing
- Optional quantitative evaluation (PSNR, SSIM, MAE, MSE, Edge Error) when GT is supplied
- Structural explainability visualization when GT is not supplied
- NVIDIA GPU compatible, with CPU fallback in the evaluation pipeline

---

## 3. System Architecture

```
Input (128×128 grayscale)
        │
        ▼
Convolutional feature extraction
        │
        ▼
Residual blocks (×8, 64 channels)
        │
        ▼
Feature refinement
        │
        ▼
PixelShuffle ×2 upsampling
        │
        ▼
Output convolution
        │
        ▼
Restored output (256×256 grayscale)
```

<details>
<summary><strong>Architecture characteristics</strong></summary>

- Grayscale, single-channel input and output
- Residual CNN backbone
- 8 residual blocks
- 64 feature channels
- PixelShuffle ×2 subpixel upsampling
- 776,705 trainable parameters

</details>

---

## 4. Model Specification

| Property | Value |
|---|---|
| Checkpoint | `models/cnn_l1_best_deployed.pt` |
| Architecture | Residual CNN + PixelShuffle ×2 |
| Parameters | 776,705 |
| Scale factor | 2× |
| Input resolution | 128 × 128 |
| Output resolution | 256 × 256 |
| Output dtype | float32 |
| Output range | [0, 1] |

---

## 5. Input and Output

**Input**
- `.npy` files
- Grayscale
- Expected low-resolution input: 128 × 128

**Output**
- `.npy` files
- Grayscale, float32
- Resolution: 256 × 256
- Values constrained to `[0, 1]`
- No NaN or Inf values

Output filenames match input filenames exactly.

```
input:  000281.npy
output: 000281.npy
```

---

## 6. Repository Structure

```
SAF/
├── run.py
├── requirements.txt
├── README.md
├── models/
│   └── cnn_l1_best_deployed.pt
└── evaluation/
    └── ...
```

**Core submission** (required for judge execution):
```
run.py
requirements.txt
README.md
models/
```

**Supporting evaluation and analysis** (not required for the judge command):

<details>
<summary>Expand full <code>evaluation/</code> contents</summary>

```
evaluation/
├── run.py                      # offline deployment pipeline
├── visual_analysis.py          # explainability / visual comparisons
├── metrics.py
├── advanced_metrics.py
├── edge_metrics.py
├── frequency_metrics.py
├── frequency_structure.py
├── lpips_evaluation.py
├── compare_baseline.py
├── compare_model.py
├── compare_final.py
├── evaluate_dataset.py
├── evaluate_restored.py
├── evaluate_bicubic_subset.py
├── bicubic_frequency.py
├── check_data_range.py
├── test_gallery.py
└── results/
```

</details>

---

## 7. Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the Python dependencies required to run the submission. The model checkpoint does not need to be downloaded separately; it is already included at `models/cnn_l1_best_deployed.pt`.

---

## 8. Running the Submission

```bash
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py "path/to/input" "path/to/output"
```

`run.py` at the repository root is the judge-facing entry point. It:

1. Validates the two required command-line arguments (input directory, output directory)
2. Validates the input directory
3. Creates the output directory if it does not already exist
4. Invokes the offline deployment pipeline in `evaluation/run.py` using the current Python interpreter
5. Passes through the input and output directories

No interactive prompts occur during this execution path. Judges do not need to invoke `evaluation/run.py` directly.

<details>
<summary>Expected pipeline behavior</summary>

1. Deployed model is located and loaded from `models/`
2. All `.npy` files in the input directory are discovered
3. Each image is restored
4. Output directory is created if required
5. Restored `.npy` files are written using the original input filenames
6. Processing progress and timing statistics are printed
7. `metrics.csv` and `summary.txt` are generated
8. If ground truth is not available, GT-dependent metrics are skipped and structural explainability artifacts are produced instead

</details>

---

## 9. Ground-Truth Evaluation (Mode 2)

When corresponding ground-truth `.npy` files are available, the evaluation pipeline compares each restored image against its GT counterpart and computes:

| Metric | Description |
|---|---|
| PSNR | Peak signal-to-noise ratio |
| SSIM | Structural similarity |
| MAE | Mean absolute error |
| MSE | Mean squared error |
| Edge Error | Edge-region reconstruction error |

Aggregate results are written to `summary.txt`, including the number of evaluated pairs, the metrics above, total inference time, and average images/sec. Per-image results are written to `metrics.csv`.

These metrics require corresponding ground-truth images and are **not** part of the official judge-facing deployment command — they are only produced when GT is supplied to the evaluation pipeline.

---

## 10. No-Ground-Truth Structural Explainability (Mode 1)

The official judge command does not require ground truth:

```bash
python run.py <input-dir> <output-dir>
```

When no GT is available, SAF still performs complete restoration and writes all restored `.npy` outputs normally. GT-dependent metrics (PSNR, SSIM, MAE, MSE, Edge Error) are not computed or claimed in this mode.

Instead, SAF generates a **structural explainability visualization** by comparing the degraded input and restored output using local image-gradient/structural information. The resulting figure contains three panels:

1. Degraded Input
2. Restored Output
3. Structural Change Map

The structural change map highlights regions where local image structure/detail changes between the degraded input and the restored output — brighter regions indicate stronger structural/detail change, darker regions indicate relatively little change.

The accompanying `explanation.txt` records the mode as `"Mode: No Ground Truth"` and describes the structural change map in these terms.

**Interpretation note:** the structural change map is an explainability visualization only. It is not a reconstruction-error map, and it is not a confidence map.

---

## 11. Output Artifacts

| Artifact | Produced when | Description |
|---|---|---|
| Restored `.npy` files | Always | One per input file, same filename, 256×256 grayscale, `[0,1]` |
| `metrics.csv` | Always | Per-image processing record; GT-dependent columns populated only when GT is supplied |
| `summary.txt` | Always | Aggregate run statistics (timing, images/sec; GT metrics when available) |
| Structural map / comparison figure | No-GT mode | Degraded input vs. restored output vs. structural change map |
| `explanation.txt` | No-GT mode | States evaluation mode and explains the structural change map |

---

## 12. Validation and Safety Checks

The inference pipeline enforces the following on every output before it is written:

- Output is grayscale
- Output values are constrained to `[0, 1]`
- NaN and Inf values are rejected/handled
- Output resolution matches the expected target (256 × 256)
- Output filename matches the corresponding input filename
- Output directory is created automatically if it does not exist

---

## 13. Offline and Hardware Compatibility

- Runs fully offline — no internet connection required at inference time
- No API keys required
- No external model downloads required
- Model checkpoint is bundled in the repository (`models/cnn_l1_best_deployed.pt`)
- Compatible with NVIDIA GPU execution
- CPU fallback available in the evaluation pipeline
- No manual model configuration or user interaction required during judge execution

---

## 14. Limitations / Interpretation

SAF distinguishes two evaluation situations, and this distinction should be read carefully:

| | Compares | Produces |
|---|---|---|
| **Ground truth available** | Restored vs. GT | PSNR, SSIM, MAE, MSE, Edge Error |
| **No ground truth** | Degraded input vs. Restored | Structural explainability visualization |

The structural change map produced in the no-GT case does not measure correctness against an unknown ground truth, does not replace PSNR/SSIM, and is not a confidence score. It exists to make the restoration's structural behavior interpretable when quantitative evaluation is not possible.

---

## 15. Technical Summary

```
Task:            KLA — AI-Based Restoration of Degraded Images
Model:           Residual CNN + PixelShuffle ×2
Parameters:      776,705
Input:           128×128 grayscale (.npy)
Output:          256×256 grayscale (.npy), float32, [0,1]
Execution:       python run.py <input-dir> <output-dir>
Ground truth:    Optional — enables PSNR/SSIM/MAE/MSE/Edge Error
No ground truth: Structural explainability visualization + explanation.txt
Runtime:         Offline, no API keys, no external downloads
Hardware:        NVIDIA GPU compatible, CPU fallback available
```