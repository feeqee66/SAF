# SAF

Offline AI-based restoration and super-resolution of degraded grayscale images via learned residual reconstruction.

## Overview

SAF addresses the KLA problem statement for AI-based restoration of degraded images. The system takes low-resolution, degraded grayscale images and reconstructs high-resolution versions using a learned Residual CNN architecture with sub-pixel convolution (PixelShuffle).

Input images are expected as 128×128 grayscale NumPy arrays (.npy format). The model produces 256×256 grayscale reconstructions, achieving a 2× upsampling factor. All inference is performed locally with no internet access, API calls, or external model downloads. The system supports both CPU and NVIDIA GPU execution.

Beyond basic inference, SAF includes a comprehensive evaluation pipeline for measuring reconstruction quality when ground-truth images are available, along with visual analysis tooling for spatial error inspection and high-error region identification.

## Key Features

- **Offline inference**: No internet, APIs, or external model downloads required
- **Grayscale .npy I/O**: Native support for NumPy array format
- **2× super-resolution**: 128×128 → 256×256 upsampling
- **Residual CNN architecture**: 776,705 parameters, 8 residual blocks with PixelShuffle
- **GPU and CPU compatible**: Automatic device detection; NVIDIA GPU acceleration supported
- **Deterministic deployment**: Complete model weights included locally
- **Evaluation pipeline**: Objective metrics (PSNR, SSIM, MAE, MSE, edge error) when ground truth is available
- **Visual error analysis**: Spatial error heatmaps and high-error region inspection
- **Reproducible**: Single-command execution with documented inputs/outputs

## System Pipeline

### Inference Pipeline (Run)

```
Degraded .npy (128×128)
        ↓
Input validation & normalization
        ↓
Residual CNN feature extraction
        ↓
Feature refinement
        ↓
PixelShuffle (2×)
        ↓
Output convolution & reconstruction
        ↓
Value clipping to [0, 1]
        ↓
Restored .npy (256×256)
```

### Evaluation Pipeline (Optional)

When paired ground-truth images are provided:

```
Restored image + Ground-truth image
        ↓
Metric computation (PSNR, SSIM, MAE, MSE, edge error)
        ↓
Spatial error analysis
        ↓
Error heatmap generation
        ↓
High-error region detection & cropping
        ↓
Results & visual analysis outputs
```

## Model Architecture

**Architecture**: Residual CNN + PixelShuffle ×2

**Specifications**:
- Input channels: 1 (grayscale)
- Input resolution: 128×128
- Output resolution: 256×256
- Upsampling scale factor: 2×
- Feature channels: 64
- Residual blocks: 8
- Total parameters: 776,705
- Framework: PyTorch
- Checkpoint: `models/cnn_l1_best_deployed.pt`

**Architecture Breakdown**:
1. Convolutional head (1 → 64 channels)
2. Eight residual blocks with skip connections
3. Feature refinement convolution
4. Sub-pixel convolution (PixelShuffle) with scale factor 2
5. Final reconstruction convolution (64 → 1 channel)

The model was trained on L1 loss and the checkpoint includes deployment metadata (format version, architecture type, training epoch, validation PSNR, validation SSIM).

## Input and Output Format

### Input

- **Format**: `.npy` files (NumPy binary format)
- **Color space**: Grayscale (single channel)
- **Resolution**: 128×128 pixels
- **Data type**: Typically float32 in range [0, 1]
- **Handling**: All `.npy` files in the input directory are processed

### Output

- **Format**: `.npy` files (NumPy binary format)
- **Color space**: Grayscale (single channel)
- **Resolution**: 256×256 pixels
- **Data type**: float32
- **Value range**: [0, 1] (clipped; NaN and Inf safeguarded)
- **Naming**: Filenames preserved from input
- **Location**: Output directory specified via command-line argument

### Example

If the input directory contains:
```
input/
├── 000001.npy
├── 000002.npy
└── 000003.npy
```

The output directory will contain:
```
output/
├── 000001.npy
├── 000002.npy
└── 000003.npy
```

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.9+
- NumPy
- (Optional) CUDA 11.0+ for GPU acceleration

### Setup

1. Clone or download the repository:
   ```bash
   git clone <repo-url>
   cd SAF
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   All required packages and model weights are included in the repository. No external downloads are necessary.

3. Verify the deployment checkpoint is present:
   ```
   models/cnn_l1_best_deployed.pt
   ```

## Running the Submission

### Official Command

```bash
python run.py <input-dir> <output-dir>
```

This is the single command required to run the complete inference pipeline.

### Examples

**Linux/macOS**:
```bash
python run.py "./test_input" "./test_output"
```

**Windows**:
```bash
python run.py "C:\path\to\input" "C:\path\to\output"
```

### Execution

- The system automatically detects input files in `<input-dir>`
- Creates `<output-dir>` if it does not exist
- Processes all `.npy` files in parallel or sequentially depending on implementation
- Outputs restored images with preserved filenames
- Terminates with status code 0 on success

**No prompts, no configuration, no manual setup required.**

## Output

After running `python run.py <input-dir> <output-dir>`, the output directory contains:

- **Restored images**: One `.npy` file per input image, with the same filename
- **Metrics (if ground truth provided)**: CSV file with objective metrics and visual analysis results
- **Error analysis (if ground truth provided)**: Spatial error maps and high-error region crops

The deployment pipeline always produces the restored `.npy` files. Evaluation and visual analysis outputs are generated when ground-truth images are available during the evaluation phase.

## Evaluation and Visual Analysis

SAF includes a comprehensive evaluation pipeline in the `evaluation/` directory. This pipeline is distinct from the deployment pipeline but shares the same trained model.

### When Ground Truth Is Available

The evaluation system can compute objective metrics:
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **Edge Error** (reconstruction accuracy on image edges)

And generate visual analysis:
- Side-by-side comparisons: degraded input, restored output, ground truth
- **Spatial error heatmaps**: Pixel-wise absolute difference between restored and ground truth
- **High-error region detection**: Automatic cropping and analysis of regions where reconstruction error is highest
- **Frequency-domain analysis**: FFT-based structural validation
- **Perceptual metrics**: LPIPS evaluation when applicable

Error heatmaps visualize:
```
E(x,y) = |Restored(x,y) - Ground Truth(x,y)|
```

This is **not** a model confidence map; it is ground-truth error computed only when paired ground-truth data is available.

### When Ground Truth Is Unavailable

- The deployment pipeline produces restored `.npy` files normally
- Objective error heatmaps cannot be generated (no ground truth to compare against)
- Quality assessment cannot be performed quantitatively

This distinction is important: the system does not fabricate error measurements or confidence scores without ground truth.

## Results

### Paired Image Evaluation (10-image test set)

When evaluated on 10 paired (degraded, ground-truth) image pairs:

| Metric | Value |
|--------|-------|
| PSNR | 27.9887 dB |
| SSIM | 0.7367 |
| MAE | 0.030892 |
| MSE | 0.002488 |
| Edge Error | 0.034087 |
| Pairs evaluated | 10 |

### Inference Performance (Deployment test)

Deployment test on 400 images without ground truth:

| Metric | Value |
|--------|-------|
| Images processed | 400 |
| Output resolution | 256×256 |
| Throughput | 4.3–4.5 images/sec |
| Hardware | NVIDIA GPU (tested environment) |

**Note**: This inference speed is measured on the team's test machine. Performance varies based on hardware (GPU model, driver version, CPU, RAM). CPU-only inference will be significantly slower.

## Offline / Deployment Design

SAF is designed for offline, deterministic deployment:

- **No internet connectivity required**: The system operates completely locally
- **No API calls**: All inference is performed by the on-device model
- **No external downloads**: Model weights are committed to the repository
- **No configuration files**: The system runs with a single command
- **Deterministic output**: Same inputs always produce the same outputs (barring floating-point numerical differences)
- **GPU acceleration**: Automatic detection and utilization of NVIDIA GPU when available
- **CPU fallback**: Inference supported on CPU if GPU is unavailable

The model checkpoint (`models/cnn_l1_best_deployed.pt`) contains all necessary parameters and metadata for immediate inference without further setup.

## Repository Structure

```
SAF/
├── run.py                      # Official submission entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── configs/                    # Configuration files
├── datasets/                   # Dataset utilities and loaders
├── evaluation/                 # Evaluation pipeline and visual analysis
│   ├── run.py                  # Evaluation entry point (internal use)
│   ├── metrics/                # Metric computation modules
│   └── analysis/               # Visual analysis and heatmap generation
├── model/                      # Model architecture definitions
├── models/                     # Trained model checkpoints
│   └── cnn_l1_best_deployed.pt # Deployment checkpoint (776K parameters)
└── scripts/                    # Utility scripts and preprocessing
```

**Key directories**:

- `run.py`: Main entry point. Accepts input and output directories and invokes the complete inference pipeline.
- `model/`: Architecture definitions for the Residual CNN model.
- `models/`: Committed model checkpoints, including the deployment version.
- `evaluation/`: Evaluation pipeline for objective metrics and visual error analysis. Used for benchmarking and analysis; not required for basic inference.
- `configs/`: Configuration files and hyperparameters.
- `datasets/`: Data loading utilities.
- `scripts/`: Helper scripts for preprocessing and analysis.

## Reproducibility

To reproduce inference on a new set of images:

```bash
python run.py <your-input-directory> <your-output-directory>
```

To reproduce evaluation results with ground-truth images, refer to the `evaluation/` directory documentation.

The model used is deterministic. Given the same input and hardware, outputs should be numerically identical (within floating-point precision).

## Technical Notes and Limitations

- **Grayscale input only**: The model expects single-channel grayscale images. Multi-channel inputs must be converted to grayscale before processing.
- **Fixed input resolution**: Input images must be 128×128. Images of different sizes require resizing before inference.
- **Ground-truth dependent metrics**: PSNR, SSIM, MAE, MSE, and edge error require paired ground-truth images. These metrics are not computed for deployment-only inference.
- **Ground-truth dependent error heatmaps**: Spatial error heatmaps require ground-truth images for comparison. Without ground truth, error heatmaps cannot be generated.
- **Output value range**: Outputs are constrained to [0, 1]. Values outside this range (including NaN and Inf) are clipped or safeguarded.
- **Inference speed**: Hardware-dependent. GPU acceleration significantly faster than CPU-only inference.
- **Model size**: 776,705 parameters; negligible memory footprint on modern hardware.

## Submission Compliance

This submission adheres to the KLA hackathon requirements:

- [x] `run.py` accepts input and output directory arguments
- [x] Reads all `.npy` files from input directory
- [x] Creates output directory automatically if necessary
- [x] Preserves input filenames in output
- [x] Produces grayscale output arrays
- [x] Output values constrained to [0, 1] range
- [x] NaN and Inf values safeguarded
- [x] Correct target resolution (256×256)
- [x] Model weights included locally in repository
- [x] `requirements.txt` included and complete
- [x] `README.md` provided with clear instructions
- [x] No internet access required during execution
- [x] No API keys or external model downloads required
- [x] NVIDIA GPU compatible (CPU fallback if applicable)
- [x] Offline-first design with no external dependencies

## Contact

For technical questions or issues, refer to the project documentation or contact the development team.