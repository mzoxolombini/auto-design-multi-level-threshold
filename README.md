# Multi-Level Image Thresholding with Bio-Inspired Algorithms

This repository implements and benchmarks **multi-level image thresholding** using a collection of
bio-inspired and hybrid metaheuristic optimisation algorithms.  All implementations accompany the
paper:

> **Automated Design of Multilevel Thresholding: Single-point vs Multipoint**

The paper is included in the repository as
`Automated_Design_of__Multilevel_Thresholding__Single_point_vs_Multipoint.pdf`.

---

## Overview

Multi-level thresholding segments a grayscale image into multiple regions by finding *k* optimal
threshold values that partition the 256-level intensity histogram.  Each algorithm searches for the
threshold set that maximises a chosen fitness function (Kapur's Entropy or Otsu's Between-Class
Variance) and reports four image-quality metrics on the resulting segmented image.

All algorithms are tested on threshold levels **k = 2 through 10** and on every image in the
supplied input folder.

---

## Algorithms

Scripts are organised into two categories that match the paper's taxonomy:

| Category | Characteristic |
|---|---|
| **Single-Point Search** | Updates one candidate solution at a time; local refinement around the current best. |
| **Multi-Point Search** | Evolves or coordinates a population of candidate solutions across multiple search points. |

### Single-Point Search

| File | Abbreviation | Description |
|---|---|---|
| `single_point/ga_multilevel_thresholding.py` | **GA** | Standard Genetic Algorithm with tournament selection, single-point crossover, and random mutation. |
| `single_point/abc_multilevel_thresholding.py` | **ABC** | Artificial Bee Colony — employed, onlooker, and scout bees cooperate to find optimal thresholds. |
| `single_point/ils_ga_multilevel_thresholding.py` | **ILSGA** | GA extended with an Iterated Local Search (ILS) perturbation phase and early-stopping. |
| `single_point/ils_abc_multilevel_thresholding.py` | **ILSABC** | ABC followed by an ILS phase that refines and perturbs the best solution to escape local optima. |
| `single_point/sa_ga_multilevel_thresholding.py` | **SAGA** | GA with Simulated Annealing applied at each generation to refine the generation-best solution. |
| `single_point/sa_optimized_abc_thresholding.py` | **SAABC** | Simulated Annealing searches the ABC hyperparameter space (population size, iterations, limit) before running the final ABC. |

### Multi-Point Search

| File | Abbreviation | Description |
|---|---|---|
| `multi_point/meta_ga_optimized_ga_thresholding.py` | **GAGA** | A Meta-Genetic Algorithm evolves GA hyperparameters; the best configuration is then used to run the final GA. |
| `multi_point/ga_optimized_abc_thresholding.py` | **GAABC** | A GA searches the ABC hyperparameter space; the best ABC configuration is then run on each image. |
| `multi_point/de_optimized_ga_thresholding.py` | **DEGA** | Differential Evolution optimises GA hyperparameters (population size, crossover rate, mutation rate, generations) before running the final GA. |
| `multi_point/de_optimized_abc_thresholding.py` | **DEABC** | Differential Evolution optimises the ABC colony parameters (colony size, iterations, limit) before running the final ABC. |

---

## Fitness Functions

Each algorithm evaluates candidate threshold sets using one of two fitness functions, selectable at
runtime via the string key shown below:

| Key | Function | Objective |
|---|---|---|
| `"kapur"` | **Kapur's Entropy** | Maximises the sum of Shannon entropies of the normalised pixel-intensity distributions across all *k + 1* segments. |
| `"otsu"` | **Otsu's Between-Class Variance** | Maximises the weighted between-class variance, equivalent to minimising within-class variance. |

Both functions are implemented identically in every script so that results across algorithms are
directly comparable.

---

## Image Quality Metrics

After finding optimal thresholds, each script segments the image using per-segment midpoint
intensity values and evaluates the result with four metrics:

| Metric | Direction | Description |
|---|---|---|
| **SSIM** | Higher is better (range −1 to 1) | Structural Similarity Index — measures perceived structural similarity between the original and segmented image. |
| **MSE** | Lower is better | Mean Squared Error — average squared pixel-intensity difference. |
| **PSNR** | Higher is better (dB) | Peak Signal-to-Noise Ratio — derived from MSE; higher values indicate less distortion. |
| **Uniformity** | Higher is better | Histogram-based uniformity — sum of squared normalised histogram probabilities of the segmented image. |

---

## Output Schema

Every script writes an Excel file (`.xlsx`) and a folder of segmented images.  All output files
share the same column schema:

| Column | Type | Notes |
|---|---|---|
| `image_name` | string | Original filename |
| `fitness_function` | string | `"kapur"` or `"otsu"` |
| `thresholding_level` | int | *k* (number of thresholds, 2–10) |
| `threshold_value` | list / string | The *k* optimal threshold values |
| `fitness_value` | float | Best fitness achieved |
| `SSIM` | float | Structural Similarity Index |
| `MSE` | float | Mean Squared Error |
| `PSNR` | float | Peak Signal-to-Noise Ratio (dB) |
| `Uniformity` | float | Histogram uniformity measure |
| `Seed` | int | Random seed used for this run |
| `Execution Time (ms)` | float | Wall-clock runtime in milliseconds |

Pre-computed results for all algorithms are also available in
`MLT_spreadsheet_results_complete.xlsx`.

---

## Requirements

```bash
pip install numpy opencv-python pandas scikit-image scipy openpyxl joblib
```

---

## Usage

1. **Set paths** — open the script you want to run and update the two path variables in the
   `__main__` block:

   ```python
   folder_path = r"path/to/your/images"   # folder containing input images
   output_path = r"path/to/your/results"   # folder where results will be saved
   ```

2. **Run the script** directly from the repository root:

   ```bash
   # Single-Point algorithms
   python single_point/ga_multilevel_thresholding.py
   python single_point/abc_multilevel_thresholding.py
   python single_point/ils_ga_multilevel_thresholding.py
   python single_point/ils_abc_multilevel_thresholding.py
   python single_point/sa_ga_multilevel_thresholding.py
   python single_point/sa_optimized_abc_thresholding.py

   # Multi-Point algorithms
   python multi_point/meta_ga_optimized_ga_thresholding.py
   python multi_point/ga_optimized_abc_thresholding.py
   python multi_point/de_optimized_ga_thresholding.py
   python multi_point/de_optimized_abc_thresholding.py
   ```

Each script processes **all images** in `folder_path`, iterates over both fitness functions and all
threshold levels 2–10, and writes results incrementally so that partial runs are not lost.

---

## Supported Image Formats

`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.gif`

---

## Threshold Levels

All algorithms test threshold levels **k = 2 through 10** by default (configurable via the
`threshold_levels` list in each script's `__main__` block).

---

## Repository Structure

```
.
├── single_point/
│   ├── ga_multilevel_thresholding.py
│   ├── abc_multilevel_thresholding.py
│   ├── ils_ga_multilevel_thresholding.py
│   ├── ils_abc_multilevel_thresholding.py
│   ├── sa_ga_multilevel_thresholding.py
│   └── sa_optimized_abc_thresholding.py
├── multi_point/
│   ├── meta_ga_optimized_ga_thresholding.py
│   ├── ga_optimized_abc_thresholding.py
│   ├── de_optimized_ga_thresholding.py
│   └── de_optimized_abc_thresholding.py
├── MLT_spreadsheet_results_complete.xlsx
└── Automated_Design_of__Multilevel_Thresholding__Single_point_vs_Multipoint.pdf
```
