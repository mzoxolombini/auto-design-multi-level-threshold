# Multi-Level Image Thresholding with Bio-Inspired Algorithms

This repository implements **multi-level image thresholding** using a variety of bio-inspired and hybrid metaheuristic optimization algorithms. Each algorithm optimizes threshold values by maximizing either **Kapur's entropy** or **Otsu's between-class variance** as a fitness function.

---

## Overview

Multi-level thresholding segments a grayscale image into multiple regions by finding optimal threshold values. This project benchmarks several standalone and hybrid algorithms across threshold levels 2–10, evaluating each using standard image quality metrics.

---

## Algorithms

### Standalone Algorithms

| File | Algorithm | Description |
|---|---|---|
| `abc_multilevel_thresholding.py` | **ABC** – Artificial Bee Colony | Simulates foraging behaviour of honey bees. Employed, onlooker, and scout bees cooperate to search for optimal thresholds. |
| `ga_multilevel_thresholding.py` | **GA** – Genetic Algorithm | Evolves a population of candidate threshold sets using tournament selection, single-point crossover, and random mutation. |

### Hybrid / Meta-Optimized Algorithms

| File | Algorithm | Description |
|---|---|---|
| `de_optimized_abc_thresholding.py` | **DE → ABC** | Differential Evolution optimizes the ABC colony parameters before running the ABC algorithm. |
| `de_optimized_ga_thresholding.py` | **DE → GA** | Differential Evolution optimizes GA hyperparameters (population size, crossover rate, mutation rate, generations). |
| `ga_optimized_abc_thresholding.py` | **GA → ABC** | A GA-configured ABC runs a pure Artificial Bee Colony with parameters informed by a GA-style configuration. |
| `meta_ga_optimized_ga_thresholding.py` | **Meta-GA → GA** | A Meta-Genetic Algorithm evolves GA hyperparameters; the best configuration is then used to run the final GA. |
| `ils_abc_multilevel_thresholding.py` | **ILS + ABC** | ABC is followed by an Iterated Local Search (ILS) phase that refines and perturbs the best solution to escape local optima. |
| `ils_ga_multilevel_thresholding.py` | **ILS + GA** | GA with an integrated ILS perturbation strategy and early-stopping. |
| `sa_optimized_abc_thresholding.py` | **SA → ABC** | Simulated Annealing tunes ABC hyperparameters (population size, iterations, limit) before running the final ABC. |
| `sa_ga_multilevel_thresholding.py` | **SA + GA** | GA with Simulated Annealing applied at each generation to refine the best solution found. |

---

## Fitness Functions

Each algorithm supports two fitness functions selectable at runtime:

- **Kapur's Entropy** – Maximizes the sum of class entropies across threshold segments.
- **Otsu's Between-Class Variance** – Maximizes the weighted between-class variance.

---

## Image Quality Metrics

After finding optimal thresholds, each script evaluates the segmented image using:

| Metric | Description |
|---|---|
| **SSIM** | Structural Similarity Index (higher is better, range –1 to 1) |
| **MSE** | Mean Squared Error (lower is better) |
| **PSNR** | Peak Signal-to-Noise Ratio in dB (higher is better) |
| **Uniformity** | Histogram-based uniformity measure of the segmented image |

---

## Requirements

```bash
pip install numpy opencv-python pandas scikit-image scipy openpyxl joblib
```

---

## Usage

Each script processes all images in a specified input folder and saves results to an Excel file. Update the `folder_path` and `output_path` variables in the `__main__` block of any script before running:

```python
folder_path = r"path/to/your/images"
output_path = r"path/to/your/results"
```

Then run the script directly:

```bash
python abc_multilevel_thresholding.py
python ga_multilevel_thresholding.py
# ... etc.
```

---

## Output

Each script produces:
- An **Excel file** (`.xlsx`) with results for all images, fitness functions, and threshold levels.
- A folder of **segmented images** (one per image × threshold level × fitness function combination).

---

## Supported Image Formats

`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.gif`

---

## Threshold Levels

All algorithms are configured to test threshold levels **2 through 10** by default.
