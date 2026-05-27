import os
import cv2
import numpy as np
import random
import pandas as pd
from scipy.ndimage import correlate
from joblib import Parallel, delayed
import sys
import time
import math


# ----------------- Corrected Utility Functions -----------------

def ssim(image1, image2):
    """Calculate Structural Similarity Index between two images."""
    img1 = image1.astype(np.float64) / 255.0
    img2 = image2.astype(np.float64) / 255.0

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    kernel_size = 11
    sigma = 1.5
    x = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    x, y = np.meshgrid(x, x)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    mu1 = correlate(img1, kernel, mode='reflect')
    mu2 = correlate(img2, kernel, mode='reflect')

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = correlate(img1 ** 2, kernel, mode='reflect') - mu1_sq
    sigma2_sq = correlate(img2 ** 2, kernel, mode='reflect') - mu2_sq
    sigma12 = correlate(img1 * img2, kernel, mode='reflect') - mu1_mu2

    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-10)
    return float(np.clip(np.mean(ssim_map), -1.0, 1.0))


def compute_2d_histogram(image, kernel_size=3):
    """Compute the joint 2D histogram of (pixel intensity, local neighborhood mean)."""
    local_mean = cv2.boxFilter(image.astype(np.float32), -1, (kernel_size, kernel_size))
    local_mean = np.clip(np.round(local_mean), 0, 255).astype(np.int32)
    flat_intensity = image.ravel().astype(np.int32)
    flat_mean = local_mean.ravel()
    hist_2d = np.zeros((256, 256), dtype=np.float64)
    np.add.at(hist_2d, (flat_intensity, flat_mean), 1)
    total = hist_2d.sum()
    if total > 0:
        hist_2d /= total
    return hist_2d


def renyi_entropy_region(p_region, q=2):
    """Compute Rényi entropy of order q for a 2D probability sub-region."""
    p_flat = p_region.ravel()
    p_flat = p_flat[p_flat > 1e-12]
    if len(p_flat) == 0:
        return 0.0
    p_norm = p_flat / p_flat.sum()
    if q == 1:
        return float(-np.sum(p_norm * np.log(p_norm)))
    return float((1.0 / (1.0 - q)) * np.log(np.sum(p_norm ** q)))


def calculate_renyi_entropy_2d(image, thresholds, q=2, kernel_size=3):
    """2D Rényi entropy (order q=2) objective function for multilevel thresholding."""
    thresholds = sorted(thresholds)
    boundaries = [0] + thresholds + [256]
    hist_2d = compute_2d_histogram(image, kernel_size)
    total_entropy = 0.0
    for i in range(len(boundaries) - 1):
        for j in range(len(boundaries) - 1):
            region = hist_2d[boundaries[i]:boundaries[i + 1],
                             boundaries[j]:boundaries[j + 1]]
            total_entropy += renyi_entropy_region(region, q)
    return total_entropy


def calculate_otsu(hist, thresholds):
    """Calculate Otsu's between-class variance for multilevel thresholding."""
    if len(thresholds) == 0:
        return 0.0

    thresholds = sorted(thresholds)
    total = np.sum(hist)
    if total == 0:
        return 0.0

    probs = hist / total
    global_mean = np.sum(np.arange(len(hist)) * probs)
    between_class_variance = 0.0
    thresholds = [0] + thresholds + [len(hist)]

    for i in range(1, len(thresholds)):
        start_idx = max(0, min(thresholds[i - 1], len(hist) - 1))
        end_idx = max(0, min(thresholds[i], len(hist)))

        if start_idx >= end_idx:
            continue

        w_i = np.sum(probs[start_idx:end_idx])
        if w_i > 1e-10:
            mu_i = np.sum(np.arange(start_idx, end_idx) *
                          probs[start_idx:end_idx]) / w_i
            between_class_variance += w_i * (mu_i - global_mean) ** 2

    return float(between_class_variance)


def calculate_mse(img1, img2):
    """Calculate Mean Squared Error."""
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def calculate_psnr(mse_value, max_pixel=255.0):
    """Calculate Peak Signal-to-Noise Ratio."""
    if mse_value == 0:
        return float('inf')
    return float(10 * math.log10((max_pixel ** 2) / mse_value))


def calculate_uniformity(image):
    """Calculate uniformity measure of an image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0:
        return 0.0
    hist_normalized = hist / total
    return float(np.sum(hist_normalized ** 2))


def apply_thresholds(image, thresholds):
    """Apply multiple thresholds to segment an image."""
    if len(thresholds) == 0:
        return np.zeros_like(image)

    thresholds = sorted(thresholds)
    segmented = np.zeros_like(image, dtype=np.uint8)

    segmented[image <= thresholds[0]] = 0

    for i in range(1, len(thresholds)):
        mask = (image > thresholds[i - 1]) & (image <= thresholds[i])
        segmented[mask] = i

    segmented[image > thresholds[-1]] = len(thresholds)

    max_val = len(thresholds)
    if max_val > 0:
        segmented = (segmented * (255 // max_val)).astype(np.uint8)

    return segmented


def format_number(value, decimals=8):
    """Format number with specified decimal places."""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if np.isinf(value):
        return "Infinity"
    fmt = f"{value:.{decimals}f}"
    return fmt  # Keep dots for decimal separator


# ----------------- Fixed Pure ABC Algorithm -----------------

class PureArtificialBeeColony:
    def __init__(self, config, num_thresholds, hist, fitness_function, seed=None):
        self.config = config
        self.num_thresholds = num_thresholds
        self.hist = hist
        self.fitness_function = fitness_function
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Ensure population size is reasonable
        pop_size = max(5, min(50, config['pop_size']))  # Reasonable bounds
        self.population = [self._generate_valid_thresholds() for _ in range(pop_size)]
        self.trials = [0] * pop_size
        self.limit = max(5, int(pop_size * config['limit_ratio']))  # Minimum limit
        self.best_solution, self.best_fitness = None, float('-inf')

        # Initialize best solution
        self._initialize_best()

    def _initialize_best(self):
        """Initialize the best solution from population."""
        for ind in self.population:
            fitness_val = self.fitness(ind)
            if fitness_val > self.best_fitness:
                self.best_solution = ind.copy()
                self.best_fitness = fitness_val

    def _generate_valid_thresholds(self):
        """Generate valid thresholds within histogram range."""
        if self.num_thresholds == 0:
            return []

        # Generate unique thresholds with proper spacing
        max_attempts = 100
        thresholds = set()

        while len(thresholds) < self.num_thresholds and len(thresholds) < 254:
            new_thresh = self.rng.randint(1, 254)
            thresholds.add(new_thresh)

            if len(thresholds) >= max_attempts:
                break

        # If we need more thresholds, add sequential ones
        if len(thresholds) < self.num_thresholds:
            needed = self.num_thresholds - len(thresholds)
            start = 10
            while len(thresholds) < self.num_thresholds and start < 245:
                thresholds.add(start)
                start += max(1, (240 - start) // needed)

        thresholds = sorted(thresholds)[:self.num_thresholds]
        return thresholds

    def fitness(self, ind):
        """Calculate fitness of an individual with error handling."""
        try:
            if len(ind) != self.num_thresholds:
                return float('-inf')

            if any(t <= 0 or t >= 255 for t in ind):
                return float('-inf')

            # Check for duplicate thresholds
            if len(set(ind)) != len(ind):
                return float('-inf')

            return self.fitness_function(self.hist, ind)
        except Exception as e:
            return float('-inf')

    def evolve(self):
        """Run the ABC optimization process with robust error handling."""
        for generation in range(self.config['max_generations']):
            success = self._run_generation(generation)

            if not success:
                print(f"Generation {generation} failed, resetting population...")
                self._reset_population()

            # Print progress every 10 generations
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness

    def _run_generation(self, generation):
        """Run a single generation with comprehensive error handling."""
        try:
            self.employed()
            self.onlooker()
            self.scout()
            return True
        except Exception as e:
            print(f"Generation {generation} error: {e}")
            return False

    def _reset_population(self):
        """Reset population while keeping the best solution."""
        best_backup = self.best_solution.copy()
        best_fitness_backup = self.best_fitness

        self.population = [self._generate_valid_thresholds() for _ in range(len(self.population))]
        self.trials = [0] * len(self.population)

        # Restore best solution
        self.best_solution = best_backup
        self.best_fitness = best_fitness_backup

        # Add best solution to population
        if self.best_solution is not None:
            self.population[0] = self.best_solution.copy()

    def employed(self):
        """Employed bee phase with error handling."""
        for i in range(len(self.population)):
            try:
                cand = self.candidate(self.population[i], i)
                cand_fitness = self.fitness(cand)
                current_fitness = self.fitness(self.population[i])

                if cand_fitness > current_fitness:
                    self.population[i] = cand
                    self.trials[i] = 0
                    if cand_fitness > self.best_fitness:
                        self.best_solution = cand.copy()
                        self.best_fitness = cand_fitness
                else:
                    self.trials[i] += 1
            except Exception as e:
                self.trials[i] += 1  # Penalize failed candidate
                continue

    def onlooker(self):
        """Onlooker bee phase with robust probability calculation."""
        try:
            fits = np.array([max(self.fitness(ind), 1e-10) for ind in self.population])

            # Handle case where all fitness values are the same
            if np.max(fits) - np.min(fits) < 1e-10:
                probs = np.ones(len(fits)) / len(fits)
            else:
                # Normalize fitness values
                fits_normalized = (fits - np.min(fits)) / (np.max(fits) - np.min(fits) + 1e-10)
                probs = fits_normalized / (np.sum(fits_normalized) + 1e-10)

            # Ensure probabilities are valid
            if np.any(np.isnan(probs)) or np.sum(probs) == 0:
                probs = np.ones(len(fits)) / len(fits)
            else:
                probs = probs / np.sum(probs)  # Re-normalize

            for _ in range(len(self.population)):
                idx = self.np_rng.choice(len(self.population), p=probs)
                try:
                    cand = self.candidate(self.population[idx], idx)
                    cand_fitness = self.fitness(cand)
                    current_fitness = self.fitness(self.population[idx])

                    if cand_fitness > current_fitness:
                        self.population[idx] = cand
                        self.trials[idx] = 0
                        if cand_fitness > self.best_fitness:
                            self.best_solution = cand.copy()
                            self.best_fitness = cand_fitness
                    else:
                        self.trials[idx] += 1
                except Exception:
                    self.trials[idx] += 1
                    continue

        except Exception as e:
            print(f"Onlooker phase error: {e}")
            # Fallback: random selection
            for _ in range(len(self.population)):
                idx = self.rng.randint(0, len(self.population))
                try:
                    cand = self.candidate(self.population[idx], idx)
                    cand_fitness = self.fitness(cand)
                    current_fitness = self.fitness(self.population[idx])

                    if cand_fitness > current_fitness:
                        self.population[idx] = cand
                        self.trials[idx] = 0
                        if cand_fitness > self.best_fitness:
                            self.best_solution = cand.copy()
                            self.best_fitness = cand_fitness
                    else:
                        self.trials[idx] += 1
                except Exception:
                    self.trials[idx] += 1
                    continue

    def scout(self):
        """Scout bee phase."""
        for i in range(len(self.population)):
            if self.trials[i] >= self.limit:
                try:
                    self.population[i] = self._generate_valid_thresholds()
                    self.trials[i] = 0
                    new_fitness = self.fitness(self.population[i])
                    if new_fitness > self.best_fitness:
                        self.best_solution = self.population[i].copy()
                        self.best_fitness = new_fitness
                except Exception:
                    continue

    def candidate(self, sol, idx):
        """Generate a candidate solution with comprehensive error handling."""
        if len(sol) == 0:
            return self._generate_valid_thresholds()

        try:
            cand = sol.copy()
            d = self.rng.randint(0, len(cand))

            if len(self.population) <= 1:
                # Random modification
                phi = self.rng.uniform(-0.5, 0.5)  # Smaller range for stability
                new_val = int(np.clip(sol[d] + phi * 10, 1, 254))
            else:
                # Choose a different individual
                available_indices = [j for j in range(len(self.population)) if j != idx]
                if not available_indices:
                    k = 0
                else:
                    k = self.rng.choice(available_indices)

                phi = self.rng.uniform(-0.5, 0.5)
                difference = sol[d] - self.population[k][d]
                new_val = int(np.clip(sol[d] + phi * difference, 1, 254))

            cand[d] = new_val

            # Ensure thresholds remain unique and sorted
            cand = sorted(set(cand))

            # If we lost thresholds due to duplication, add new ones
            while len(cand) < self.num_thresholds:
                new_thresh = self.rng.randint(1, 254)
                if new_thresh not in cand:
                    cand.append(new_thresh)
            cand = sorted(cand)[:self.num_thresholds]

            return cand

        except Exception as e:
            # Fallback: return original solution
            return sol.copy()


# ----------------- Worker Function -----------------

def process_single(filename, folder_path, num_thresholds, fitness_function, abc_config, random_seed, images_folder,
                   base_seed):
    """Process a single image with given parameters."""
    filepath = os.path.join(folder_path, filename)
    print(f"[START] {filename} | {fitness_function} | thresholds={num_thresholds}")

    # Load image
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[SKIP] Could not load {filename}")
        return None

    # Generate consistent seed based on image name, fitness function, and threshold level
    # This ensures same seed for same combination across runs
    seed_str = f"{filename}_{fitness_function}_{num_thresholds}_{base_seed}"
    run_seed = hash(seed_str) % 1000000
    random.seed(run_seed)
    np.random.seed(run_seed)

    start_time = time.time()

    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

    # Select fitness function
    if fitness_function == "renyi_2d":
        _image = image
        def fitness_func(hist, thresholds, _img=_image):
            return calculate_renyi_entropy_2d(_img, thresholds)
    else:
        fitness_func = calculate_otsu

    # Run ABC optimization
    try:
        abc = PureArtificialBeeColony(abc_config, num_thresholds, hist, fitness_func, run_seed)
        best_solution, best_fitness = abc.evolve()

        if best_solution is None or len(best_solution) != num_thresholds:
            raise ValueError("Invalid solution from ABC")

    except Exception as e:
        print(f"[ERROR] ABC failed for {filename}: {e}")
        # Simple fallback: evenly spaced thresholds
        best_solution = [int(i * 255 / (num_thresholds + 1)) for i in range(1, num_thresholds + 1)]
        best_fitness = fitness_func(hist, best_solution)

    execution_time_ms = (time.time() - start_time) * 1000

    # Apply thresholds and calculate metrics
    try:
        seg_img = apply_thresholds(image, best_solution)
        mse = calculate_mse(image, seg_img)
        psnr = calculate_psnr(mse)
        ssim_value = ssim(image, seg_img)
        uniformity_value = calculate_uniformity(seg_img)
    except Exception as e:
        print(f"[WARNING] Metric calculation failed: {e}")
        mse = 10000.0
        psnr = 0.0
        ssim_value = 0.0
        uniformity_value = 0.0
        seg_img = np.zeros_like(image)

    # Format threshold values exactly as requested
    threshold_str = f"[{', '.join(map(str, sorted(best_solution)))}]"

    print(f"[DONE] {filename} | {fitness_function} | thresholds={num_thresholds} | "
          f"fitness={best_fitness:.4f} | time={execution_time_ms:.2f}ms")

    # Save segmented image
    try:
        image_name_base = os.path.splitext(filename)[0]
        output_filename = f"{image_name_base}_{fitness_function}_{num_thresholds}_thresholds.png"
        output_image_path = os.path.join(images_folder, output_filename)
        cv2.imwrite(output_image_path, seg_img)
    except Exception as e:
        print(f"[WARNING] Could not save image: {e}")

    return {
        'image_name': filename,
        'fitness_function': fitness_function,  # Keep lowercase
        'thresholding_level': num_thresholds,
        'threshold_value': threshold_str,
        'fitness_value': format_number(best_fitness, 8),
        'SSIM': format_number(ssim_value, 6),
        'MSE': format_number(mse, 2),
        'PSNR': format_number(psnr, 2),
        'Uniformity Measure': format_number(uniformity_value, 6),
        'Seed': run_seed,
        'Execution Time (ms)': format_number(execution_time_ms, 2)
    }


# ----------------- Main Runner -----------------

def process_images_in_folder(folder_path, threshold_levels, output_path, n_jobs=1):
    """Main function to process all images in a folder."""
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_GAABC")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    if not files:
        print("No image files found.")
        return None

    print(f"Found {len(files)} images to process.")

    # Use simple default configuration
    abc_config = {
        'pop_size': 20,
        'max_generations': 50,
        'mutation_factor': 0.5,
        'limit_ratio': 0.3
    }

    print("Using default ABC configuration:", abc_config)

    # Create tasks in the specific order: image1-renyi_2d-all_levels, image1-otsu-all_levels, image2-renyi_2d-all_levels, etc.
    tasks = []
    base_seed = 42  # Fixed base seed for reproducibility

    for f in files:
        for func in ["renyi_2d", "otsu"]:  # Process renyi_2d first for each image, then otsu
            for t in threshold_levels:
                tasks.append((f, folder_path, t, func, abc_config, 42, images_folder, base_seed))

    print(f"Processing {len(tasks)} tasks sequentially...")

    # Sequential processing
    results = []
    for i, task in enumerate(tasks):
        try:
            result = process_single(*task)
            if result is not None:
                results.append(result)
            print(f"Progress: {i + 1}/{len(tasks)}")
        except Exception as e:
            print(f"Task {i + 1} failed: {e}")
            # Continue with next task

    if not results:
        print("No results obtained.")
        return None

    # Create DataFrame
    df = pd.DataFrame(results)

    # Use exact column names as requested
    column_order = [
        'image_name', 'fitness_function', 'thresholding_level', 'threshold_value',
        'fitness_value', 'SSIM', 'MSE', 'PSNR', 'Uniformity Measure',
        'Seed', 'Execution Time (ms)'
    ]

    df = df[column_order]

    # Create a custom sorting key for fitness_function to ensure "renyi_2d" comes before "otsu"
    def fitness_function_order(func):
        return 0 if func == "renyi_2d" else 1

    # Sort the results to match your desired order:
    # 1. First by image name
    # 2. Then by fitness function (renyi_2d first, then otsu) using custom order
    # 3. Then by thresholding level (ascending)
    df_sorted = df.copy()
    df_sorted['fitness_order'] = df_sorted['fitness_function'].apply(fitness_function_order)
    df_sorted = df_sorted.sort_values(['image_name', 'fitness_order', 'thresholding_level'])
    df_sorted = df_sorted.drop('fitness_order', axis=1)
    df_sorted = df_sorted.reset_index(drop=True)

    # Save to Excel
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "GAABC_results.xlsx")

    # Create Excel writer with formatting options
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_sorted.to_excel(writer, sheet_name='Results', index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets['Results']
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    # Also save as CSV for easy viewing
    csv_file = os.path.join(output_path, "GAABC_results.csv")
    df_sorted.to_csv(csv_file, index=False, sep='\t')  # Tab-separated

    print(f"Results saved to {output_file} and {csv_file}")
    print(f"Processed {len(results)} image configurations")

    # Print the results in the exact format for verification
    print("\nResults preview (first 12 rows):")
    print("-" * 120)
    print(
        f"{'image_name':<15} {'fitness_function':<15} {'thresholding_level':<18} {'threshold_value':<15} {'fitness_value':<15} {'SSIM':<10} {'MSE':<10} {'PSNR':<8} {'Uniformity Measure':<18} {'Seed':<10} {'Execution Time (ms)':<15}")
    print("-" * 120)

    for i, row in df_sorted.head(12).iterrows():
        print(
            f"{row['image_name']:<15} {row['fitness_function']:<15} {row['thresholding_level']:<18} {row['threshold_value']:<15} {row['fitness_value']:<15} {row['SSIM']:<10} {row['MSE']:<10} {row['PSNR']:<8} {row['Uniformity Measure']:<18} {row['Seed']:<10} {row['Execution Time (ms)']:<15}")

    return df_sorted


# ----------------- Main Execution -----------------

if __name__ == "__main__":
    folder = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images"
    output = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images\results"
    levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # Added level 4 to match your example

    df = process_images_in_folder(folder, levels, output, n_jobs=1)

    if df is not None:
        print("\nProcessing completed successfully!")
        print(f"Total results: {len(df)}")

        # Show grouping by image and fitness function
        print("\nGrouping structure:")
        grouped = df.groupby(['image_name', 'fitness_function']).size()
        for (image, func), count in grouped.items():
            print(f"{image} - {func}: {count} threshold levels")

    else:
        print("Processing failed!")