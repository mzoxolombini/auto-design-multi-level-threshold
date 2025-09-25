import os
import cv2
import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed
from scipy.ndimage import correlate
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import mean_squared_error
import math
import time


# ===============================
# Corrected Utility Functions
# ===============================

def calculate_histogram(image):
    """Compute normalized histogram of grayscale image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0:
        return np.zeros_like(hist)
    return hist / total


def kapur_entropy(hist, thresholds):
    """Compute Kapur's entropy for given thresholds (corrected)."""
    if len(thresholds) == 0:
        return 0.0

    thresholds = sorted(thresholds)
    bins = [0] + thresholds + [len(hist)]
    total_entropy = 0.0

    # Precompute cumulative sums
    cumulative = np.cumsum(hist)

    for i in range(len(bins) - 1):
        start = bins[i]
        end = bins[i + 1]

        if start >= end:  # Skip invalid ranges
            continue

        class_hist = hist[start:end]
        prob_sum = cumulative[end - 1] - (cumulative[start - 1] if start > 0 else 0)

        if prob_sum > 1e-10:
            # Remove zero probabilities to avoid log(0)
            class_probs = class_hist / prob_sum
            class_probs_nonzero = class_probs[class_probs > 1e-10]

            if len(class_probs_nonzero) > 0:
                entropy_val = -np.sum(class_probs_nonzero * np.log(class_probs_nonzero))
                total_entropy += entropy_val

    return total_entropy


def otsu_between_class_variance(hist, thresholds):
    """Compute Otsu's between-class variance for given thresholds (corrected)."""
    if len(thresholds) == 0:
        return 0.0

    thresholds = sorted(thresholds)
    bins = [0] + thresholds + [len(hist)]

    total_weight = np.sum(hist)
    if total_weight == 0:
        return 0.0

    global_mean = np.sum(np.arange(len(hist)) * hist) / total_weight
    between_class_variance = 0.0

    cumulative = np.cumsum(hist)
    mean_cumulative = np.cumsum(np.arange(len(hist)) * hist)

    for i in range(len(bins) - 1):
        start = bins[i]
        end = bins[i + 1]

        if start >= end:  # Skip invalid ranges
            continue

        w = cumulative[end - 1] - (cumulative[start - 1] if start > 0 else 0)

        if w > 1e-10:
            m_sum = mean_cumulative[end - 1] - (mean_cumulative[start - 1] if start > 0 else 0)
            m = m_sum / w
            between_class_variance += w * (m - global_mean) ** 2

    return between_class_variance / total_weight


def calculate_ssim(original, segmented):
    """Calculate Structural Similarity Index (SSIM) between original and segmented images (corrected)."""
    # Ensure images have the same dimensions
    min_shape = (min(original.shape[0], segmented.shape[0]),
                 min(original.shape[1], segmented.shape[1]))

    if original.shape != min_shape:
        original = cv2.resize(original, min_shape[::-1])
    if segmented.shape != min_shape:
        segmented = cv2.resize(segmented, min_shape[::-1])

    # Use data_range=255 for 8-bit images
    data_range = 255 if original.max() > 1.0 else 1.0
    return ssim_skimage(original, segmented, data_range=data_range)


def calculate_mse(original, segmented):
    """Calculate Mean Squared Error (MSE) between original and segmented images."""
    # Ensure images have the same dimensions
    min_shape = (min(original.shape[0], segmented.shape[0]),
                 min(original.shape[1], segmented.shape[1]))

    if original.shape != min_shape:
        original = cv2.resize(original, min_shape[::-1])
    if segmented.shape != min_shape:
        segmented = cv2.resize(segmented, min_shape[::-1])

    return mean_squared_error(original, segmented)


def calculate_psnr(original, segmented):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between original and segmented images (corrected)."""
    mse = calculate_mse(original, segmented)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 10 * math.log10((max_pixel ** 2) / mse)  # Corrected formula


def calculate_uniformity_measure(segmented, thresholds):
    """Calculate Uniformity Measure for segmented image (corrected)."""
    if len(thresholds) == 0:
        return 0.0

    uniformity = 0.0
    total_pixels = segmented.size
    if total_pixels == 0:
        return 0.0

    # Sort thresholds and add boundaries
    sorted_thresholds = sorted(thresholds)
    region_boundaries = [0] + sorted_thresholds + [255]

    for i in range(len(region_boundaries) - 1):
        lower = region_boundaries[i]
        upper = region_boundaries[i + 1]

        # Create mask for current region
        mask = (segmented >= lower) & (segmented <= upper)
        region_pixels = np.sum(mask)

        if region_pixels > 0:
            # Calculate mean intensity in the region
            region_mean = np.mean(segmented[mask])

            # Calculate variance in the region
            variance = np.var(segmented[mask])

            # Calculate maximum possible variance for this intensity range
            max_variance = ((upper - lower) ** 2) / 4.0  # Maximum variance for uniform distribution

            if max_variance > 0:
                # Uniformity measure: 1 - (actual_variance / max_possible_variance)
                region_uniformity = 1.0 - (variance / max_variance)
                region_uniformity = max(0.0, min(1.0, region_uniformity))  # Clip to [0,1]

                # Weight by region size
                uniformity += region_uniformity * (region_pixels / total_pixels)

    return uniformity


def apply_thresholds(image, thresholds):
    """Apply thresholds to image to create segmented image (corrected)."""
    if len(thresholds) == 0:
        return np.zeros_like(image)

    # Sort thresholds
    thresholds = sorted(thresholds)

    # Create segmented image
    segmented = np.zeros_like(image, dtype=np.uint8)

    # First region: 0 to first threshold
    mask = image <= thresholds[0]
    segmented[mask] = 0

    # Middle regions
    for i in range(1, len(thresholds)):
        mask = (image > thresholds[i - 1]) & (image <= thresholds[i])
        segmented[mask] = i

    # Last region: last threshold to 255
    mask = image > thresholds[-1]
    segmented[mask] = len(thresholds)

    # Normalize to 0-255 for better visualization
    max_class = len(thresholds)
    if max_class > 0:
        segmented = (segmented * (255 // max_class)).astype(np.uint8)

    return segmented


def format_number(value, decimals=8):
    """Format number with specified decimal places."""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if np.isinf(value):
        return "Infinity"
    fmt = f"{value:.{decimals}f}"
    return fmt.replace('.', ',')


# ===============================
# Corrected Artificial Bee Colony
# ===============================

class PureArtificialBeeColony:
    def __init__(self, config, num_thresholds, hist, fitness_func, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.pop_size = config['pop_size']
        self.max_generations = config['max_generations']
        self.limit = int(config['limit_ratio'] * self.pop_size)
        self.mutation_factor = config['mutation_factor']
        self.num_thresholds = num_thresholds
        self.hist = hist
        self.fitness_func = fitness_func

        # Initialize population with unique thresholds
        self.population = []
        for _ in range(self.pop_size):
            individual = np.sort(np.random.choice(range(1, 255), num_thresholds, replace=False))
            self.population.append(individual)

        self.fitness = [self.fitness_func(self.hist, ind) for ind in self.population]
        self.trial_counters = np.zeros(self.pop_size)
        self.best_solution = self.population[np.argmax(self.fitness)].copy()
        self.best_fitness = max(self.fitness)

    def evolve(self):
        """Run ABC optimization."""
        for generation in range(self.max_generations):
            # Employed bee phase
            for i in range(self.pop_size):
                # Select a random different bee
                k = np.random.choice([j for j in range(self.pop_size) if j != i])

                # Generate candidate solution
                candidate = self.population[i].copy()
                d = np.random.randint(0, self.num_thresholds)  # Random dimension
                phi = np.random.uniform(-1, 1) * self.mutation_factor

                candidate[d] = int(np.clip(
                    candidate[d] + phi * (candidate[d] - self.population[k][d]),
                    1, 254  # Keep within valid range
                ))
                candidate.sort()

                candidate_fitness = self.fitness_func(self.hist, candidate)

                # Greedy selection
                if candidate_fitness > self.fitness[i]:
                    self.population[i] = candidate
                    self.fitness[i] = candidate_fitness
                    self.trial_counters[i] = 0

                    # Update global best
                    if candidate_fitness > self.best_fitness:
                        self.best_solution = candidate.copy()
                        self.best_fitness = candidate_fitness
                else:
                    self.trial_counters[i] += 1

            # Onlooker bee phase (simplified - same as employed phase in basic ABC)
            for i in range(self.pop_size):
                # Select based on fitness probability
                fitness_probs = np.array(self.fitness) - min(self.fitness) + 1e-10
                fitness_probs = fitness_probs / fitness_probs.sum()
                i_onlooker = np.random.choice(range(self.pop_size), p=fitness_probs)

                k = np.random.choice([j for j in range(self.pop_size) if j != i_onlooker])
                candidate = self.population[i_onlooker].copy()
                d = np.random.randint(0, self.num_thresholds)
                phi = np.random.uniform(-1, 1) * self.mutation_factor

                candidate[d] = int(np.clip(
                    candidate[d] + phi * (candidate[d] - self.population[k][d]),
                    1, 254
                ))
                candidate.sort()

                candidate_fitness = self.fitness_func(self.hist, candidate)

                if candidate_fitness > self.fitness[i_onlooker]:
                    self.population[i_onlooker] = candidate
                    self.fitness[i_onlooker] = candidate_fitness
                    self.trial_counters[i_onlooker] = 0

                    if candidate_fitness > self.best_fitness:
                        self.best_solution = candidate.copy()
                        self.best_fitness = candidate_fitness
                else:
                    self.trial_counters[i_onlooker] += 1

            # Scout bee phase
            for i in range(self.pop_size):
                if self.trial_counters[i] > self.limit:
                    self.population[i] = np.sort(np.random.choice(range(1, 255), self.num_thresholds, replace=False))
                    self.fitness[i] = self.fitness_func(self.hist, self.population[i])
                    self.trial_counters[i] = 0

                    if self.fitness[i] > self.best_fitness:
                        self.best_solution = self.population[i].copy()
                        self.best_fitness = self.fitness[i]

            # Print progress every 10 generations
            if generation % 10 == 0:
                print(f"ABC Generation {generation}: Best fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness


# ===============================
# Corrected Differential Evolution Configurator
# ===============================

class DEABCConfigurator:
    def __init__(self, de_pop_size=20, de_generations=20, F=0.5, CR=0.9, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.de_pop_size = de_pop_size
        self.de_generations = de_generations
        self.F = F
        self.CR = CR

    def optimize_abc_params(self, num_thresholds, image, fitness_func, seed=None):
        """Optimize ABC parameters using Differential Evolution."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        hist = calculate_histogram(image)

        # Parameter bounds
        bounds = {
            'pop_size': (20, 100),
            'max_generations': (50, 200),
            'mutation_factor': (0.1, 1.0),
            'limit_ratio': (0.1, 0.5)
        }

        # Initialize population
        population = []
        for _ in range(self.de_pop_size):
            individual = {
                'pop_size': random.randint(bounds['pop_size'][0], bounds['pop_size'][1]),
                'max_generations': random.randint(bounds['max_generations'][0], bounds['max_generations'][1]),
                'mutation_factor': random.uniform(bounds['mutation_factor'][0], bounds['mutation_factor'][1]),
                'limit_ratio': random.uniform(bounds['limit_ratio'][0], bounds['limit_ratio'][1])
            }
            population.append(individual)

        # Evaluate initial population
        fitness = [self.evaluate(ind, num_thresholds, hist, fitness_func, seed)
                   for ind in population]

        # Differential Evolution optimization
        for gen in range(self.de_generations):
            for i in range(self.de_pop_size):
                # Select three distinct individuals
                indices = [j for j in range(self.de_pop_size) if j != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Create trial individual
                trial = {}
                for key in population[i].keys():
                    if random.random() < self.CR:
                        if key in ['pop_size', 'max_generations']:
                            # Integer parameters
                            val = int(round(a[key] + self.F * (b[key] - c[key])))
                            val = max(bounds[key][0], min(bounds[key][1], val))
                        else:
                            # Continuous parameters
                            val = a[key] + self.F * (b[key] - c[key])
                            val = max(bounds[key][0], min(bounds[key][1], val))
                        trial[key] = val
                    else:
                        trial[key] = population[i][key]

                # Evaluate trial
                trial_fitness = self.evaluate(trial, num_thresholds, hist, fitness_func, seed)

                # Selection
                if trial_fitness > fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Print progress
            if gen % 5 == 0:
                best_fit = max(fitness)
                print(f"DE Generation {gen}: Best fitness = {best_fit:.6f}")

        # Return best individual
        best_index = np.argmax(fitness)
        return population[best_index]

    def evaluate(self, config, num_thresholds, hist, fitness_func, seed=None):
        """Evaluate ABC configuration."""
        try:
            abc = PureArtificialBeeColony(config, num_thresholds, hist, fitness_func, seed)
            _, best_fitness = abc.evolve()
            return best_fitness
        except Exception as e:
            print(f"Evaluation error: {e}")
            return float('-inf')


# ===============================
# Corrected Image Processing
# ===============================

def process_single(file_name, folder_path, num_thresholds, func_name, de_configurator, seed=None, images_folder=None):
    """Process a single image with given parameters."""
    try:
        # Generate unique seed for this run
        run_seed = (seed + hash(file_name) + num_thresholds + hash(func_name)) % 1000000
        np.random.seed(run_seed)
        random.seed(run_seed)

        # Start timing
        start_time = time.time()

        # Load image
        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[ERROR] Could not load image: {file_name}")
            return None

        print(f"[START] Processing {file_name} | {func_name} | thresholds={num_thresholds}")

        # Calculate histogram
        hist = calculate_histogram(image)

        # Select fitness function
        fitness_function = kapur_entropy if func_name == "kapur" else otsu_between_class_variance

        # Optimize ABC parameters using DE
        optimal_config = de_configurator.optimize_abc_params(num_thresholds, image, fitness_function, run_seed)

        # Run ABC with optimized parameters
        abc = PureArtificialBeeColony(optimal_config, num_thresholds, hist, fitness_function, run_seed)
        best_thresholds, best_score = abc.evolve()

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Apply thresholds and calculate metrics
        segmented_image = apply_thresholds(image, best_thresholds)

        ssim_value = calculate_ssim(image, segmented_image)
        mse_value = calculate_mse(image, segmented_image)
        psnr_value = calculate_psnr(image, segmented_image)
        uniformity = calculate_uniformity_measure(segmented_image, best_thresholds)
        threshold_value_str = "[" + ", ".join(map(str, best_thresholds)) + "]"

        # Save segmented image
        if images_folder:
            image_name_base = os.path.splitext(file_name)[0]
            output_filename = f"{image_name_base}_{func_name}_{num_thresholds}_thresholds.png"
            output_image_path = os.path.join(images_folder, output_filename)
            cv2.imwrite(output_image_path, segmented_image)

        print(f"[DONE] {file_name} | {func_name} | thresholds={num_thresholds} | "
              f"fitness={best_score:.4f} | time={execution_time_ms:.2f}ms")

        return {
            "image_name": file_name,
            "fitness_function": func_name.capitalize(),
            "thresholding_level": num_thresholds,
            "threshold_value": threshold_value_str,
            "fitness_value": format_number(best_score, 8),
            "SSIM": format_number(ssim_value, 7),
            "MSE": format_number(mse_value, 7),
            "PSNR": format_number(psnr_value, 7),
            "Uniformity Measure": format_number(uniformity, 7),
            "Random Seed": run_seed,
            "Execution Time (ms)": format_number(execution_time_ms, 2)
        }

    except Exception as e:
        print(f"[ERROR] Processing {file_name} | {func_name}: {str(e)}")
        return {
            "image_name": file_name,
            "fitness_function": func_name,
            "error": str(e)
        }


def process_images_in_folder(folder_path, threshold_levels, output_file, n_jobs=-1, seed=None):
    """Main function to process all images in a folder."""
    # Create output directories
    output_path = os.path.dirname(output_file)
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_DEABC")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # Get image files
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    if not files:
        print("No image files found in the specified folder.")
        return pd.DataFrame()

    print(f"Found {len(files)} images to process.")

    # Set random seed
    if seed is None:
        seed = int(time.time() * 1000) % 1000000
    np.random.seed(seed)
    random.seed(seed)
    print(f"Using random seed: {seed}")

    # Initialize DE configurator
    de_configurator = DEABCConfigurator(seed=seed)

    # Create tasks for parallel processing
    tasks = []
    for f in files:
        for t in threshold_levels:
            for func in ["kapur", "otsu"]:
                tasks.append((f, folder_path, t, func, de_configurator, seed, images_folder))

    print(f"Processing {len(tasks)} tasks with {n_jobs} workers...")

    # Process images in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single)(*task) for task in tasks
    )

    # Filter results
    valid_results = [r for r in results if r is not None and 'error' not in r]
    error_results = [r for r in results if r is not None and 'error' in r]

    if valid_results:
        # Create DataFrame
        df = pd.DataFrame(valid_results)

        # Define column order
        column_order = [
            'image_name',
            'fitness_function',
            'thresholding_level',
            'threshold_value',
            'fitness_value',
            'SSIM',
            'MSE',
            'PSNR',
            'Uniformity Measure',
            'Random Seed',
            'Execution Time (ms)'
        ]

        df = df[column_order]

        # Save results
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
        print(f"Successfully processed {len(valid_results)} configurations")
    else:
        print("No valid results to save")
        df = pd.DataFrame()

    # Report errors
    if error_results:
        print(f"\nErrors encountered in {len(error_results)} processing tasks:")
        for error_result in error_results:
            print(f"  {error_result['image_name']} - {error_result['fitness_function']}: {error_result['error']}")

    return df


# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    # Configuration
    folder = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images"
    levels = [2, 3, 4, 5]  # Reduced for testing, can expand to [2, 3, 4, 5, 6, 7, 8, 9, 10]
    output_dir = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images\results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "DEABC_results.xlsx")

    # Process images
    df = process_images_in_folder(folder, levels, output_file, n_jobs=-1, seed=None)

    if not df.empty:
        print("\nProcessing completed successfully!")
        print(f"Total results: {len(df)}")

        # Verify metric ranges
        print(f"SSIM value range: [{df['SSIM'].astype(float).min():.6f} - {df['SSIM'].astype(float).max():.6f}]")
        print(f"PSNR value range: [{df['PSNR'].astype(float).min():.2f} - {df['PSNR'].astype(float).max():.2f}] dB")
        print(f"MSE value range: [{df['MSE'].astype(float).min():.2f} - {df['MSE'].astype(float).max():.2f}]")
        print(
            f"Uniformity value range: [{df['Uniformity Measure'].astype(float).min():.6f} - {df['Uniformity Measure'].astype(float).max():.6f}]")
    else:
        print("Processing completed with errors!")