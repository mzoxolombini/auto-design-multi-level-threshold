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
    """Calculate normalized histogram of an image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0:
        return np.zeros_like(hist)
    return hist / total


def kapur_entropy(hist, thresholds):
    """Calculate Kapur's entropy for multilevel thresholding (corrected)."""
    if len(thresholds) == 0:
        return 0.0

    thresholds = sorted(thresholds)
    bins = [0] + thresholds + [256]  # Changed from len(hist) to 256
    total_entropy = 0.0

    # Calculate cumulative sums for efficiency
    cumulative_hist = np.cumsum(hist)

    for i in range(len(bins) - 1):
        start = bins[i]
        end = bins[i + 1]

        if start >= end:  # Skip invalid ranges
            continue

        # Ensure indices are integers
        start_idx = int(start)
        end_idx = int(end)

        class_hist = hist[start_idx:end_idx]
        class_sum = cumulative_hist[end_idx - 1] if end_idx > 0 else 0
        if start_idx > 0:
            class_sum -= cumulative_hist[start_idx - 1]

        if class_sum > 1e-10:
            # Remove zero probabilities to avoid log(0)
            class_probs = class_hist / class_sum
            class_probs_nonzero = class_probs[class_probs > 1e-10]

            if len(class_probs_nonzero) > 0:
                entropy_val = -np.sum(class_probs_nonzero * np.log(class_probs_nonzero))
                total_entropy += entropy_val

    return total_entropy


def otsu_between_class_variance(hist, thresholds):
    """Calculate Otsu's between-class variance for multilevel thresholding (corrected)."""
    if len(thresholds) == 0:
        return 0.0

    thresholds = sorted(thresholds)
    bins = [0] + thresholds + [256]  # Changed from len(hist) to 256

    total_weight = np.sum(hist)
    if total_weight == 0:
        return 0.0

    global_mean = np.sum(np.arange(256) * hist) / total_weight  # Fixed: use 256 instead of len(hist)
    between_class_variance = 0.0

    # Calculate cumulative sums for efficiency
    cumulative_hist = np.cumsum(hist)
    cumulative_mean = np.cumsum(np.arange(256) * hist)  # Fixed: use 256 instead of len(hist)

    for i in range(len(bins) - 1):
        start = bins[i]
        end = bins[i + 1]

        if start >= end:  # Skip invalid ranges
            continue

        # Ensure indices are integers
        start_idx = int(start)
        end_idx = int(end)

        class_weight = cumulative_hist[end_idx - 1] if end_idx > 0 else 0
        if start_idx > 0:
            class_weight -= cumulative_hist[start_idx - 1]

        if class_weight > 1e-10:
            class_mean_sum = cumulative_mean[end_idx - 1] if end_idx > 0 else 0
            if start_idx > 0:
                class_mean_sum -= cumulative_mean[start_idx - 1]
            class_mean = class_mean_sum / class_weight
            between_class_variance += class_weight * (class_mean - global_mean) ** 2

    return between_class_variance / total_weight


def calculate_ssim(original, segmented):
    """Calculate SSIM using scikit-image (corrected)."""
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
    """Calculate Mean Squared Error."""
    # Ensure images have the same dimensions
    min_shape = (min(original.shape[0], segmented.shape[0]),
                 min(original.shape[1], segmented.shape[1]))

    if original.shape != min_shape:
        original = cv2.resize(original, min_shape[::-1])
    if segmented.shape != min_shape:
        segmented = cv2.resize(segmented, min_shape[::-1])

    return mean_squared_error(original, segmented)


def calculate_psnr(original, segmented):
    """Calculate PSNR (corrected formula)."""
    mse = calculate_mse(original, segmented)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 10 * math.log10((max_pixel ** 2) / mse)  # Corrected formula


def calculate_uniformity_measure(segmented, thresholds):
    """Calculate uniformity measure (corrected)."""
    if len(thresholds) == 0:
        return 0.0

    uniformity = 0.0
    total_pixels = segmented.size
    if total_pixels == 0:
        return 0.0

    sorted_thresholds = sorted(thresholds)

    # Create intensity classes based on thresholds
    intensity_levels = len(sorted_thresholds) + 1
    class_means = []
    class_pixels = []

    # Calculate statistics for each class
    for i in range(intensity_levels):
        if i == 0:
            mask = segmented <= sorted_thresholds[0] if len(sorted_thresholds) > 0 else segmented <= 0
        elif i == len(sorted_thresholds):
            mask = segmented > sorted_thresholds[-1]
        else:
            mask = (segmented > sorted_thresholds[i - 1]) & (segmented <= sorted_thresholds[i])

        class_pixel_count = np.sum(mask)
        class_pixels.append(class_pixel_count)

        if class_pixel_count > 0:
            class_mean = np.mean(segmented[mask])
            class_means.append(class_mean)
        else:
            class_means.append(0.0)

    # Calculate uniformity based on variance within classes
    total_uniformity = 0.0
    for i, pixel_count in enumerate(class_pixels):
        if pixel_count > 0:
            if i == 0:
                class_range = sorted_thresholds[0] if len(sorted_thresholds) > 0 else 1
            elif i == len(sorted_thresholds):
                class_range = 255 - sorted_thresholds[-1]
            else:
                class_range = sorted_thresholds[i] - sorted_thresholds[i - 1]

            if class_range > 0:
                # Calculate variance for this class
                if i == 0:
                    mask = segmented <= sorted_thresholds[0] if len(sorted_thresholds) > 0 else segmented <= 0
                elif i == len(sorted_thresholds):
                    mask = segmented > sorted_thresholds[-1]
                else:
                    mask = (segmented > sorted_thresholds[i - 1]) & (segmented <= sorted_thresholds[i])

                variance = np.var(segmented[mask]) if np.sum(mask) > 0 else 0
                class_uniformity = 1 - (variance / (class_range ** 2)) if class_range > 0 else 0
                class_uniformity = max(0, min(1, class_uniformity))  # Clip to [0,1]
                total_uniformity += class_uniformity * (pixel_count / total_pixels)

    return total_uniformity


def apply_thresholds(image, thresholds):
    """Apply multiple thresholds to segment an image (corrected)."""
    if len(thresholds) == 0:
        return np.zeros_like(image)

    thresholds = sorted(thresholds)
    segmented = np.zeros_like(image, dtype=np.uint8)

    # First class: values <= first threshold
    segmented[image <= thresholds[0]] = 0

    # Middle classes
    for i in range(1, len(thresholds)):
        mask = (image > thresholds[i - 1]) & (image <= thresholds[i])
        segmented[mask] = i

    # Last class: values > last threshold
    segmented[image > thresholds[-1]] = len(thresholds)

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
# Genetic Algorithm with Elitism (Corrected)
# ===============================

class GeneticAlgorithm:
    def __init__(self, config, num_thresholds, hist, fitness_func, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.pop_size = config['pop_size']
        self.max_generations = config['max_generations']
        self.crossover_rate = config['crossover_rate']
        self.mutation_rate = config['mutation_rate']
        self.num_thresholds = num_thresholds
        self.hist = hist
        self.fitness_func = fitness_func

        # Fitness cache to avoid duplicate evaluations
        self.fitness_cache = {}

        # Initialize population with unique thresholds
        self.population = []
        for _ in range(self.pop_size):
            individual = np.sort(np.random.choice(range(1, 255), num_thresholds, replace=False))
            self.population.append(individual)

        self.fitness_values = [self.evaluate(ind) for ind in self.population]

    def evaluate(self, individual):
        """Evaluate fitness of an individual."""
        key = tuple(individual)
        if key not in self.fitness_cache:
            try:
                self.fitness_cache[key] = self.fitness_func(self.hist, individual)
            except:
                self.fitness_cache[key] = float('-inf')  # Invalid solution
        return self.fitness_cache[key]

    def select_parent(self):
        """Tournament selection."""
        idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
        return self.population[idx1] if self.fitness_values[idx1] > self.fitness_values[idx2] else self.population[idx2]

    def crossover(self, parent1, parent2):
        """Single-point crossover."""
        if np.random.rand() < self.crossover_rate and self.num_thresholds > 1:
            point = np.random.randint(1, self.num_thresholds)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return np.sort(child1), np.sort(child2)
        return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        """Random mutation."""
        mutated = individual.copy()
        for i in range(self.num_thresholds):
            if np.random.rand() < self.mutation_rate:
                # Mutate within valid range, ensuring uniqueness
                new_val = np.random.randint(1, 255)
                while new_val in mutated:
                    new_val = np.random.randint(1, 255)
                mutated[i] = new_val
        return np.sort(mutated)

    def evolve(self):
        """Run genetic algorithm evolution."""
        best_idx = np.argmax(self.fitness_values)
        best_individual = self.population[best_idx].copy()
        best_fitness = self.fitness_values[best_idx]

        for generation in range(self.max_generations):
            # Create new offspring
            offspring = []
            offspring_fitness = []

            for _ in range(self.pop_size // 2):
                parent1 = self.select_parent()
                parent2 = self.select_parent()
                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                offspring.extend([child1, child2])
                offspring_fitness.extend([self.evaluate(child1), self.evaluate(child2)])

            # Combine population and offspring
            combined_pop = self.population + offspring
            combined_fitness = self.fitness_values + offspring_fitness

            # Select best individuals for next generation
            indices = np.argsort(combined_fitness)[-self.pop_size:]
            self.population = [combined_pop[i] for i in indices]
            self.fitness_values = [combined_fitness[i] for i in indices]

            # Update best solution
            current_best_idx = np.argmax(self.fitness_values)
            if self.fitness_values[current_best_idx] > best_fitness:
                best_fitness = self.fitness_values[current_best_idx]
                best_individual = self.population[current_best_idx].copy()

            # Print progress every 10 generations
            if generation % 10 == 0:
                print(f"GA Generation {generation}: Best fitness = {best_fitness:.6f}")

        return best_individual, best_fitness


# ===============================
# Differential Evolution Configurator (Corrected)
# ===============================

class DEGAConfigurator:
    def __init__(self, de_pop_size=20, de_generations=20, F=0.5, CR=0.9, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.de_pop_size = de_pop_size
        self.de_generations = de_generations
        self.F = F
        self.CR = CR

    def optimize_ga_params(self, num_thresholds, image, fitness_func, seed=None):
        """Optimize GA parameters using Differential Evolution."""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        hist = calculate_histogram(image)

        # Parameter bounds
        bounds = {
            'pop_size': (20, 100),
            'max_generations': (50, 200),
            'crossover_rate': (0.6, 0.95),
            'mutation_rate': (0.01, 0.3)
        }

        # Initialize population
        population = []
        for _ in range(self.de_pop_size):
            individual = {
                'pop_size': random.randint(bounds['pop_size'][0], bounds['pop_size'][1]),
                'max_generations': random.randint(bounds['max_generations'][0], bounds['max_generations'][1]),
                'crossover_rate': random.uniform(bounds['crossover_rate'][0], bounds['crossover_rate'][1]),
                'mutation_rate': random.uniform(bounds['mutation_rate'][0], bounds['mutation_rate'][1])
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
        """Evaluate GA configuration."""
        try:
            ga = GeneticAlgorithm(config, num_thresholds, hist, fitness_func, seed)
            _, best_fitness = ga.evolve()
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

        # Optimize GA parameters using DE
        optimal_config = de_configurator.optimize_ga_params(num_thresholds, image, fitness_function, run_seed)

        # Run GA with optimized parameters
        ga = GeneticAlgorithm(optimal_config, num_thresholds, hist, fitness_function, run_seed)
        best_thresholds, best_score = ga.evolve()

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
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_DEGA")
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
    de_configurator = DEGAConfigurator(seed=seed)

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
    folder = r"C:\Users\mzoxo\OneDrive\Documents\test_images"
    levels = [2, 3, 4]  # Reduced for testing, can expand to [2, 3, 4, 5, 6, 7, 8, 9, 10]
    output_dir = r"C:\Users\mzoxo\OneDrive\Documents\test_images\results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "DEGA_results.xlsx")

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