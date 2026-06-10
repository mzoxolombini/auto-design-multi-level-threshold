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


def calculate_kapur_entropy(image, thresholds):
    """Kapur's Entropy objective function for multilevel thresholding.
    Maximises the sum of class entropies across threshold segments.
    """
    thresholds = sorted(thresholds)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist / total
    boundaries = [0] + thresholds + [256]
    total_entropy = 0.0
    for i in range(len(boundaries) - 1):
        class_prob = prob[boundaries[i]:boundaries[i + 1]]
        class_sum = class_prob.sum()
        if class_sum > 1e-12:
            p_norm = class_prob / class_sum
            p_nonzero = p_norm[p_norm > 1e-12]
            total_entropy += float(-np.sum(p_nonzero * np.log(p_nonzero)))
    return total_entropy

def calculate_otsu(hist, thresholds):
    """Otsu's between-class variance for multilevel thresholding (unnormalised)."""
    thresholds = sorted(thresholds)
    total = np.sum(hist)
    if total == 0:
        return 0.0
    probs = hist / total
    thresholds_ext = [0] + thresholds + [len(hist)]
    global_mean = np.sum(probs * np.arange(len(hist)))
    between_class_variance = 0.0
    for i in range(1, len(thresholds_ext)):
        class_probs = probs[thresholds_ext[i - 1]:thresholds_ext[i]]
        class_sum = np.sum(class_probs)
        if class_sum > 0:
            class_mean = (np.sum(class_probs * np.arange(thresholds_ext[i - 1], thresholds_ext[i]))
                          / class_sum)
            between_class_variance += class_sum * (class_mean - global_mean) ** 2
    return float(between_class_variance)


def calculate_ssim(original, segmented):
    """Calculate SSIM using scikit-image."""
    min_shape = (min(original.shape[0], segmented.shape[0]),
                 min(original.shape[1], segmented.shape[1]))

    if original.shape != min_shape:
        original = cv2.resize(original, min_shape[::-1])
    if segmented.shape != min_shape:
        segmented = cv2.resize(segmented, min_shape[::-1])

    data_range = 255 if original.max() > 1.0 else 1.0
    return ssim_skimage(original, segmented, data_range=data_range)


def calculate_mse(original, segmented):
    """Calculate Mean Squared Error."""
    min_shape = (min(original.shape[0], segmented.shape[0]),
                 min(original.shape[1], segmented.shape[1]))

    if original.shape != min_shape:
        original = cv2.resize(original, min_shape[::-1])
    if segmented.shape != min_shape:
        segmented = cv2.resize(segmented, min_shape[::-1])

    return mean_squared_error(original, segmented)


def calculate_psnr(original, segmented):
    """Calculate PSNR."""
    mse = calculate_mse(original, segmented)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 10 * math.log10((max_pixel ** 2) / mse)


def calculate_uniformity(image):
    """Histogram-based uniformity measure of the segmented image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    s = hist.sum()
    if s == 0:
        return 0.0
    hist_normalised = hist / s
    return float(np.sum(hist_normalised ** 2))


def apply_thresholds(image, thresholds):
    """Apply multiple thresholds using per-segment midpoint values."""
    thresholds = sorted(thresholds)
    segmented = np.zeros_like(image, dtype=np.uint8)
    bounds = [0] + thresholds + [256]
    for i in range(len(bounds) - 1):
        mask = (image >= bounds[i]) & (image < bounds[i + 1])
        midpoint = (bounds[i] + bounds[i + 1]) // 2
        segmented[mask] = midpoint
    return segmented


def format_number(value, decimals=8):
    """Format number with specified decimal places."""
    if isinstance(value, (int, np.integer)):
        return str(value)
    if np.isinf(value):
        return "Infinity"
    if np.isnan(value):
        return "NaN"
    return f"{value:.{decimals}f}"


# ===============================
# Genetic Algorithm with Elitism
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

        self.fitness_cache = {}
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
            except Exception as e:
                print(f"Fitness evaluation error: {e}")
                self.fitness_cache[key] = float('-inf')
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

            combined_pop = self.population + offspring
            combined_fitness = self.fitness_values + offspring_fitness

            indices = np.argsort(combined_fitness)[-self.pop_size:]
            self.population = [combined_pop[i] for i in indices]
            self.fitness_values = [combined_fitness[i] for i in indices]

            current_best_idx = np.argmax(self.fitness_values)
            if self.fitness_values[current_best_idx] > best_fitness:
                best_fitness = self.fitness_values[current_best_idx]
                best_individual = self.population[current_best_idx].copy()

            if generation % 10 == 0:
                print(f"GA Generation {generation}: Best fitness = {best_fitness:.6f}")

        return best_individual, best_fitness


# ===============================
# Differential Evolution Configurator
# ===============================

class DEGAConfigurator:
    def __init__(self, de_pop_size=10, de_generations=10, F=0.5, CR=0.9, seed=None):
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

        bounds = {
            'pop_size': (10, 50),
            'max_generations': (10, 50),
            'crossover_rate': (0.6, 0.95),
            'mutation_rate': (0.01, 0.3)
        }

        population = []
        for _ in range(self.de_pop_size):
            individual = {
                'pop_size': random.randint(bounds['pop_size'][0], bounds['pop_size'][1]),
                'max_generations': random.randint(bounds['max_generations'][0], bounds['max_generations'][1]),
                'crossover_rate': random.uniform(bounds['crossover_rate'][0], bounds['crossover_rate'][1]),
                'mutation_rate': random.uniform(bounds['mutation_rate'][0], bounds['mutation_rate'][1])
            }
            population.append(individual)

        fitness = []
        for ind in population:
            fitness.append(self.evaluate(ind, num_thresholds, hist, fitness_func, seed))

        for gen in range(self.de_generations):
            for i in range(self.de_pop_size):
                indices = [j for j in range(self.de_pop_size) if j != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                trial = {}
                for key in population[i].keys():
                    if random.random() < self.CR:
                        if key in ['pop_size', 'max_generations']:
                            val = int(round(a[key] + self.F * (b[key] - c[key])))
                            val = max(bounds[key][0], min(bounds[key][1], val))
                        else:
                            val = a[key] + self.F * (b[key] - c[key])
                            val = max(bounds[key][0], min(bounds[key][1], val))
                        trial[key] = val
                    else:
                        trial[key] = population[i][key]

                trial_fitness = self.evaluate(trial, num_thresholds, hist, fitness_func, seed)

                if trial_fitness > fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            if gen % 5 == 0:
                best_fit = max(fitness)
                print(f"DE Generation {gen}: Best fitness = {best_fit:.6f}")

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
# Image Processing
# ===============================

def process_single(file_name, folder_path, num_thresholds, func_name, de_configurator, seed=None, images_folder=None):
    """Process a single image with given parameters."""
    try:
        # Generate consistent seed based on image name, fitness function, and threshold level
        seed_str = f"{file_name}_{func_name}_{num_thresholds}_{seed}"
        run_seed = hash(seed_str) % 1000000
        np.random.seed(run_seed)
        random.seed(run_seed)

        start_time = time.time()

        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[ERROR] Could not load image: {file_name}")
            return None

        print(f"[START] Processing {file_name} | {func_name} | thresholds={num_thresholds}")

        hist = calculate_histogram(image)
        if func_name == "kapur":
            _image = image
            def fitness_function(hist, thresholds, _img=_image):
                return calculate_kapur_entropy(_img, thresholds)
        else:
            fitness_function = calculate_otsu

        config = de_configurator.optimize_ga_params(num_thresholds, image, fitness_function, run_seed)

        ga = GeneticAlgorithm(config, num_thresholds, hist, fitness_function, run_seed)
        best_thresholds, best_score = ga.evolve()

        execution_time_ms = (time.time() - start_time) * 1000

        segmented_image = apply_thresholds(image, best_thresholds)

        ssim_value = calculate_ssim(image, segmented_image)
        mse_value = calculate_mse(image, segmented_image)
        psnr_value = calculate_psnr(image, segmented_image)
        uniformity = calculate_uniformity(segmented_image)
        threshold_value_str = "[" + ", ".join(map(str, best_thresholds)) + "]"

        if images_folder:
            image_name_base = os.path.splitext(file_name)[0]
            output_filename = f"{image_name_base}_{func_name}_{num_thresholds}_thresholds.png"
            output_image_path = os.path.join(images_folder, output_filename)
            cv2.imwrite(output_image_path, segmented_image)

        print(f"[DONE] {file_name} | {func_name} | thresholds={num_thresholds} | "
              f"fitness={best_score:.4f} | time={execution_time_ms:.2f}ms")

        return {
            "image_name": file_name,
            "fitness_function": func_name,  # Keep lowercase
            "thresholding_level": num_thresholds,
            "threshold_value": threshold_value_str,
            "fitness_value": best_score,
            "SSIM": ssim_value,
            "MSE": mse_value,
            "PSNR": psnr_value,
            "Uniformity": uniformity,
            "Seed": run_seed,
            "Execution Time (ms)": execution_time_ms
        }

    except Exception as e:
        print(f"[ERROR] Processing {file_name} | {func_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "image_name": file_name,
            "fitness_function": func_name,
            "error": str(e)
        }


def safe_save_excel(df, filepath):
    """Safely save DataFrame to Excel with error handling."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Close any open file handles first
            import gc
            gc.collect()

            # Try to save with a temporary filename first
            temp_filepath = filepath.replace('.xlsx', f'_temp_{attempt}.xlsx')
            df.to_excel(temp_filepath, index=False)

            # If successful, rename to target filename
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass  # Ignore removal errors

            os.rename(temp_filepath, filepath)
            print(f"Successfully saved to: {filepath}")
            return True

        except PermissionError:
            print(f"Permission denied (attempt {attempt + 1}/{max_retries}). Retrying...")
            time.sleep(1)
        except Exception as e:
            print(f"Error saving to Excel (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(1)

    # If all retries fail, try CSV
    try:
        csv_path = filepath.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved results to CSV instead: {csv_path}")
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False


def process_images_in_folder(folder_path, threshold_levels, output_file, n_jobs=1, seed=None):
    """Main function to process all images in a folder."""
    output_path = os.path.dirname(output_file)
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_DEGA")

    # Create directories if they don't exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    if not files:
        print("No image files found in the specified folder.")
        return pd.DataFrame()

    print(f"Found {len(files)} images to process.")

    if seed is None:
        seed = int(time.time() * 1000) % 1000000
    np.random.seed(seed)
    random.seed(seed)
    print(f"Using random seed: {seed}")

    de_configurator = DEGAConfigurator(seed=seed)

    # Create tasks in the specific order: image1-kapur-all_levels, image1-otsu-all_levels, etc.
    tasks = []
    for f in files:
        for func in ["kapur", "otsu"]:  # Process kapur first for each image, then otsu
            for t in threshold_levels:
                tasks.append((f, folder_path, t, func, de_configurator, seed, images_folder))

    print(f"Processing {len(tasks)} tasks with {n_jobs} workers...")

    if n_jobs == 1:
        results = []
        for i, task in enumerate(tasks):
            print(f"Processing task {i + 1}/{len(tasks)}...")
            result = process_single(*task)
            results.append(result)
    else:
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_single)(*task) for task in tasks
        )

    valid_results = [r for r in results if r is not None and 'error' not in r]
    error_results = [r for r in results if r is not None and 'error' in r]

    if valid_results:
        df = pd.DataFrame(valid_results)

        # Use exact column names as requested
        column_order = [
            'image_name',
            'fitness_function',
            'thresholding_level',
            'threshold_value',
            'fitness_value',
            'SSIM',
            'MSE',
            'PSNR',
            'Uniformity',
            'Seed',
            'Execution Time (ms)'
        ]

        # Ensure all columns exist
        for col in column_order:
            if col not in df.columns:
                df[col] = None

        df = df[column_order]

        # Create a custom sorting key for fitness_function to ensure "kapur" comes before "otsu"
        def fitness_function_order(func):
            return 0 if func == "kapur" else 1

        # Sort the results to match the desired order:
        # 1. First by image name
        # 2. Then by fitness function (kapur first, then otsu) using custom order
        # 3. Then by thresholding level (ascending)
        df_sorted = df.copy()
        df_sorted['fitness_order'] = df_sorted['fitness_function'].apply(fitness_function_order)
        df_sorted = df_sorted.sort_values(['image_name', 'fitness_order', 'thresholding_level'])
        df_sorted = df_sorted.drop('fitness_order', axis=1)
        df_sorted = df_sorted.reset_index(drop=True)

        # Format numbers properly
        df_sorted['fitness_value'] = df_sorted['fitness_value'].apply(lambda x: format_number(x, 8))
        df_sorted['SSIM'] = df_sorted['SSIM'].apply(lambda x: format_number(x, 6))
        df_sorted['MSE'] = df_sorted['MSE'].apply(lambda x: format_number(x, 2))
        df_sorted['PSNR'] = df_sorted['PSNR'].apply(lambda x: format_number(x, 2))
        df_sorted['Uniformity'] = df_sorted['Uniformity'].apply(lambda x: format_number(x, 6))
        df_sorted['Execution Time (ms)'] = df_sorted['Execution Time (ms)'].apply(lambda x: format_number(x, 2))

        # Save the results
        if safe_save_excel(df_sorted, output_file):
            print(f"Results saved to {output_file}")
        else:
            # Last resort: print results to console
            print("Could not save to file. Printing results:")
            print(df_sorted.to_string())

        print(f"Successfully processed {len(valid_results)} configurations")

        # Display preview of results
        print("\nResults preview (first 12 rows):")
        print("-" * 120)
        print(
            f"{'image_name':<15} {'fitness_function':<15} {'thresholding_level':<18} {'threshold_value':<15} {'fitness_value':<15} {'SSIM':<10} {'MSE':<10} {'PSNR':<8} {'Uniformity':<18} {'Seed':<10} {'Execution Time (ms)':<15}")
        print("-" * 120)

        for i, row in df_sorted.head(12).iterrows():
            print(
                f"{row['image_name']:<15} {row['fitness_function']:<15} {row['thresholding_level']:<18} {row['threshold_value']:<15} {row['fitness_value']:<15} {row['SSIM']:<10} {row['MSE']:<10} {row['PSNR']:<8} {row['Uniformity']:<18} {row['Seed']:<10} {row['Execution Time (ms)']:<15}")

        # Display summary statistics
        print("\nSummary Statistics:")
        numeric_columns = ['SSIM', 'PSNR', 'MSE', 'Uniformity', 'fitness_value']
        for col in numeric_columns:
            if col in df_sorted.columns and df_sorted[col].notna().any():
                # Convert back to numeric for statistics
                col_data = pd.to_numeric(df_sorted[col], errors='coerce').dropna()
                if len(col_data) > 0:
                    print(f"{col} value range: [{col_data.min():.6f} - {col_data.max():.6f}]")

    else:
        print("No valid results to save")
        df_sorted = pd.DataFrame()

    if error_results:
        print(f"\nErrors encountered in {len(error_results)} processing tasks:")
        for error_result in error_results:
            print(f"  {error_result['image_name']} - {error_result['fitness_function']}: {error_result['error']}")

    return df_sorted


# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    # Configuration with your specific paths
    # ── Update these two paths before running ─────────────────────────────────────
    folder = r"path/to/your/images"          # folder containing input images
    output_dir = r"path/to/your/results"          # folder where results will be saved
    # ──────────────────────────────────────────────────────────────────────────────
    levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Create results directory
    os.makedirs(output_dir, exist_ok=True)

    # Use a unique filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"DEGA_results_{timestamp}.xlsx")

    print(f"Input folder: {folder}")
    print(f"Output file: {output_file}")
    print(f"Threshold levels: {levels}")

    try:
        df = process_images_in_folder(folder, levels, output_file, n_jobs=1, seed=42)

        if not df.empty:
            print("\nProcessing completed successfully!")
            print(f"Total results: {len(df)}")

            # Display final file locations
            images_folder = os.path.join(output_dir, "imagesPerThresholdLevel_DEGA")
            print(f"Results saved to: {output_file}")
            print(f"Segmented images saved to: {images_folder}")

            # Show grouping by image and fitness function
            print("\nGrouping structure:")
            grouped = df.groupby(['image_name', 'fitness_function']).size()
            for (image, func), count in grouped.items():
                print(f"{image} - {func}: {count} threshold levels")
        else:
            print("Processing completed with errors!")
    except Exception as e:
        print(f"Critical error during processing: {e}")
        import traceback

        traceback.print_exc()