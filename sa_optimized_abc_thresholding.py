import os
import cv2
import numpy as np
import random
import pandas as pd
import time
from skimage.metrics import structural_similarity as ssim
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from functools import lru_cache
import warnings

warnings.filterwarnings('ignore')


# ---------------------------
# OPTIMIZED Thresholding utilities (ORIGINAL PARAMETERS)
# ---------------------------

def apply_thresholds_optimized(image, thresholds):
    """Optimized threshold application using vectorized operations"""
    thresholds = np.sort(thresholds)
    output = np.zeros_like(image, dtype=np.uint8)

    # Vectorized threshold application
    prev_threshold = 0
    for threshold in thresholds:
        mask = (image >= prev_threshold) & (image < threshold)
        output[mask] = threshold
        prev_threshold = threshold

    # Handle values above the last threshold
    output[image >= thresholds[-1]] = 255
    return output


def kapur_entropy_optimized(image, thresholds, hist=None, prob=None):
    """OPTIMIZED Kapur's entropy with pre-calculated histogram"""
    if hist is None or prob is None:
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        total_pixels = np.sum(hist)
        prob = hist.astype(float) / total_pixels

    thresholds = sorted(thresholds)
    class_boundaries = [0] + thresholds + [256]
    total_entropy = 0.0

    for i in range(len(class_boundaries) - 1):
        start, end = class_boundaries[i], class_boundaries[i + 1]
        class_probs = prob[start:end]
        class_probs = class_probs[class_probs > 0]

        if len(class_probs) > 0:
            class_prob_sum = np.sum(class_probs)
            if class_prob_sum > 0:
                normalized_probs = class_probs / class_prob_sum
                class_entropy = -np.sum(normalized_probs * np.log(normalized_probs))
                total_entropy += class_entropy

    return total_entropy


def otsu_variance_optimized(image, thresholds, hist=None, prob=None):
    """OPTIMIZED Otsu's between-class variance with pre-calculated histogram"""
    if hist is None or prob is None:
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        total_pixels = np.sum(hist)
        prob = hist.astype(float) / total_pixels

    global_mean = np.sum(np.arange(256) * prob)
    thresholds = sorted(thresholds)
    class_boundaries = [0] + thresholds + [256]
    between_class_variance = 0.0

    for i in range(len(class_boundaries) - 1):
        start, end = class_boundaries[i], class_boundaries[i + 1]
        class_weight = np.sum(prob[start:end])

        if class_weight > 0:
            class_indices = np.arange(start, end)
            class_mean = np.sum(class_indices * prob[start:end]) / class_weight
            between_class_variance += class_weight * (class_mean - global_mean) ** 2

    return between_class_variance


# ---------------------------
# OPTIMIZED Metrics Functions
# ---------------------------

def calculate_mse_optimized(original, segmented):
    """Optimized MSE calculation"""
    diff = original.astype(np.int32) - segmented.astype(np.int32)
    return np.mean(diff.astype(np.float64) ** 2)


def calculate_psnr_optimized(mse_value, max_pixel=255.0):
    """Optimized PSNR calculation"""
    if mse_value <= 1e-10:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse_value)


def calculate_uniformity_measure_optimized(segmented):
    """Optimized Uniformity Measure calculation"""
    hist, _ = np.histogram(segmented.flatten(), bins=256, range=(0, 256))
    prob = hist.astype(float) / np.sum(hist)
    return np.sum(prob ** 2)


def validate_metrics_optimized(ssim_val, mse_val, psnr_val, uniformity_val, filename=""):
    """Optimized metric validation"""
    ssim_val = np.clip(ssim_val, -1.0, 1.0)
    mse_val = max(0.0, mse_val)
    uniformity_val = np.clip(uniformity_val, 0.0, 1.0)
    return ssim_val, mse_val, psnr_val, uniformity_val


# ---------------------------
# OPTIMIZED ABC with ORIGINAL PARAMETERS
# ---------------------------

class ABC_Optimized:
    def __init__(self, image, obj_func, num_thresholds, pop_size=20, max_iter=50, limit=5):
        self.image = image
        self.obj_func = obj_func
        self.num_thresholds = num_thresholds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.limit = limit
        self.dim = num_thresholds
        self.bounds = (1, 254)

        # Pre-calculate histogram for the entire image once
        self.hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        self.total_pixels = np.sum(self.hist)
        self.prob = self.hist.astype(float) / self.total_pixels

    def initialize_population_optimized(self):
        """Optimized population initialization"""
        population = []
        max_attempts = 100

        for _ in range(self.pop_size):
            for attempt in range(max_attempts):
                thresholds = np.sort(np.random.randint(self.bounds[0], self.bounds[1], self.dim))
                if np.all(np.diff(thresholds) >= 10):
                    population.append(thresholds.tolist())
                    break
            else:
                thresholds = np.linspace(self.bounds[0], self.bounds[1] - 1, self.dim, dtype=int)
                population.append(thresholds.tolist())

        return population

    def evaluate_optimized(self, solution):
        """Use pre-calculated histogram for faster evaluation"""
        return self.obj_func(self.image, solution, self.hist, self.prob)

    def evolve_optimized(self):
        """Optimized ABC evolution"""
        population = self.initialize_population_optimized()
        fitness = np.array([self.evaluate_optimized(sol) for sol in population])
        trial = np.zeros(self.pop_size, dtype=int)

        best_idx = np.argmax(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for iteration in range(self.max_iter):
            # Employed bees phase
            for i in range(self.pop_size):
                k = np.random.choice([x for x in range(self.pop_size) if x != i])
                j = np.random.randint(0, self.dim)

                phi = np.random.uniform(-1, 1)
                new_sol = population[i].copy()
                new_sol[j] = int(np.clip(
                    population[i][j] + phi * (population[i][j] - population[k][j]),
                    self.bounds[0], self.bounds[1]
                ))

                new_sol_sorted = sorted(set(new_sol))
                if len(new_sol_sorted) == self.dim:
                    new_sol = new_sol_sorted
                    new_fit = self.evaluate_optimized(new_sol)

                    if new_fit > fitness[i]:
                        population[i] = new_sol
                        fitness[i] = new_fit
                        trial[i] = 0
                    else:
                        trial[i] += 1
                else:
                    trial[i] += 1

            # Onlooker bees phase
            fitness_positive = np.maximum(fitness, 1e-10)
            probs = fitness_positive / np.sum(fitness_positive)
            probs = np.nan_to_num(probs, nan=1.0 / self.pop_size)

            for _ in range(self.pop_size):
                i = np.random.choice(range(self.pop_size), p=probs)
                k = np.random.choice([x for x in range(self.pop_size) if x != i])
                j = np.random.randint(0, self.dim)

                phi = np.random.uniform(-1, 1)
                new_sol = population[i].copy()
                new_sol[j] = int(np.clip(
                    population[i][j] + phi * (population[i][j] - population[k][j]),
                    self.bounds[0], self.bounds[1]
                ))

                new_sol_sorted = sorted(set(new_sol))
                if len(new_sol_sorted) == self.dim:
                    new_sol = new_sol_sorted
                    new_fit = self.evaluate_optimized(new_sol)

                    if new_fit > fitness[i]:
                        population[i] = new_sol
                        fitness[i] = new_fit
                        trial[i] = 0
                    else:
                        trial[i] += 1
                else:
                    trial[i] += 1

            # Scout bees phase
            scout_indices = np.where(trial > self.limit)[0]
            for i in scout_indices:
                for attempt in range(50):
                    new_sol = sorted(np.random.randint(self.bounds[0], self.bounds[1], self.dim))
                    if all(new_sol[i + 1] - new_sol[i] >= 10 for i in range(len(new_sol) - 1)):
                        population[i] = new_sol
                        fitness[i] = self.evaluate_optimized(new_sol)
                        trial[i] = 0
                        break
                else:
                    new_sol = list(np.linspace(self.bounds[0], self.bounds[1] - 1, self.dim, dtype=int))
                    population[i] = new_sol
                    fitness[i] = self.evaluate_optimized(new_sol)
                    trial[i] = 0

            # Update best solution
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_solution = population[current_best_idx]
                best_fitness = fitness[current_best_idx]

        return best_solution, best_fitness


# ---------------------------
# SA-ABC Optimizer with ORIGINAL PARAMETERS
# ---------------------------

class SA_ABC_Optimizer_OriginalParams:
    def __init__(self, image, obj_func, num_thresholds,
                 initial_temp=1000, cooling_rate=0.95, max_iter=20):
        self.image = image
        self.obj_func = obj_func
        self.num_thresholds = num_thresholds
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter

        self.param_bounds = {
            'pop_size': (15, 40),
            'max_iter': (30, 80),
            'limit': (5, 15)
        }

    def generate_initial_solution(self):
        """Smarter initialization with original ranges"""
        return {
            'pop_size': random.randint(*self.param_bounds['pop_size']),
            'max_iter': random.randint(*self.param_bounds['max_iter']),
            'limit': random.randint(*self.param_bounds['limit'])
        }

    def generate_neighbor_optimized(self, current_solution):
        """Neighbor generation with original parameter ranges"""
        neighbor = current_solution.copy()
        param = random.choice(list(self.param_bounds.keys()))
        step = random.choice([-2, -1, 1, 2])
        new_val = neighbor[param] + step
        new_val = max(self.param_bounds[param][0], min(self.param_bounds[param][1], new_val))
        neighbor[param] = new_val
        return neighbor

    def evaluate_abc_params_optimized(self, params):
        """ABC parameter evaluation"""
        abc = ABC_Optimized(self.image, self.obj_func, self.num_thresholds,
                            pop_size=params['pop_size'],
                            max_iter=params['max_iter'],
                            limit=params['limit'])
        _, fitness = abc.evolve_optimized()
        return fitness

    def optimize_improved(self):
        """SA optimization with original parameters"""
        current_solution = self.generate_initial_solution()
        current_fitness = self.evaluate_abc_params_optimized(current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness

        temperature = self.initial_temp

        for iteration in range(self.max_iter):
            neighbor = self.generate_neighbor_optimized(current_solution)
            neighbor_fitness = self.evaluate_abc_params_optimized(neighbor)

            delta = neighbor_fitness - current_fitness

            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness

                if neighbor_fitness > best_fitness:
                    best_solution = neighbor.copy()
                    best_fitness = neighbor_fitness

            temperature *= self.cooling_rate

        return best_solution, best_fitness


# ---------------------------
# Parallel Processing Functions
# ---------------------------

def process_single_image_optimized(args):
    """Process a single image with ORIGINAL parameters"""
    filename, folder_path, threshold_levels, output_path = args

    try:
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        # Pre-calculate image properties once per image
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        total_pixels = np.sum(hist)
        prob = hist.astype(float) / total_pixels

        image_results = []

        def kapur_wrapper(img, thresh, h=None, p=None):
            return kapur_entropy_optimized(img, thresh, h, p)

        def otsu_wrapper(img, thresh, h=None, p=None):
            return otsu_variance_optimized(img, thresh, h, p)

        for method, obj_func in [("kapur", kapur_wrapper), ("otsu", otsu_wrapper)]:
            for m in threshold_levels:
                start_time = time.time()
                run_seed = int((start_time * 10000) % 1000000) + 1
                random.seed(run_seed)
                np.random.seed(run_seed)

                # Use ORIGINAL SA parameters
                sa_optimizer = SA_ABC_Optimizer_OriginalParams(
                    image, obj_func, m,
                    initial_temp=1000,
                    cooling_rate=0.95,
                    max_iter=20
                )

                optimized_params, _ = sa_optimizer.optimize_improved()

                # Run ABC with optimized parameters
                abc = ABC_Optimized(image, obj_func, m,
                                    pop_size=optimized_params['pop_size'],
                                    max_iter=optimized_params['max_iter'],
                                    limit=optimized_params['limit'])

                best_sol, best_fit = abc.evolve_optimized()
                execution_time_ms = (time.time() - start_time) * 1000

                # Create segmented image and calculate metrics
                segmented = apply_thresholds_optimized(image, best_sol)

                # Calculate metrics
                mse_value = calculate_mse_optimized(image, segmented)
                psnr_value = calculate_psnr_optimized(mse_value)
                ssim_value = ssim(image, segmented, data_range=255)
                uniformity = calculate_uniformity_measure_optimized(segmented)

                # Validate metrics
                ssim_value, mse_value, psnr_value, uniformity = validate_metrics_optimized(
                    ssim_value, mse_value, psnr_value, uniformity, filename
                )

                # Calculate both fitness values using pre-calculated histogram
                kapur_value = kapur_entropy_optimized(image, best_sol, hist, prob)
                otsu_value = otsu_variance_optimized(image, best_sol, hist, prob)

                threshold_str = f"[{', '.join(map(str, best_sol))}]"

                result = {
                    "image_name": filename,
                    "fitness_function": method,
                    "thresholding_level": m,
                    "threshold_value": threshold_str,
                    "fitness_value": best_fit,
                    "Kapur_Value": kapur_value,
                    "Otsu_Value": otsu_value,
                    "SSIM": ssim_value,
                    "MSE": mse_value,
                    "PSNR": psnr_value,
                    "Uniformity Measure": uniformity,
                    "Seed": run_seed,
                    "Execution Time (ms)": execution_time_ms,
                    "ABC_pop_size": optimized_params['pop_size'],
                    "ABC_max_iter": optimized_params['max_iter'],
                    "ABC_limit": optimized_params['limit']
                }

                image_results.append(result)

        return image_results

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None


def process_images_parallel(folder_path, threshold_levels, output_path, use_parallel=True):
    """Process images in parallel or sequentially"""
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]

    if use_parallel and len(image_files) > 1:
        # Use parallel processing for multiple images
        num_workers = min(multiprocessing.cpu_count(), len(image_files))
        print(f"Using parallel processing with {num_workers} workers")

        args_list = [(f, folder_path, threshold_levels, output_path) for f in image_files]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_single_image_optimized, args_list))
    else:
        # Sequential processing
        print("Using sequential processing")
        results = []
        for filename in image_files:
            args = (filename, folder_path, threshold_levels, output_path)
            result = process_single_image_optimized(args)
            results.append(result)

    # Flatten results
    all_results = []
    for result in results:
        if result is not None:
            all_results.extend(result)

    return all_results


# ---------------------------
# Results Saving Function
# ---------------------------

def save_results_optimized(results, output_path):
    """Optimized results saving"""
    if not results:
        print("No results to save")
        return

    df = pd.DataFrame(results)
    excel_file_path = os.path.join(output_path, "SA-ABC_Results_Optimized.xlsx")

    # Format numeric columns efficiently
    numeric_columns = {
        'fitness_value': "{:.8f}",
        'Kapur_Value': "{:.8f}",
        'Otsu_Value': "{:.8f}",
        'SSIM': "{:.6f}",
        'MSE': "{:.2f}",
        'PSNR': "{:.2f}",
        'Uniformity Measure': "{:.6f}",
        'Execution Time (ms)': "{:.2f}"
    }

    for col, fmt in numeric_columns.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt.format(x))

    # Save to Excel
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
        worksheet = writer.sheets['Results']

        # Set column widths
        for col_idx, column in enumerate(worksheet.columns, 1):
            max_length = max(len(str(cell.value)) for cell in column)
            worksheet.column_dimensions[chr(64 + col_idx)].width = min(max_length + 2, 40)

    # Print summary statistics
    print("\n=== OPTIMIZED PROCESSING SUMMARY ===")
    if results:
        print(f"Total results: {len(results)}")
        print(f"SSIM range: [{min(r['SSIM'] for r in results):.6f}, {max(r['SSIM'] for r in results):.6f}]")
        print(f"MSE range: [{min(r['MSE'] for r in results):.2f}, {max(r['MSE'] for r in results):.2f}]")
        print(f"PSNR range: [{min(r['PSNR'] for r in results):.2f}, {max(r['PSNR'] for r in results):.2f}]")
        print(f"Average execution time: {np.mean([r['Execution Time (ms)'] for r in results]):.2f} ms")

    print(f"\nResults saved to {excel_file_path}")


# ---------------------------
# Validation Function
# ---------------------------

def validate_functions_optimized():
    """Optimized function validation"""
    print("=== VALIDATING OPTIMIZED FUNCTIONS ===")
    test_image = np.random.randint(0, 256, (50, 50))
    thresholds = [128]

    # Test with pre-calculated histogram
    hist, _ = np.histogram(test_image.flatten(), bins=256, range=(0, 256))
    total_pixels = np.sum(hist)
    prob = hist.astype(float) / total_pixels

    kapur_val = kapur_entropy_optimized(test_image, thresholds, hist, prob)
    otsu_val = otsu_variance_optimized(test_image, thresholds, hist, prob)

    print(f"Validation completed successfully")
    print(f"Kapur entropy: {kapur_val:.6f}")
    print(f"Otsu variance: {otsu_val:.6f}")

    return True


# ---------------------------
# OPTIMIZED Main Processing Function
# ---------------------------

def process_images_optimized(folder_path, threshold_levels, output_path, use_parallel=True):
    """Main optimized processing function with ORIGINAL parameters"""
    start_total = time.time()

    # Validate functions
    validate_functions_optimized()

    # Create output directory
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_SAABC_Optimized")
    os.makedirs(images_folder, exist_ok=True)

    print(f"Processing images from: {folder_path}")
    print(f"Threshold levels: {threshold_levels}")
    print(f"Output path: {output_path}")
    print(f"Parallel processing: {use_parallel}")
    print("Using ORIGINAL parameters for SA-ABC optimization")

    # Process images
    results = process_images_parallel(folder_path, threshold_levels, output_path, use_parallel)

    # Save results
    save_results_optimized(results, output_path)

    total_time = (time.time() - start_total) / 3600
    print(f"\n=== TOTAL PROCESSING TIME: {total_time:.2f} hours ===")

    return results


# ---------------------------
# Main Execution with ORIGINAL parameters
# ---------------------------

if __name__ == "__main__":
    # Configuration with ORIGINAL parameters
    folder_path = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images"
    output_path = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images\results"
    threshold_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]  # ORIGINAL FULL RANGE

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Run optimized processing with ORIGINAL parameters
    print("=== STARTING OPTIMIZED SA-ABC PROCESSING WITH ORIGINAL PARAMETERS ===")
    results = process_images_optimized(folder_path, threshold_levels, output_path, use_parallel=True)
    print("=== PROCESSING COMPLETED SUCCESSFULLY ===")