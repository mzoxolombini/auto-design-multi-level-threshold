import os
import cv2
import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed
from scipy.ndimage import correlate
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import math
import time


# ===============================
# Utility Functions
# ===============================

def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    return hist / hist.sum()


def kapur_entropy(hist, thresholds):
    thresholds = sorted(thresholds)
    bins = [0] + thresholds + [len(hist)]
    total_entropy = 0
    cumulative = np.cumsum(hist)

    for start, end in zip(bins[:-1], bins[1:]):
        prob = hist[start:end]
        prob_sum = cumulative[end - 1] - (cumulative[start - 1] if start > 0 else 0)
        if prob_sum > 0:
            prob = prob / prob_sum
            prob = prob[prob > 0]
            total_entropy += -np.sum(prob * np.log(prob))
    return total_entropy


def otsu_between_class_variance(hist, thresholds):
    thresholds = sorted(thresholds)
    bins = [0] + thresholds + [len(hist)]
    total_mean = np.sum(np.arange(len(hist)) * hist)
    total_weight = np.sum(hist)
    variance = 0

    cumulative = np.cumsum(hist)
    mean_cumulative = np.cumsum(np.arange(len(hist)) * hist)

    for start, end in zip(bins[:-1], bins[1:]):
        w = cumulative[end - 1] - (cumulative[start - 1] if start > 0 else 0)
        if w > 0:
            m = mean_cumulative[end - 1] - (mean_cumulative[start - 1] if start > 0 else 0)
            m /= w
            variance += w * (m - total_mean) ** 2
    return variance / total_weight


def calculate_ssim(original, segmented):
    return ssim(original, segmented, data_range=segmented.max() - segmented.min())


def calculate_mse(original, segmented):
    return mean_squared_error(original, segmented)


def calculate_psnr(original, segmented):
    mse = calculate_mse(original, segmented)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))


def calculate_uniformity_measure(segmented, thresholds):
    if len(thresholds) == 0:
        return 0
    uniformity = 0
    total_pixels = segmented.size
    sorted_thresholds = sorted(thresholds)
    region_boundaries = [0] + sorted_thresholds + [255]

    for i in range(len(region_boundaries) - 1):
        lower = region_boundaries[i]
        upper = region_boundaries[i + 1]
        mask = (segmented >= lower) & (segmented <= upper)
        region_pixels = np.sum(mask)
        if region_pixels > 0:
            region_mean = np.mean(segmented[mask])
            squared_diff = np.sum((segmented[mask] - region_mean) ** 2)
            uniformity += 1 - (squared_diff / (region_pixels * (upper - lower) ** 2))
    return uniformity


def apply_thresholds(image, thresholds):
    if len(thresholds) == 0:
        return image
    thresholds = sorted(thresholds)
    segmented = np.zeros_like(image)
    mask = image <= thresholds[0]
    segmented[mask] = 0
    for i in range(len(thresholds) - 1):
        mask = (image > thresholds[i]) & (image <= thresholds[i + 1])
        segmented[mask] = thresholds[i]
    mask = image > thresholds[-1]
    segmented[mask] = thresholds[-1]
    return segmented


def format_number(value, decimals=8):
    if isinstance(value, (int, np.integer)):
        return str(value)
    fmt = f"{value:.{decimals}f}"
    return fmt.replace('.', ',')


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

        # cache fitness to avoid duplicate evaluations
        self.fitness_cache = {}
        self.population = [np.sort(np.random.randint(0, 256, num_thresholds))
                           for _ in range(self.pop_size)]
        self.fitness = [self.evaluate(ind) for ind in self.population]

    def evaluate(self, individual):
        key = tuple(individual)
        if key not in self.fitness_cache:
            self.fitness_cache[key] = self.fitness_func(self.hist, individual)
        return self.fitness_cache[key]

    def select_parent(self):
        idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
        return self.population[idx1] if self.fitness[idx1] > self.fitness[idx2] else self.population[idx2]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.num_thresholds)
            child1 = np.concatenate((parent1[:point], parent2[point:]))
            child2 = np.concatenate((parent2[:point], parent1[point:]))
            return np.sort(child1), np.sort(child2)
        return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        for i in range(self.num_thresholds):
            if np.random.rand() < self.mutation_rate:
                individual[i] = np.random.randint(0, 256)
        return np.sort(individual)

    def evolve(self):
        best_idx = np.argmax(self.fitness)
        best_ind = self.population[best_idx].copy()
        best_fit = self.fitness[best_idx]

        for _ in range(self.max_generations):
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            f1 = self.evaluate(child1)
            f2 = self.evaluate(child2)

            # replace worst individuals
            worst_indices = np.argsort(self.fitness)[:2]
            for idx, (child, f) in zip(worst_indices, [(child1, f1), (child2, f2)]):
                self.population[idx] = child
                self.fitness[idx] = f

            # elitism
            if best_fit > min(self.fitness):
                worst_idx = np.argmin(self.fitness)
                self.population[worst_idx] = best_ind.copy()
                self.fitness[worst_idx] = best_fit

            # update best
            current_best_idx = np.argmax(self.fitness)
            if self.fitness[current_best_idx] > best_fit:
                best_fit = self.fitness[current_best_idx]
                best_ind = self.population[current_best_idx].copy()

        return best_ind, best_fit


# ===============================
# Differential Evolution Configurator
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
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        hist = calculate_histogram(image)
        bounds = {
            'pop_size': (20, 80),
            'max_generations': (30, 150),
            'crossover_rate': (0.6, 0.9),
            'mutation_rate': (0.01, 0.2)
        }

        population = []
        for _ in range(self.de_pop_size):
            population.append({
                'pop_size': random.randint(*bounds['pop_size']),
                'max_generations': random.randint(*bounds['max_generations']),
                'crossover_rate': random.uniform(*bounds['crossover_rate']),
                'mutation_rate': random.uniform(*bounds['mutation_rate'])
            })

        fitness = [self.evaluate(ind, num_thresholds, hist, fitness_func, seed) for ind in population]

        for _ in range(self.de_generations):
            for i in range(self.de_pop_size):
                idxs = list(range(self.de_pop_size))
                idxs.remove(i)
                a, b, c = random.sample([population[j] for j in idxs], 3)
                trial = {}
                for key in population[i].keys():
                    if random.random() < self.CR:
                        if key in ['pop_size', 'max_generations']:
                            val = int(round(a[key] + self.F * (b[key] - c[key])))
                            val = int(np.clip(val, bounds[key][0], bounds[key][1]))
                        else:
                            val = a[key] + self.F * (b[key] - c[key])
                            val = float(np.clip(val, bounds[key][0], bounds[key][1]))
                        trial[key] = val
                    else:
                        trial[key] = population[i][key]
                trial_fitness = self.evaluate(trial, num_thresholds, hist, fitness_func, seed)
                if trial_fitness > fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

        best_index = np.argmax(fitness)
        return population[best_index]

    def evaluate(self, config, num_thresholds, hist, fitness_func, seed=None):
        ga = GeneticAlgorithm(config, num_thresholds, hist, fitness_func, seed)
        _, best_fit = ga.evolve()
        return best_fit


# ===============================
# Image Processing
# ===============================

def process_single(file_name, folder_path, num_thresholds, func_name, de_configurator, seed=None, images_folder=None):
    try:
        # Generate unique seed for this specific run
        run_seed = int(time.time() * 1000) % 1000000
        np.random.seed(run_seed)
        random.seed(run_seed)

        # Start timing
        start_time = time.time()

        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        # Calculate histogram once for the image
        hist = calculate_histogram(image)

        fitness_function = kapur_entropy if func_name == "kapur" else otsu_between_class_variance
        optimal_cfg = de_configurator.optimize_ga_params(num_thresholds, image, fitness_function, run_seed)

        ga = GeneticAlgorithm(optimal_cfg, num_thresholds, hist, fitness_function, run_seed)
        best_thresholds, best_score = ga.evolve()

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        segmented_image = apply_thresholds(image, best_thresholds)

        ssim_value = calculate_ssim(image, segmented_image)
        mse_value = calculate_mse(image, segmented_image)
        psnr_value = calculate_psnr(image, segmented_image)
        uniformity = calculate_uniformity_measure(segmented_image, best_thresholds)
        threshold_value_str = "[" + ", ".join(map(str, best_thresholds)) + "]"

        # Save the thresholded image
        if images_folder:
            image_name_base = os.path.splitext(file_name)[0]
            output_filename = f"{image_name_base}_{func_name}_{num_thresholds}_thresholds.png"
            output_image_path = os.path.join(images_folder, output_filename)
            cv2.imwrite(output_image_path, segmented_image)

        return {
            "image_name": file_name,
            "fitness_function": func_name.capitalize(),  # Capitalize first letter
            "thresholding_level": num_thresholds,
            "threshold_value": threshold_value_str,
            "fitness_value": format_number(best_score, 8),  # 8 decimal places
            "SSIM": format_number(ssim_value, 7),  # 7 decimal places
            "MSE": format_number(mse_value, 7),  # 7 decimal places
            "PSNR": format_number(psnr_value, 7),  # 7 decimal places
            "Uniformity Measure": format_number(uniformity, 7),  # 7 decimal places
            "Random Seed": run_seed,  # Use the unique run seed
            "Execution Time (ms)": format_number(execution_time_ms, 2)  # 2 decimal places
        }
    except Exception as e:
        return {"image_name": file_name, "fitness_function": func_name, "error": str(e)}


def process_images_in_folder(folder_path, threshold_levels, output_file, n_jobs=-1, seed=None):
    # Create images folder
    output_path = os.path.dirname(output_file)
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_DEGA")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Generate unique seed for main process
    if seed is None:
        seed = int(time.time() * 1000) % 1000000
    np.random.seed(seed)
    random.seed(seed)

    de_configurator = DEGAConfigurator(seed=seed)

    tasks = [(f, folder_path, t, func, de_configurator, seed, images_folder)
             for f in files for func in ["kapur", "otsu"] for t in threshold_levels]

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single)(*task) for task in tasks
    )
    results = [r for r in results if r is not None]

    # Filter out error results
    valid_results = [r for r in results if 'error' not in r]

    if valid_results:
        # Create DataFrame with the exact column order
        df = pd.DataFrame(valid_results)

        # Reorder columns to match the desired format exactly
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

        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")

        # Print error results if any
        error_results = [r for r in results if 'error' in r]
        if error_results:
            print(f"Errors encountered in {len(error_results)} processing tasks:")
            for error_result in error_results:
                print(f"  {error_result['image_name']} - {error_result['fitness_function']}: {error_result['error']}")
    else:
        print("No valid results to save")
        df = pd.DataFrame()

    return df


# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    folder = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images"
    levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    output_dir = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images\results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "DEGA_results.xlsx")

    random_seed = int(time.time() * 1000) % 1000000
    print(f"Using random seed: {random_seed}")

    df = process_images_in_folder(folder, levels, output_file, n_jobs=-1, seed=random_seed)
    print(df)