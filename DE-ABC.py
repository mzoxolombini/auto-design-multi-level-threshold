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
    """Compute normalized histogram of grayscale image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    return hist / hist.sum()


def kapur_entropy(hist, thresholds):
    """Compute Kapur's entropy for given thresholds."""
    thresholds = sorted(thresholds)
    bins = [0] + thresholds + [len(hist)]
    total_entropy = 0

    # Precompute cumulative sums to avoid repeated work
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
    """Compute Otsu's between-class variance for given thresholds."""
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
    """Calculate Structural Similarity Index (SSIM) between original and segmented images."""
    return ssim(original, segmented, data_range=segmented.max() - segmented.min())


def calculate_mse(original, segmented):
    """Calculate Mean Squared Error (MSE) between original and segmented images."""
    return mean_squared_error(original, segmented)


def calculate_psnr(original, segmented):
    """Calculate Peak Signal-to-Noise Ratio (PSNR) between original and segmented images."""
    mse = calculate_mse(original, segmented)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))


def calculate_uniformity_measure(segmented, thresholds):
    """Calculate Uniformity Measure for segmented image."""
    if len(thresholds) == 0:
        return 0

    # Create a binary mask for each region
    uniformity = 0
    total_pixels = segmented.size

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

            # Calculate sum of squared differences from mean
            squared_diff = np.sum((segmented[mask] - region_mean) ** 2)

            # Update uniformity measure
            uniformity += 1 - (squared_diff / (region_pixels * (upper - lower) ** 2))

    return uniformity


def apply_thresholds(image, thresholds):
    """Apply thresholds to image to create segmented image."""
    if len(thresholds) == 0:
        return image

    # Sort thresholds
    thresholds = sorted(thresholds)

    # Create segmented image
    segmented = np.zeros_like(image)

    # First region: 0 to first threshold
    mask = image <= thresholds[0]
    segmented[mask] = 0

    # Middle regions
    for i in range(len(thresholds) - 1):
        mask = (image > thresholds[i]) & (image <= thresholds[i + 1])
        segmented[mask] = thresholds[i]

    # Last region: last threshold to 255
    mask = image > thresholds[-1]
    segmented[mask] = thresholds[-1]

    return segmented


# ===============================
# Artificial Bee Colony
# ===============================

class PureArtificialBeeColony:
    def __init__(self, config, num_thresholds, hist, fitness_func, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.pop_size = config['pop_size']
        self.max_generations = config['max_generations']
        self.limit = int(config['limit_ratio'] * self.pop_size * num_thresholds)
        self.mutation_factor = config['mutation_factor']
        self.num_thresholds = num_thresholds
        self.hist = hist
        self.fitness_func = fitness_func

        self.population = [np.sort(np.random.randint(0, 256, num_thresholds))
                           for _ in range(self.pop_size)]
        self.fitness = [self.fitness_func(self.hist, ind) for ind in self.population]
        self.trial_counters = np.zeros(self.pop_size)

    def evolve(self):
        for _ in range(self.max_generations):
            for i in range(self.pop_size):
                k = np.random.choice([j for j in range(self.pop_size) if j != i])
                phi = np.random.uniform(-1, 1, self.num_thresholds)
                candidate = np.clip(
                    self.population[i] + phi * (self.population[i] - self.population[k]),
                    0, 255
                ).astype(int)
                candidate.sort()
                candidate_fitness = self.fitness_func(self.hist, candidate)

                if candidate_fitness > self.fitness[i]:
                    self.population[i] = candidate
                    self.fitness[i] = candidate_fitness
                    self.trial_counters[i] = 0
                else:
                    self.trial_counters[i] += 1

            # Scout phase
            for i in range(self.pop_size):
                if self.trial_counters[i] > self.limit:
                    self.population[i] = np.sort(np.random.randint(0, 256, self.num_thresholds))
                    self.fitness[i] = self.fitness_func(self.hist, self.population[i])
                    self.trial_counters[i] = 0

        best_index = np.argmax(self.fitness)
        return self.population[best_index], self.fitness[best_index]


# ===============================
# Differential Evolution Configurator
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
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        hist = calculate_histogram(image)
        bounds = {
            'pop_size': (20, 80),
            'max_generations': (30, 150),
            'mutation_factor': (0.3, 0.8),
            'limit_ratio': (0.1, 0.3)
        }

        population = []
        for _ in range(self.de_pop_size):
            population.append({
                'pop_size': random.randint(*bounds['pop_size']),
                'max_generations': random.randint(*bounds['max_generations']),
                'mutation_factor': random.uniform(*bounds['mutation_factor']),
                'limit_ratio': random.uniform(*bounds['limit_ratio'])
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
        abc = PureArtificialBeeColony(config, num_thresholds, hist, fitness_func, seed)
        _, best_fit = abc.evolve()
        return best_fit


# ===============================
# Image Processing
# ===============================

def process_single(file_name, folder_path, num_thresholds, func_name, de_configurator, seed=None, images_folder=None):
    try:
        # Start timing
        start_time = time.time()

        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        # Calculate histogram once for the image
        hist = calculate_histogram(image)

        fitness_function = kapur_entropy if func_name == "kapur" else otsu_between_class_variance
        optimal_cfg = de_configurator.optimize_abc_params(num_thresholds, image, fitness_function, seed)

        abc = PureArtificialBeeColony(optimal_cfg, num_thresholds, hist, fitness_function, seed)
        best_thresholds, best_score = abc.evolve()

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Apply thresholds to create segmented image
        segmented_image = apply_thresholds(image, best_thresholds)

        # Calculate both fitness values
        kapur_value = kapur_entropy(hist, best_thresholds)
        otsu_value = otsu_between_class_variance(hist, best_thresholds)

        # Calculate additional metrics
        ssim_value = calculate_ssim(image, segmented_image)
        mse_value = calculate_mse(image, segmented_image)
        psnr_value = calculate_psnr(image, segmented_image)
        uniformity = calculate_uniformity_measure(segmented_image, best_thresholds)

        # Format threshold values as comma-separated values within square brackets
        threshold_value_str = "[" + ", ".join(map(str, best_thresholds)) + "]"

        # Save the thresholded image
        if images_folder:
            image_name_base = os.path.splitext(file_name)[0]
            output_filename = f"{image_name_base}_{func_name}_{num_thresholds}_thresholds.png"
            output_image_path = os.path.join(images_folder, output_filename)
            cv2.imwrite(output_image_path, segmented_image)

        return {
            "image_name": file_name,
            "fitness_function": func_name,
            "thresholding_level": num_thresholds,
            "threshold_value": threshold_value_str,
            "fitness_value": best_score,
            "Kapur_Value": kapur_value,
            "Otsu_Value": otsu_value,
            "SSIM": ssim_value,
            "MSE": mse_value,
            "PSNR": psnr_value,
            "Uniformity Measure": uniformity,
            "Random Seed": seed,
            "Execution Time (ms)": execution_time_ms
        }
    except Exception as e:
        return {"image_name": file_name, "fitness_function": func_name, "error": str(e)}


def process_images_in_folder(folder_path, threshold_levels, output_file, n_jobs=-1, seed=None):
    # Create images folder
    output_path = os.path.dirname(output_file)
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_DEABC")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Generate a random seed if not provided
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)

    np.random.seed(seed)
    random.seed(seed)

    de_configurator = DEABCConfigurator(seed=seed)

    tasks = [(f, folder_path, t, func, de_configurator, seed, images_folder)
             for f in files for func in ["kapur", "otsu"] for t in threshold_levels]

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single)(*task) for task in tasks
    )

    results = [r for r in results if r is not None]

    # Filter out error results
    valid_results = [r for r in results if 'error' not in r]

    if valid_results:
        df = pd.DataFrame(valid_results)

        # Format the numeric columns for better Excel display
        df['fitness_value'] = df['fitness_value'].apply(lambda x: f"{x:,.8f}")
        df['Kapur_Value'] = df['Kapur_Value'].apply(lambda x: f"{x:,.8f}")
        df['Otsu_Value'] = df['Otsu_Value'].apply(lambda x: f"{x:,.8f}")
        df['SSIM'] = df['SSIM'].apply(lambda x: f"{x:,.7f}")
        df['MSE'] = df['MSE'].apply(lambda x: f"{x:,.7f}")
        df['PSNR'] = df['PSNR'].apply(lambda x: f"{x:,.7f}")
        df['Uniformity Measure'] = df['Uniformity Measure'].apply(lambda x: f"{x:,.7f}")
        df['Execution Time (ms)'] = df['Execution Time (ms)'].apply(lambda x: f"{x:,.2f}")

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
    output_file = os.path.join(output_dir, "DEABC_results.xlsx")

    # Generate a random seed for this run
    random_seed = random.randint(0, 2 ** 32 - 1)
    print(f"Using random seed: {random_seed}")

    df = process_images_in_folder(folder, levels, output_file, n_jobs=-1, seed=random_seed)
    print(df)