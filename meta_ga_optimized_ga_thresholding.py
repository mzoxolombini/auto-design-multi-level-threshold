import os
import cv2
import numpy as np
import random
import pandas as pd
from scipy.ndimage import correlate
import math
import time
import json
import locale
from skimage.metrics import structural_similarity as ssim

# Set locale for comma decimal separator
try:
    locale.setlocale(locale.LC_NUMERIC, 'de_DE' if os.name == 'nt' else 'de_DE.UTF-8')
except Exception:
    pass


# Corrected Metrics Functions
def calculate_mse(image1, image2):
    """Calculate Mean Squared Error"""
    # Convert to float64 to avoid integer overflow
    image1_float = image1.astype(np.float64)
    image2_float = image2.astype(np.float64)
    return float(np.mean((image1_float - image2_float) ** 2))


def calculate_psnr(mse_value, max_pixel=255.0):
    """Calculate Peak Signal-to-Noise Ratio (corrected formula)"""
    if mse_value == 0:
        return float('inf')
    return 10 * math.log10((max_pixel ** 2) / mse_value)


def calculate_ssim_value(image1, image2):
    """Calculate Structural Similarity Index with correct data range"""
    return float(ssim(image1, image2, data_range=255))


def calculate_uniformity(image):
    """Calculate Histogram Uniformity Measure"""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    s = hist.sum()
    if s == 0:
        return 0.0
    hist_normalized = hist / s
    return float(np.sum(hist_normalized ** 2))


def validate_metrics(ssim_val, mse_val, psnr_val, uniformity_val, filename=""):
    """Validate that all metrics are within expected ranges"""
    if not (-1 <= ssim_val <= 1):
        print(f"Warning: SSIM value {ssim_val:.6f} out of range for {filename}")
        ssim_val = max(-1.0, min(1.0, ssim_val))  # Clamp to valid range

    if mse_val < 0:
        print(f"Warning: MSE value {mse_val:.6f} negative for {filename}")
        mse_val = max(0.0, mse_val)

    if psnr_val < 0 and psnr_val != float('inf'):
        print(f"Warning: PSNR value {psnr_val:.6f} invalid for {filename}")

    if not (0 <= uniformity_val <= 1):
        print(f"Warning: Uniformity value {uniformity_val:.6f} out of range for {filename}")
        uniformity_val = max(0.0, min(1.0, uniformity_val))

    return ssim_val, mse_val, psnr_val, uniformity_val


# Kapur entropy
def calculate_entropy(hist, thresholds):
    thresholds = sorted(thresholds)
    total = np.sum(hist)
    if total == 0:
        return 0.0
    probs = hist / total

    entropies = []
    prev = 0
    for t in thresholds + [len(hist)]:
        class_probs = probs[prev:t]
        class_sum = np.sum(class_probs)
        if class_sum > 0:
            parts = class_probs / class_sum
            ent = -np.sum(parts * np.log(parts + 1e-10))
            entropies.append(ent)
        else:
            entropies.append(0.0)
        prev = t

    return float(np.sum(entropies))


# Otsu between-class variance
def calculate_otsu(hist, thresholds):
    thresholds = sorted(thresholds)
    total = np.sum(hist)
    if total == 0:
        return 0.0
    probs = hist / total
    bins = np.arange(len(hist))
    global_mean = float(np.sum(bins * probs))

    between_var = 0.0
    thr = [0] + thresholds + [len(hist)]
    for i in range(1, len(thr)):
        w = np.sum(probs[thr[i - 1]:thr[i]])
        if w > 0:
            mu = float(np.sum(bins[thr[i - 1]:thr[i]] * probs[thr[i - 1]:thr[i]]) / w)
            between_var += w * (mu - global_mean) ** 2
    return float(between_var)


def apply_thresholds(image, thresholds):
    thresholds = sorted(thresholds)
    segmented = np.zeros_like(image, dtype=np.uint8)

    for i in range(len(thresholds) + 1):
        if i == 0:
            mask = image <= thresholds[0]
            segmented[mask] = 0
        elif i == len(thresholds):
            mask = image > thresholds[-1]
            segmented[mask] = len(thresholds)
        else:
            mask = (image > thresholds[i - 1]) & (image <= thresholds[i])
            segmented[mask] = i

    if len(thresholds) > 0:
        multiplier = 255 // (len(thresholds) + 1)
        segmented = (segmented * multiplier).astype(np.uint8)
    return segmented


def format_number(value, decimals=8):
    if isinstance(value, (int, np.integer)):
        return str(value)
    fmt = f"{value:.{decimals}f}"
    return fmt.replace('.', ',')


class GeneticAlgorithm:
    def __init__(self, image, num_thresholds, objective,
                 population_size=50, generations=50,
                 crossover_rate=0.8, mutation_rate=0.1,
                 tournament_size=3, seed=None):

        self.image = image
        self.num_thresholds = num_thresholds
        self.objective = objective
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.rng = random.Random(seed)
        self.hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        self.population = self.initialize_population()
        self.early_stop_min_delta = 1e-6
        self.early_stop_patience = 10

    def initialize_population(self):
        return [sorted(self.rng.sample(range(1, 255), self.num_thresholds))
                for _ in range(self.population_size)]

    def fitness(self, individual):
        if self.objective == 'kapur':
            return calculate_entropy(self.hist, individual)
        elif self.objective == 'otsu':
            return calculate_otsu(self.hist, individual)
        raise ValueError('Unknown fitness function')

    def tournament_pick(self, population, fitness_cache):
        k = min(self.tournament_size, len(population))
        contenders = self.rng.sample(population, k)
        best = max(contenders, key=lambda ind: fitness_cache[tuple(ind)])
        return best

    def crossover(self, parent1, parent2):
        if self.rng.random() < self.crossover_rate and self.num_thresholds > 1:
            idx = self.rng.randint(1, self.num_thresholds - 1)
            c1 = sorted(parent1[:idx] + parent2[idx:])
            c2 = sorted(parent2[:idx] + parent1[idx:])
            return c1, c2
        return parent1[:], parent2[:]

    def mutate(self, individual):
        if self.rng.random() < self.mutation_rate:
            idx = self.rng.randint(0, self.num_thresholds - 1)
            individual[idx] = self.rng.randint(1, 254)
            individual.sort()
        return individual

    def evolve(self):
        fitness_cache = {tuple(ind): self.fitness(ind) for ind in self.population}
        best = max(self.population, key=lambda ind: fitness_cache[tuple(ind)])
        best_score = fitness_cache[tuple(best)]
        no_improve = 0

        for generation in range(self.generations):
            new_population = []
            while len(new_population) < self.population_size:
                p1 = self.tournament_pick(self.population, fitness_cache)
                p2 = self.tournament_pick(self.population, fitness_cache)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                for child in (c1, c2):
                    key = tuple(child)
                    if key not in fitness_cache:
                        fitness_cache[key] = self.fitness(child)
                new_population.extend([c1, c2])

            all_pop = new_population + [best]
            unique = []
            seen = set()
            for ind in all_pop:
                t = tuple(ind)
                if t not in seen:
                    seen.add(t)
                    unique.append(ind)

            unique.sort(key=lambda ind: fitness_cache[tuple(ind)], reverse=True)
            self.population = unique[:self.population_size]

            current_best = self.population[0]
            current_score = fitness_cache[tuple(current_best)]

            if current_score > best_score + self.early_stop_min_delta:
                best = current_best
                best_score = current_score
                no_improve = 0
            else:
                no_improve += 1

            if (generation + 1) % 10 == 0:
                print(f"    Generation {generation + 1}/{self.generations}, Best Fitness {best_score:.6f}")

            if no_improve >= self.early_stop_patience:
                print(f"    Early stop at generation {generation + 1}. Best Fitness {best_score:.6f}")
                break

        return best, best_score


class MetaGeneticAlgorithm:
    def __init__(self, image, num_thresholds, fitness_function='kapur', seed=None):
        self.pop_size = 8
        self.max_generations = 4
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.image = image
        self.num_thresholds = num_thresholds
        self.fitness_function = fitness_function
        self.param_space = {
            'pop_size': (30, 60),
            'crossover_rate': (0.6, 0.9),
            'mutation_rate': (0.02, 0.25),
            'max_generations': (40, 80)
        }
        self.population = self.initialize_population()
        self.eval_cache = {}

    def initialize_population(self):
        return [
            {
                'pop_size': self.rng.randint(*self.param_space['pop_size']),
                'crossover_rate': self.rng.uniform(*self.param_space['crossover_rate']),
                'mutation_rate': self.rng.uniform(*self.param_space['mutation_rate']),
                'max_generations': self.rng.randint(*self.param_space['max_generations'])
            }
            for _ in range(self.pop_size)
        ]

    def _key(self, config):
        return (int(config['pop_size']), round(float(config['crossover_rate']), 4),
                round(float(config['mutation_rate']), 4), int(config['max_generations']))

    def evaluate(self, config):
        key = self._key(config)
        if key in self.eval_cache:
            return self.eval_cache[key]

        ga = GeneticAlgorithm(
            image=self.image,
            num_thresholds=self.num_thresholds,
            objective=self.fitness_function,
            population_size=config["pop_size"],
            generations=config["max_generations"],
            crossover_rate=config["crossover_rate"],
            mutation_rate=config["mutation_rate"],
            tournament_size=3,
            seed=self.seed
        )

        best_solution, best_fitness = ga.evolve()
        self.eval_cache[key] = best_fitness
        return best_fitness

    def select_parent(self):
        contenders = self.rng.sample(self.population, 3)
        scored = [(cfg, self.evaluate(cfg)) for cfg in contenders]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def crossover(self, p1, p2):
        child = {}
        for k in self.param_space.keys():
            child[k] = p1[k] if self.rng.random() < 0.5 else p2[k]
        return child

    def mutate(self, individual):
        if self.rng.random() < 0.3:
            param = self.rng.choice(list(self.param_space.keys()))
            if isinstance(self.param_space[param][0], int):
                individual[param] = self.rng.randint(*self.param_space[param])
            else:
                individual[param] = self.rng.uniform(*self.param_space[param])
        return individual

    def evolve(self):
        print(f"  Meta-GA optimizing {self.fitness_function} with {self.num_thresholds} thresholds")
        for gen in range(self.max_generations):
            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = self.select_parent()
                p2 = self.select_parent()
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)

            scored = [(cfg, self.evaluate(cfg)) for cfg in new_pop]
            scored.sort(key=lambda x: x[1], reverse=True)
            self.population = [cfg for cfg, _ in scored[:self.pop_size]]
            best_cfg, best_score = scored[0]
            print(f"    Meta-GA Gen {gen + 1}/{self.max_generations} Best Score {best_score:.6f}")
        return self.population[0]


def process_images_in_folder(folder_path, threshold_levels, output_path, runs_per_threshold=1):
    results = []

    # Create images folder
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_GAGA")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    total_images = len(image_files)
    total_ops = total_images * len(threshold_levels) * 2 * runs_per_threshold
    processed = 0

    # Check if there are any images to process
    if total_images == 0:
        print("No images found in the folder!")
        return pd.DataFrame()  # Return empty DataFrame instead of None

    for filename in image_files:
        print('\n' + '=' * 60)
        print(f"PROCESSING IMAGE: {filename}")
        print('=' * 60)

        filepath = os.path.join(folder_path, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not load image {filename}")
            continue

        # Calculate histogram once for the image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

        for fitness_function in ['kapur', 'otsu']:
            for num_thresholds in threshold_levels:
                for run in range(runs_per_threshold):
                    processed += 1
                    print(
                        f"\n[{processed}/{total_ops}] {filename} {num_thresholds} thresholds {fitness_function} run {run + 1}")

                    # Generate unique seed for this specific run
                    run_seed = int(time.time() * 1000) % 1000000  # More unique seed
                    random.seed(run_seed)
                    np.random.seed(run_seed)

                    # Start timing
                    start_time = time.time()

                    try:
                        meta_ga = MetaGeneticAlgorithm(image=image,
                                                       num_thresholds=num_thresholds,
                                                       fitness_function=fitness_function,
                                                       seed=run_seed)
                        best_config = meta_ga.evolve()
                        print(f"  Best GA config: {best_config}")

                        ga = GeneticAlgorithm(
                            image=image,
                            num_thresholds=num_thresholds,
                            objective=fitness_function,
                            population_size=int(best_config['pop_size']),
                            generations=int(best_config['max_generations']),
                            crossover_rate=float(best_config['crossover_rate']),
                            mutation_rate=float(best_config['mutation_rate']),
                            tournament_size=3,
                            seed=run_seed
                        )

                        best_solution, best_fitness = ga.evolve()
                        execution_time_ms = (time.time() - start_time) * 1000

                        thresholded_image = apply_thresholds(image, best_solution)

                        # Use corrected metric functions
                        mse_value = calculate_mse(image, thresholded_image)
                        psnr_value = calculate_psnr(mse_value)
                        ssim_value = calculate_ssim_value(image, thresholded_image)
                        uniformity_value = calculate_uniformity(thresholded_image)

                        # Validate metrics
                        ssim_value, mse_value, psnr_value, uniformity_value = validate_metrics(
                            ssim_value, mse_value, psnr_value, uniformity_value, filename
                        )

                        threshold_str = f"[{', '.join(map(str, sorted(best_solution)))}]"

                        # Format the results exactly as requested
                        results.append({
                            'image_name': filename,
                            'fitness_function': fitness_function.capitalize(),  # Capitalize first letter
                            'thresholding_level': num_thresholds,
                            'threshold_value': threshold_str,
                            'fitness_value': format_number(best_fitness, 8),  # 8 decimal places
                            'SSIM': format_number(ssim_value, 7),  # 7 decimal places
                            'MSE': format_number(mse_value, 7),  # 7 decimal places
                            'PSNR': format_number(psnr_value, 7),  # 7 decimal places
                            'Uniformity Measure': format_number(uniformity_value, 7),  # 7 decimal places
                            'Random Seed': run_seed,
                            'Execution Time (ms)': format_number(execution_time_ms, 2)  # 2 decimal places with comma
                        })

                        print(
                            f"  Completed. Fitness {best_fitness:.6f}, SSIM {ssim_value:.6f}, Time {execution_time_ms:.2f}ms, Seed {run_seed}")

                        # Save the thresholded image
                        image_name_base = os.path.splitext(filename)[0]
                        output_filename = f"{image_name_base}_{fitness_function}_{num_thresholds}_thresholds.png"
                        output_image_path = os.path.join(images_folder, output_filename)
                        cv2.imwrite(output_image_path, thresholded_image)

                    except Exception as e:
                        print(f"  ERROR processing {filename}: {str(e)}")
                        continue

    # Create DataFrame with the exact column order
    df = pd.DataFrame(results)

    # Only create Excel file if there are results
    if len(results) > 0:
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

        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, 'GAGA_Results_corrected.xlsx')

        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            print(f"\nResults saved to {output_file}")
        except Exception as e:
            print(f"Error saving Excel file: {e}")
            # Still return the DataFrame even if Excel save fails

        # Print summary of metric ranges for verification
        print("\n=== GAGA Metric Ranges Summary ===")

        # Convert back to float for range calculation
        ssim_values = [float(r['SSIM'].replace(',', '.')) for r in results]
        mse_values = [float(r['MSE'].replace(',', '.')) for r in results]
        psnr_values = [float(r['PSNR'].replace(',', '.')) for r in results]
        uniformity_values = [float(r['Uniformity Measure'].replace(',', '.')) for r in results]

        print(f"SSIM range: [{min(ssim_values):.6f}, {max(ssim_values):.6f}]")
        print(f"MSE range: [{min(mse_values):.2f}, {max(mse_values):.2f}]")
        print(f"PSNR range: [{min(psnr_values):.2f}, {max(psnr_values):.2f}]")
        print(f"Uniformity range: [{min(uniformity_values):.6f}, {max(uniformity_values):.6f}]")
    else:
        print("No results were generated!")

    return df  # Always return the DataFrame (even if empty)


if __name__ == '__main__':
    folder_path = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images"
    output_path = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images\results"
    quick_test = True

    if quick_test:
        threshold_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        runs_per_threshold = 1
    else:
        threshold_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        runs_per_threshold = 1

    os.makedirs(output_path, exist_ok=True)

    print('Starting optimized GA thresholding')
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    print(f"Images to process: {len(image_files)}")

    results_df = process_images_in_folder(folder_path, threshold_levels, output_path, runs_per_threshold)

    print('\nProcessing completed')

    # Check if results_df is not None and not empty
    if results_df is not None and len(results_df) > 0:
        print(f"Total results: {len(results_df)}")

        for fitness_func in ['kapur', 'otsu']:
            for threshold in threshold_levels:
                count = len(results_df[(results_df['fitness_function'] == fitness_func.capitalize()) &
                                       (results_df['thresholding_level'] == threshold)])
                print(f"{fitness_func.capitalize()} with {threshold} thresholds: {count} runs")
    else:
        print("No results were generated!")