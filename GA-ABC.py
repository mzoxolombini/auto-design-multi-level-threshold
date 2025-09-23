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


# ----------------- Utility Functions -----------------

def ssim(image1, image2, C1=0.01 ** 2, C2=0.03 ** 2):
    kernel = np.ones((3, 3)) / 9
    mu1 = correlate(image1, kernel, mode='reflect')
    mu2 = correlate(image2, kernel, mode='reflect')
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = correlate(image1 ** 2, kernel, mode='reflect') - mu1_sq
    sigma2_sq = correlate(image2 ** 2, kernel, mode='reflect') - mu2_sq
    sigma12 = correlate(image1 * image2, kernel, mode='reflect') - mu1_mu2
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(np.mean(numerator / (denominator + 1e-10)))


def calculate_entropy(hist, thresholds):
    thresholds = sorted(thresholds)
    total = np.sum(hist)
    probs = hist / total
    entropies = []
    prev_threshold = 0
    for threshold in thresholds + [len(hist)]:
        class_probs = probs[prev_threshold:threshold]
        class_sum = np.sum(class_probs)
        if class_sum > 0:
            entropies.append(-np.sum((class_probs / class_sum) * np.log(class_probs / class_sum + 1e-10)))
        else:
            entropies.append(0)
        prev_threshold = threshold
    return float(np.sum(entropies))


def calculate_otsu(hist, thresholds):
    thresholds = sorted(thresholds)
    total = np.sum(hist)
    probs = hist / total
    global_mean = np.sum(np.arange(len(hist)) * probs)
    between_class_variance = 0
    thresholds = [0] + thresholds + [len(hist)]
    for i in range(1, len(thresholds)):
        w_i = np.sum(probs[thresholds[i - 1]:thresholds[i]])
        if w_i > 0:
            mu_i = np.sum(np.arange(thresholds[i - 1], thresholds[i]) *
                          probs[thresholds[i - 1]:thresholds[i]]) / w_i
            between_class_variance += w_i * (mu_i - global_mean) ** 2
    return float(between_class_variance)


def calculate_mse(img1, img2):
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def calculate_psnr(mse_value, max_pixel=255.0):
    if mse_value == 0:
        return float('inf')
    return float(20 * math.log10(max_pixel / math.sqrt(mse_value)))


def calculate_uniformity(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist_normalized = hist / hist.sum()
    return float(np.sum(hist_normalized ** 2))


def apply_thresholds(image, thresholds):
    thresholds = sorted(thresholds)
    segmented = np.zeros_like(image)
    for i in range(len(thresholds) + 1):
        if i == 0:
            segmented[image <= thresholds[0]] = 0
        elif i == len(thresholds):
            segmented[image > thresholds[-1]] = len(thresholds)
        else:
            segmented[(image > thresholds[i - 1]) & (image <= thresholds[i])] = i
    if len(thresholds) > 0:
        segmented = (segmented * (255 // (len(thresholds) + 1))).astype(np.uint8)
    return segmented


def format_number(value, decimals=8):
    if isinstance(value, (int, np.integer)):
        return str(value)
    fmt = f"{value:.{decimals}f}"
    return fmt.replace('.', ',')


# ----------------- Pure ABC -----------------

class PureArtificialBeeColony:
    def __init__(self, config, num_thresholds, hist, fitness_function, seed=None):
        self.config = config
        self.num_thresholds = num_thresholds
        self.hist = hist
        self.fitness_function = fitness_function
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.population = [sorted(self.rng.sample(range(1, 255), num_thresholds))
                           for _ in range(config['pop_size'])]
        self.trials = [0] * config['pop_size']
        self.limit = int(config['pop_size'] * config['limit_ratio'])
        self.best_solution, self.best_fitness = None, float('-inf')

    def fitness(self, ind):
        return self.fitness_function(self.hist, ind)

    def evolve(self):
        for _ in range(self.config['max_generations']):
            self.employed()
            self.onlooker()
            self.scout()
        return self.best_solution, self.best_fitness

    def employed(self):
        for i in range(len(self.population)):
            cand = self.candidate(self.population[i], i)
            if self.fitness(cand) > self.fitness(self.population[i]):
                self.population[i], self.trials[i] = cand, 0
                if self.fitness(cand) > self.best_fitness:
                    self.best_solution, self.best_fitness = cand.copy(), self.fitness(cand)
            else:
                self.trials[i] += 1

    def onlooker(self):
        fits = np.array([self.fitness(ind) for ind in self.population])
        probs = fits / fits.sum() if fits.sum() > 0 else np.ones(len(fits)) / len(fits)
        for _ in range(len(self.population)):
            # Use numpy for weighted random choice
            idx = self.np_rng.choice(len(self.population), p=probs)
            cand = self.candidate(self.population[idx], idx)
            if self.fitness(cand) > self.fitness(self.population[idx]):
                self.population[idx], self.trials[idx] = cand, 0
                if self.fitness(cand) > self.best_fitness:
                    self.best_solution, self.best_fitness = cand.copy(), self.fitness(cand)

    def scout(self):
        for i in range(len(self.population)):
            if self.trials[i] >= self.limit:
                self.population[i] = sorted(self.rng.sample(range(1, 255), self.num_thresholds))
                self.trials[i] = 0

    def candidate(self, sol, idx):
        cand = sol.copy()
        d = self.rng.randint(0, len(cand) - 1)
        k = self.rng.choice([j for j in range(len(self.population)) if j != idx])
        phi = self.rng.uniform(-1, 1) * self.config['mutation_factor']
        cand[d] = int(np.clip(sol[d] + phi * (sol[d] - self.population[k][d]), 1, 254))
        return sorted(cand)


# ----------------- GA to configure ABC -----------------

class GAforABC:
    def __init__(self, ga_config, hist, fitness_function, num_thresholds, seed=None):
        self.ga_config = ga_config
        self.hist = hist
        self.fitness_function = fitness_function
        self.num_thresholds = num_thresholds
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.population = [
            [self.rng.randint(20, 80),  # pop_size
             self.rng.uniform(0.1, 1.0),  # mutation_factor
             self.rng.uniform(0.1, 0.5)]  # limit_ratio
            for _ in range(ga_config['pop_size'])
        ]

    def fitness(self, chromosome):
        pop_size, mutation_factor, limit_ratio = chromosome
        abc_config = {
            'pop_size': int(pop_size),
            'max_generations': 30,  # smaller for GA evaluation
            'mutation_factor': mutation_factor,
            'limit_ratio': limit_ratio
        }
        abc = PureArtificialBeeColony(abc_config, self.num_thresholds, self.hist, self.fitness_function,
                                      self.rng.randint(0, 1000000))
        _, best_fitness = abc.evolve()
        return best_fitness

    def evolve(self):
        best_chromosome = None
        best_score = float('-inf')
        for _ in range(self.ga_config['max_generations']):
            scores = [self.fitness(ind) for ind in self.population]
            ranked = sorted(zip(self.population, scores), key=lambda x: x[1], reverse=True)
            self.population = [x[0] for x in ranked]
            if ranked[0][1] > best_score:
                best_chromosome, best_score = ranked[0]

            # selection (top 50%)
            parents = self.population[:len(self.population) // 2]
            new_pop = parents.copy()

            # crossover + mutation
            while len(new_pop) < self.ga_config['pop_size']:
                p1, p2 = self.rng.sample(parents, 2)
                child = [
                    int((p1[0] + p2[0]) / 2 + self.rng.randint(-5, 5)),
                    max(0.1, min(1.0, (p1[1] + p2[1]) / 2 + self.rng.uniform(-0.1, 0.1))),
                    max(0.1, min(0.5, (p1[2] + p2[2]) / 2 + self.rng.uniform(-0.05, 0.05)))
                ]
                child[0] = max(10, min(100, child[0]))
                new_pop.append(child)
            self.population = new_pop

        return best_chromosome, best_score


# ----------------- Worker -----------------

def process_single(filename, folder_path, num_thresholds, fitness_function, abc_config, random_seed, images_folder):
    filepath = os.path.join(folder_path, filename)
    print(f"[START] {filename} | {fitness_function} | thresholds={num_thresholds}", flush=True, file=sys.stderr)

    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[SKIP] Could not load {filename}", flush=True, file=sys.stderr)
        return None

    # Generate unique seed for this specific run
    run_seed = int(time.time() * 1000) % 1000000
    random.seed(run_seed)
    np.random.seed(run_seed)

    # Start timing
    start_time = time.time()

    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    fitness_func = calculate_entropy if fitness_function == "kapur" else calculate_otsu

    abc = PureArtificialBeeColony(abc_config, num_thresholds, hist, fitness_func, run_seed)
    best_solution, best_fitness = abc.evolve()

    # Calculate execution time
    execution_time_ms = (time.time() - start_time) * 1000

    seg_img = apply_thresholds(image, best_solution)
    mse = calculate_mse(image, seg_img)
    psnr = calculate_psnr(mse)
    ssim_value = ssim(image, seg_img)
    uniformity_value = calculate_uniformity(seg_img)

    threshold_str = f"[{', '.join(map(str, sorted(best_solution)))}]"

    print(
        f"[DONE] {filename} | {fitness_function} | thresholds={num_thresholds} | fitness={best_fitness:.4f} | time={execution_time_ms:.2f}ms",
        flush=True, file=sys.stderr)

    # Save the thresholded image
    image_name_base = os.path.splitext(filename)[0]
    output_filename = f"{image_name_base}_{fitness_function}_{num_thresholds}_thresholds.png"
    output_image_path = os.path.join(images_folder, output_filename)
    cv2.imwrite(output_image_path, seg_img)

    return {
        'image_name': filename,
        'fitness_function': fitness_function.capitalize(),  # Capitalize first letter
        'thresholding_level': num_thresholds,
        'threshold_value': threshold_str,
        'fitness_value': format_number(best_fitness, 8),  # 8 decimal places
        'SSIM': format_number(ssim_value, 7),  # 7 decimal places
        'MSE': format_number(mse, 7),  # 7 decimal places
        'PSNR': format_number(psnr, 7),  # 7 decimal places
        'Uniformity Measure': format_number(uniformity_value, 7),  # 7 decimal places
        'Random Seed': run_seed,  # Use the unique run seed
        'Execution Time (ms)': format_number(execution_time_ms, 2)  # 2 decimal places
    }


# ----------------- Main Runner -----------------

def process_images_in_folder(folder_path, threshold_levels, output_path, n_jobs=-1):
    # Create images folder
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_GAABC")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    # Generate unique seed for GA configuration
    ga_seed = int(time.time() * 1000) % 1000000
    random.seed(ga_seed)
    np.random.seed(ga_seed)

    # Use first image to tune ABC via GA
    sample_img = cv2.imread(os.path.join(folder_path, files[0]), cv2.IMREAD_GRAYSCALE)
    hist = cv2.calcHist([sample_img], [0], None, [256], [0, 256]).flatten()

    ga_config = {'pop_size': 6, 'max_generations': 5}  # keep light
    ga = GAforABC(ga_config, hist, calculate_entropy, num_thresholds=3, seed=ga_seed)
    best_params, _ = ga.evolve()
    abc_config = {
        'pop_size': int(best_params[0]),
        'max_generations': 100,
        'mutation_factor': best_params[1],
        'limit_ratio': best_params[2]
    }
    print("Best ABC parameters from GA:", abc_config)

    tasks = [(f, folder_path, t, func, abc_config, ga_seed, images_folder)
             for f in files for func in ["kapur", "otsu"] for t in threshold_levels]
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single)(*task) for task in tasks
    )
    results = [r for r in results if r is not None]

    # Create DataFrame with the exact column order
    df = pd.DataFrame(results)

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
    output_file = os.path.join(output_path, "GAABC_results.xlsx")

    df.to_excel(output_file, index=False)
    print("Results saved to", output_file)
    return df


if __name__ == "__main__":
    folder = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images"
    output = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images\results"
    levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    df = process_images_in_folder(folder, levels, output, n_jobs=-1)