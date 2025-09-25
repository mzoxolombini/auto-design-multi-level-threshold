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
    """
    Calculate Structural Similarity Index between two images.
    Returns value in range [-1, 1], typically [0, 1].
    """
    # Normalize to [0,1] range
    img1 = image1.astype(np.float64) / 255.0
    img2 = image2.astype(np.float64) / 255.0

    # Constants for normalized images [0,1]
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Gaussian kernel
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

    # Ensure numerical stability
    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-10)
    return float(np.clip(np.mean(ssim_map), -1.0, 1.0))


def calculate_entropy(hist, thresholds):
    """Calculate Kapur's entropy for multilevel thresholding."""
    if len(thresholds) == 0:
        return 0.0

    thresholds = sorted(thresholds)
    total = np.sum(hist)
    if total == 0:
        return 0.0

    probs = hist / total
    entropies = []
    prev_threshold = 0

    for threshold in thresholds + [len(hist)]:
        class_probs = probs[prev_threshold:threshold]
        class_sum = np.sum(class_probs)

        if class_sum > 1e-10:
            # Remove zero probabilities to avoid log(0)
            class_probs_nonzero = class_probs[class_probs > 1e-10]
            if len(class_probs_nonzero) > 0:
                entropy_val = -np.sum((class_probs_nonzero / class_sum) *
                                      np.log(class_probs_nonzero / class_sum))
                entropies.append(entropy_val)
            else:
                entropies.append(0.0)
        else:
            entropies.append(0.0)
        prev_threshold = threshold

    return float(np.sum(entropies))


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
        w_i = np.sum(probs[thresholds[i - 1]:thresholds[i]])
        if w_i > 1e-10:
            mu_i = np.sum(np.arange(thresholds[i - 1], thresholds[i]) *
                          probs[thresholds[i - 1]:thresholds[i]]) / w_i
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

    # First class: values <= first threshold
    segmented[image <= thresholds[0]] = 0

    # Middle classes
    for i in range(1, len(thresholds)):
        mask = (image > thresholds[i - 1]) & (image <= thresholds[i])
        segmented[mask] = i

    # Last class: values > last threshold
    segmented[image > thresholds[-1]] = len(thresholds)

    # Normalize to 0-255 for display
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
    return fmt.replace('.', ',')


# ----------------- Pure ABC Algorithm -----------------

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
        """Calculate fitness of an individual."""
        return self.fitness_function(self.hist, ind)

    def evolve(self):
        """Run the ABC optimization process."""
        for generation in range(self.config['max_generations']):
            self.employed()
            self.onlooker()
            self.scout()

            # Print progress every 10 generations
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness

    def employed(self):
        """Employed bee phase."""
        for i in range(len(self.population)):
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

    def onlooker(self):
        """Onlooker bee phase."""
        fits = np.array([self.fitness(ind) for ind in self.population])
        # Handle negative fitness values
        fits = fits - fits.min() + 1e-10
        probs = fits / fits.sum()

        for _ in range(len(self.population)):
            idx = self.np_rng.choice(len(self.population), p=probs)
            cand = self.candidate(self.population[idx], idx)
            cand_fitness = self.fitness(cand)
            current_fitness = self.fitness(self.population[idx])

            if cand_fitness > current_fitness:
                self.population[idx] = cand
                self.trials[idx] = 0
                if cand_fitness > self.best_fitness:
                    self.best_solution = cand.copy()
                    self.best_fitness = cand_fitness

    def scout(self):
        """Scout bee phase."""
        for i in range(len(self.population)):
            if self.trials[i] >= self.limit:
                self.population[i] = sorted(self.rng.sample(range(1, 255), self.num_thresholds))
                self.trials[i] = 0

    def candidate(self, sol, idx):
        """Generate a candidate solution."""
        cand = sol.copy()
        d = self.rng.randint(0, len(cand))
        k = self.rng.choice([j for j in range(len(self.population)) if j != idx])
        phi = self.rng.uniform(-1, 1) * self.config['mutation_factor']
        cand[d] = int(np.clip(sol[d] + phi * (sol[d] - self.population[k][d]), 1, 254))
        return sorted(cand)


# ----------------- GA for ABC Configuration -----------------

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
        """Evaluate fitness of a GA chromosome."""
        pop_size, mutation_factor, limit_ratio = chromosome
        abc_config = {
            'pop_size': int(pop_size),
            'max_generations': 30,  # Smaller for faster GA evaluation
            'mutation_factor': mutation_factor,
            'limit_ratio': limit_ratio
        }

        try:
            abc = PureArtificialBeeColony(abc_config, self.num_thresholds, self.hist,
                                          self.fitness_function, self.rng.randint(0, 1000000))
            _, best_fitness = abc.evolve()
            return best_fitness
        except:
            return float('-inf')  # Return worst fitness if error occurs

    def evolve(self):
        """Run GA evolution to find best ABC parameters."""
        best_chromosome = None
        best_score = float('-inf')

        for generation in range(self.ga_config['max_generations']):
            # Evaluate population
            scores = []
            for ind in self.population:
                score = self.fitness(ind)
                scores.append(score)

            # Rank population
            ranked = sorted(zip(self.population, scores), key=lambda x: x[1], reverse=True)
            self.population = [x[0] for x in ranked]
            scores_sorted = [x[1] for x in ranked]

            # Update best solution
            if scores_sorted[0] > best_score:
                best_chromosome = self.population[0].copy()
                best_score = scores_sorted[0]
                print(f"GA Generation {generation}: Best score = {best_score:.6f}")

            # Selection (top 50%)
            elite_size = len(self.population) // 2
            parents = self.population[:elite_size]
            new_population = parents.copy()

            # Crossover and mutation
            while len(new_population) < self.ga_config['pop_size']:
                p1, p2 = self.rng.sample(parents, 2)
                child = [
                    int((p1[0] + p2[0]) / 2 + self.rng.randint(-5, 5)),  # pop_size
                    max(0.1, min(1.0, (p1[1] + p2[1]) / 2 + self.rng.uniform(-0.1, 0.1))),  # mutation_factor
                    max(0.1, min(0.5, (p1[2] + p2[2]) / 2 + self.rng.uniform(-0.05, 0.05)))  # limit_ratio
                ]
                child[0] = max(10, min(100, child[0]))  # Ensure pop_size is within bounds
                new_population.append(child)

            self.population = new_population

        return best_chromosome, best_score


# ----------------- Worker Function -----------------

def process_single(filename, folder_path, num_thresholds, fitness_function, abc_config, random_seed, images_folder):
    """Process a single image with given parameters."""
    filepath = os.path.join(folder_path, filename)
    print(f"[START] {filename} | {fitness_function} | thresholds={num_thresholds}", flush=True, file=sys.stderr)

    # Load image
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[SKIP] Could not load {filename}", flush=True, file=sys.stderr)
        return None

    # Generate unique seed for this run
    run_seed = (random_seed + hash(filename) + num_thresholds) % 1000000
    random.seed(run_seed)
    np.random.seed(run_seed)

    # Start timing
    start_time = time.time()

    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

    # Select fitness function
    if fitness_function == "kapur":
        fitness_func = calculate_entropy
    else:  # "otsu"
        fitness_func = calculate_otsu

    # Run ABC optimization
    abc = PureArtificialBeeColony(abc_config, num_thresholds, hist, fitness_func, run_seed)
    best_solution, best_fitness = abc.evolve()

    # Calculate execution time
    execution_time_ms = (time.time() - start_time) * 1000

    # Apply thresholds and calculate metrics
    seg_img = apply_thresholds(image, best_solution)
    mse = calculate_mse(image, seg_img)
    psnr = calculate_psnr(mse)
    ssim_value = ssim(image, seg_img)
    uniformity_value = calculate_uniformity(seg_img)

    # Format thresholds as string
    threshold_str = f"[{', '.join(map(str, sorted(best_solution)))}]"

    print(f"[DONE] {filename} | {fitness_function} | thresholds={num_thresholds} | "
          f"fitness={best_fitness:.4f} | time={execution_time_ms:.2f}ms",
          flush=True, file=sys.stderr)

    # Save segmented image
    image_name_base = os.path.splitext(filename)[0]
    output_filename = f"{image_name_base}_{fitness_function}_{num_thresholds}_thresholds.png"
    output_image_path = os.path.join(images_folder, output_filename)
    cv2.imwrite(output_image_path, seg_img)

    return {
        'image_name': filename,
        'fitness_function': fitness_function.capitalize(),
        'thresholding_level': num_thresholds,
        'threshold_value': threshold_str,
        'fitness_value': format_number(best_fitness, 8),
        'SSIM': format_number(ssim_value, 7),
        'MSE': format_number(mse, 7),
        'PSNR': format_number(psnr, 7),
        'Uniformity Measure': format_number(uniformity_value, 7),
        'Random Seed': run_seed,
        'Execution Time (ms)': format_number(execution_time_ms, 2)
    }


# ----------------- Main Runner -----------------

def process_images_in_folder(folder_path, threshold_levels, output_path, n_jobs=-1):
    """Main function to process all images in a folder."""
    # Create output directories
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_GAABC")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # Get image files
    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    if not files:
        print("No image files found in the specified folder.")
        return None

    print(f"Found {len(files)} images to process.")

    # Generate unique seed for GA configuration
    ga_seed = int(time.time() * 1000) % 1000000
    random.seed(ga_seed)
    np.random.seed(ga_seed)

    # Use first image to tune ABC via GA
    sample_file = files[0]
    sample_img = cv2.imread(os.path.join(folder_path, sample_file), cv2.IMREAD_GRAYSCALE)
    if sample_img is None:
        print(f"Could not load sample image {sample_file}. Using default parameters.")
        abc_config = {
            'pop_size': 50,
            'max_generations': 100,
            'mutation_factor': 0.5,
            'limit_ratio': 0.3
        }
    else:
        hist = cv2.calcHist([sample_img], [0], None, [256], [0, 256]).flatten()

        # Configure GA for ABC parameter tuning
        ga_config = {'pop_size': 6, 'max_generations': 5}  # Keep light for speed
        ga = GAforABC(ga_config, hist, calculate_entropy, num_thresholds=3, seed=ga_seed)
        best_params, _ = ga.evolve()

        abc_config = {
            'pop_size': int(best_params[0]),
            'max_generations': 100,
            'mutation_factor': best_params[1],
            'limit_ratio': best_params[2]
        }
        print("Best ABC parameters from GA:", abc_config)

    # Create tasks for parallel processing
    tasks = []
    for f in files:
        for t in threshold_levels:
            for func in ["kapur", "otsu"]:
                tasks.append((f, folder_path, t, func, abc_config, ga_seed, images_folder))

    print(f"Processing {len(tasks)} tasks with {n_jobs} workers...")

    # Process images in parallel
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single)(*task) for task in tasks
    )

    # Filter out None results
    results = [r for r in results if r is not None]

    if not results:
        print("No results obtained. Check if images were processed correctly.")
        return None

    # Create DataFrame
    df = pd.DataFrame(results)

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

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "GAABC_results.xlsx")

    # Save results
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(f"Processed {len(results)} image configurations")

    return df


# ----------------- Main Execution -----------------

if __name__ == "__main__":
    # Configuration
    folder = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images"
    output = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images\results"
    levels = [2, 3, 4, 5]  # Reduced for testing, you can expand to [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Process images
    df = process_images_in_folder(folder, levels, output, n_jobs=-1)

    if df is not None:
        print("\nProcessing completed successfully!")
        print(f"Total results: {len(df)}")
        print(f"SSIM value range: [{df['SSIM'].min()} - {df['SSIM'].max()}]")
        print(f"PSNR value range: [{df['PSNR'].min()} - {df['PSNR'].max()}]")
    else:
        print("Processing failed!")