import os
import cv2
import numpy as np
import random
import pandas as pd
import time
import csv
from skimage.metrics import structural_similarity as ssim


# Kapur's Entropy Calculation (no normalization)
def calculate_entropy(hist, thresholds):
    thresholds = sorted(thresholds)
    total = np.sum(hist)
    if total == 0:
        return 0

    probs = hist / total
    entropies = []
    prev_threshold = 0

    for threshold in thresholds + [len(hist)]:
        class_probs = probs[prev_threshold:threshold]
        class_sum = np.sum(class_probs)
        if class_sum > 0:
            class_entropy = -np.sum(
                (class_probs / class_sum) * np.log(class_probs / class_sum + 1e-12)
            )
            entropies.append(class_entropy)
        else:
            entropies.append(0)
        prev_threshold = threshold

    # return sum of entropies (not normalized)
    return np.sum(entropies)


# Otsu's Between-Class Variance Calculation (no normalization)
def calculate_otsu(hist, thresholds):
    thresholds = sorted(thresholds)
    total = np.sum(hist)
    if total == 0:
        return 0

    probs = hist / total
    thresholds = [0] + thresholds + [len(hist)]
    global_mean = np.sum(probs * np.arange(len(hist)))
    between_class_variance = 0

    for i in range(1, len(thresholds)):
        class_probs = probs[thresholds[i - 1]:thresholds[i]]
        class_sum = np.sum(class_probs)
        if class_sum > 0:
            class_mean = np.sum(class_probs * np.arange(thresholds[i - 1], thresholds[i])) / class_sum
            between_class_variance += class_sum * (class_mean - global_mean) ** 2

    # return raw between-class variance (not normalized)
    return between_class_variance


# Genetic Algorithm for Multi-Thresholding
class GeneticAlgorithm:
    def __init__(self, params, num_thresholds, image, fitness_function="kapur", seed=None):
        self.pop_size = params["pop_size"]
        self.selection_method = params["selection_method"]
        self.crossover_rate = params["crossover_rate"]
        self.mutation_rate = params["mutation_rate"]
        self.max_generations = params["max_generations"]
        self.num_thresholds = num_thresholds
        self.image = image
        self.hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        self.fitness_function = fitness_function

        # Set seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.population = self.initialize_population()

    def initialize_population(self):
        return [sorted(random.sample(range(1, 255), self.num_thresholds)) for _ in range(self.pop_size)]

    def fitness(self, individual):
        if self.fitness_function == "kapur":
            return calculate_entropy(self.hist, individual)
        elif self.fitness_function == "otsu":
            return calculate_otsu(self.hist, individual)
        else:
            raise ValueError("Unknown fitness function")

    def select_parents(self):
        if self.selection_method == "tournament":
            return self.tournament_selection()
        else:
            raise ValueError("Unknown selection method")

    def tournament_selection(self, size=10):
        selected = random.sample(self.population, size)
        selected.sort(key=self.fitness, reverse=True)
        return selected[0]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            idx = random.randint(1, self.num_thresholds - 1)
            child1 = sorted(parent1[:idx] + parent2[idx:])
            child2 = sorted(parent2[:idx] + parent1[idx:])
            return child1, child2
        return parent1, parent2

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.num_thresholds - 1)
            individual[idx] = random.randint(1, 254)
            individual = sorted(individual)
        return individual

    def evolve(self):
        for _ in range(self.max_generations):
            new_population = []
            for _ in range(self.pop_size // 2):
                parent1 = self.select_parents()
                parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            self.population = sorted(new_population, key=self.fitness, reverse=True)[:self.pop_size]
        return self.population[0], self.fitness(self.population[0])


# Metrics
def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)


def calculate_psnr(mse_value, max_pixel=255.0):
    if mse_value == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))


def calculate_uniformity(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist_normalized = hist / hist.sum()
    return np.sum(hist_normalized ** 2)


# Apply thresholds using midpoints
def apply_thresholds(image, thresholds):
    thresholds = sorted(thresholds)
    segmented = np.zeros_like(image)
    bounds = [0] + thresholds + [255]

    for i in range(len(bounds) - 1):
        mask = (image >= bounds[i]) & (image < bounds[i + 1])
        midpoint = (bounds[i] + bounds[i + 1]) // 2
        segmented[mask] = midpoint

    return segmented


# Initialize CSV file with headers
def initialize_csv(output_path):
    csv_file = os.path.join(output_path, "GA_experiment_results.csv")
    header = ['Seed', 'Runtime', 'Image', 'Level', 'Method', 'Fitness', 'Thresholds', 'SSIM', 'MSE', 'PSNR',
              'Uniformity']

    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    return csv_file


# Process all images in a folder
def process_images_in_folder(folder_path, threshold_levels, param_settings, output_path):
    results = []
    csv_file = initialize_csv(output_path)

    # create folder for thresholded images
    images_output_path = os.path.join(output_path, "imagesPerThresholdLevel")
    os.makedirs(images_output_path, exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    print(f"Found {len(image_files)} images to process: {image_files}")

    for filename in image_files:
        filepath = os.path.join(folder_path, filename)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not load image {filename}. Skipping...")
            continue

        for fitness_function in ["kapur", "otsu"]:
            for num_thresholds in threshold_levels:
                # 1. Generate and Set the Seed for this specific run
                run_seed = int(time.time() * 1000) % 1000000  # More unique seed
                random.seed(run_seed)
                np.random.seed(run_seed)

                print(
                    f"--- Starting Experiment for {filename} {fitness_function} {num_thresholds} with Seed: {run_seed} ---")

                # Track runtime
                start_time = time.time()

                # 2. Run GA
                params = param_settings[num_thresholds]
                ga = GeneticAlgorithm(params=params, num_thresholds=num_thresholds,
                                      image=image, fitness_function=fitness_function,
                                      seed=run_seed)
                best_solution, best_fitness = ga.evolve()

                # Runtime in milliseconds
                runtime = int((time.time() - start_time) * 1000)

                thresholded_image = apply_thresholds(image, best_solution)

                mse_value = calculate_mse(image, thresholded_image)
                psnr_value = calculate_psnr(mse_value)
                ssim_value = ssim(image, thresholded_image)
                uniformity_value = calculate_uniformity(thresholded_image)

                # Save thresholded image
                image_save_name = f"{os.path.splitext(filename)[0]}_level{num_thresholds}_{fitness_function}.png"
                save_path = os.path.join(images_output_path, image_save_name)
                cv2.imwrite(save_path, thresholded_image)

                # Store results
                results.append({
                    'Seed': run_seed,
                    'Runtime': runtime,
                    'Image': filename,
                    'Level': num_thresholds,
                    'Method': fitness_function,
                    'Fitness': best_fitness,
                    'Thresholds': best_solution,
                    'SSIM': ssim_value,
                    'MSE': mse_value,
                    'PSNR': psnr_value,
                    'Uniformity': uniformity_value
                })

                # Save immediately to CSV
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        run_seed, runtime, filename, num_thresholds, fitness_function,
                        best_fitness, str(best_solution), ssim_value,
                        mse_value, psnr_value, uniformity_value
                    ])

                print(
                    f"--- Results saved. Best Fitness for {num_thresholds}-level {fitness_function}: {best_fitness:.6f} ---")
                print(f"{filename} {fitness_function} {num_thresholds} -> "
                      f"Thresholds {best_solution}, Fitness {best_fitness:.6f}, "
                      f"SSIM {ssim_value:.4f}, MSE {mse_value:.2f}, "
                      f"PSNR {psnr_value:.2f}, Uniformity {uniformity_value:.4f}, "
                      f"Runtime {runtime} ms")

    # Save to Excel
    if results:  # Only save if there are results
        df = pd.DataFrame(results)
        output_file = os.path.join(output_path, "GA_results.xlsx")
        df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results to save.")


# Parameter settings
def get_initial_params():
    return {
        2: {'pop_size': 200, 'selection_method': 'tournament', 'crossover_rate': 0.85, 'mutation_rate': 0.25,
            'max_generations': 200},
        3: {'pop_size': 200, 'selection_method': 'tournament', 'crossover_rate': 0.80, 'mutation_rate': 0.30,
            'max_generations': 200},
        4: {'pop_size': 200, 'selection_method': 'tournament', 'crossover_rate': 0.70, 'mutation_rate': 0.25,
            'max_generations': 200},
        5: {'pop_size': 200, 'selection_method': 'tournament', 'crossover_rate': 0.65, 'mutation_rate': 0.20,
            'max_generations': 200},
        6: {'pop_size': 200, 'selection_method': 'tournament', 'crossover_rate': 0.60, 'mutation_rate': 0.15,
            'max_generations': 200},
        7: {'pop_size': 200, 'selection_method': 'tournament', 'crossover_rate': 0.55, 'mutation_rate': 0.15,
            'max_generations': 200},
        8: {'pop_size': 200, 'selection_method': 'tournament', 'crossover_rate': 0.50, 'mutation_rate': 0.15,
            'max_generations': 200},
        9: {'pop_size': 200, 'selection_method': 'tournament', 'crossover_rate': 0.45, 'mutation_rate': 0.15,
            'max_generations': 200},
        10: {'pop_size': 200, 'selection_method': 'tournament', 'crossover_rate': 0.40, 'mutation_rate': 0.15,
             'max_generations': 200}
    }


if __name__ == "__main__":
    folder_path = r"C:\Users\mzoxo\OneDrive\Documents\test_images"
    output_path = r"C:\Users\mzoxo\OneDrive\Documents\test_images\results"
    threshold_levels = [2, 3]
    param_settings = get_initial_params()

    os.makedirs(output_path, exist_ok=True)

    # Process images only once
    process_images_in_folder(folder_path, threshold_levels, param_settings, output_path)
    print("Processing completed.")