import os
import cv2
import numpy as np
import random
import pandas as pd
import time
from skimage.metrics import structural_similarity as ssim


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

    return np.sum(entropies)


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

    return between_class_variance


# Corrected Metrics Functions
def calculate_mse(image1, image2):
    """Calculate Mean Squared Error"""
    # Convert to float64 to avoid integer overflow
    image1_float = image1.astype(np.float64)
    image2_float = image2.astype(np.float64)
    return np.mean((image1_float - image2_float) ** 2)


def calculate_psnr(mse_value, max_pixel=255.0):
    """Calculate Peak Signal-to-Noise Ratio (corrected formula)"""
    if mse_value == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse_value)


def calculate_ssim_value(image1, image2):
    """Calculate Structural Similarity Index with correct data range"""
    return ssim(image1, image2, data_range=255)


def calculate_uniformity(image):
    """Calculate Histogram Uniformity Measure"""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist_normalized = hist / hist.sum()
    return np.sum(hist_normalized ** 2)


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


class GeneticAlgorithmSA:
    def __init__(self, params, num_thresholds, image, fitness_function="kapur"):
        self.params = params
        self.pop_size = self.params["pop_size"]
        self.selection_method = self.params["selection_method"]
        self.crossover_rate = self.params["crossover_rate"]
        self.mutation_rate = self.params["mutation_rate"]
        self.max_generations = self.params["max_generations"]
        self.no_improvement_limit = self.params["no_improvement_limit"]
        self.num_thresholds = num_thresholds
        self.image = image
        self.hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        self.population = self.initialize_population()
        self.fitness_function = fitness_function
        self.best_fitness_history = []

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

    def tournament_selection(self, size=3):
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
        mutated = individual.copy()
        if random.random() < self.mutation_rate:
            idx = random.randint(0, self.num_thresholds - 1)
            mutated[idx] = random.randint(1, 254)
            mutated = sorted(mutated)
        return mutated

    def simulated_annealing(self, solution, fitness, T=100, cooling=0.95, iterations=50):
        current = solution[:]
        current_fitness = fitness
        best = current[:]
        best_fitness = current_fitness

        while T > 1e-3:
            for _ in range(iterations):
                neighbor = current[:]
                idx = random.randint(0, self.num_thresholds - 1)
                neighbor[idx] = random.randint(1, 254)
                neighbor = sorted(neighbor)

                neighbor_fitness = self.fitness(neighbor)
                delta = neighbor_fitness - current_fitness

                if delta > 0 or random.random() < np.exp(delta / T):
                    current = neighbor
                    current_fitness = neighbor_fitness

                    if neighbor_fitness > best_fitness:
                        best = neighbor
                        best_fitness = neighbor_fitness
            T *= cooling

        return best, best_fitness

    def evolve(self):
        no_improvement_count = 0
        best_fitness = float('-inf')
        best_solution = None
        self.best_fitness_history = []

        for generation in range(self.max_generations):
            fitness_values = [self.fitness(ind) for ind in self.population]
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            current_best_solution = self.population[current_best_idx]

            # Apply simulated annealing to refine the best solution
            sa_solution, sa_fitness = self.simulated_annealing(current_best_solution, current_best_fitness)
            if sa_fitness > current_best_fitness:
                current_best_solution, current_best_fitness = sa_solution, sa_fitness

            self.best_fitness_history.append(current_best_fitness)

            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_solution = current_best_solution
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            new_population = [best_solution]

            while len(new_population) < self.pop_size:
                parent1 = self.select_parents()
                parent2 = self.select_parents()
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])

            self.population = new_population[:self.pop_size]

            if no_improvement_count >= self.no_improvement_limit:
                print(f"  Early stopping at generation {generation}")
                break

        return best_solution, best_fitness


def process_images_in_folder(folder_path, threshold_levels, param_settings, output_path):
    results = []

    # Create images folder
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_SAGA")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            filepath = os.path.join(folder_path, filename)
            print(f"Processing image: {filename}")

            try:
                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Could not load image: {filename}")
                    continue

                # Calculate histogram once for the image
                hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()

                for fitness_function_name in ["kapur", "otsu"]:
                    for num_thresholds in threshold_levels:
                        if num_thresholds not in param_settings:
                            continue

                        # Generate unique seed for this run
                        start_time = time.time()
                        run_seed = int((start_time * 10000) % 1000000) + 1  # More unique seed
                        random.seed(run_seed)
                        np.random.seed(run_seed)

                        print(
                            f"--- Experiment {filename} {fitness_function_name} {num_thresholds} Seed: {run_seed} ---")

                        params = param_settings[num_thresholds]
                        ga_sa = GeneticAlgorithmSA(
                            params=params,
                            num_thresholds=num_thresholds,
                            image=image,
                            fitness_function=fitness_function_name
                        )

                        best_solution, best_fitness = ga_sa.evolve()
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

                        # Calculate both fitness values
                        kapur_value = calculate_entropy(hist, best_solution)
                        otsu_value = calculate_otsu(hist, best_solution)

                        results.append({
                            'image_name': filename,
                            'fitness_function': fitness_function_name,
                            'thresholding_level': num_thresholds,
                            'threshold_value': best_solution,
                            'fitness_value': best_fitness,
                            'Kapur_Value': kapur_value,
                            'Otsu_Value': otsu_value,
                            'SSIM': ssim_value,
                            'MSE': mse_value,
                            'PSNR': psnr_value,
                            'Uniformity Measure': uniformity_value,
                            'Seed': run_seed,
                            'Execution Time (ms)': execution_time_ms
                        })

                        print(
                            f"    Done: Thresholds {best_solution}, Fitness {best_fitness:.4f}, Time {execution_time_ms:.2f}ms")

                        # Save the thresholded image
                        image_name_base = os.path.splitext(filename)[0]
                        output_filename = f"{image_name_base}_{fitness_function_name}_{num_thresholds}_thresholds.png"
                        output_image_path = os.path.join(images_folder, output_filename)
                        cv2.imwrite(output_image_path, thresholded_image)

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    if results:
        df = pd.DataFrame(results)
        output_file = os.path.join(output_path, "SAGA_results_corrected.xlsx")

        # Format the numeric columns for better Excel display
        df['fitness_value'] = df['fitness_value'].apply(lambda x: f"{x:.8f}")
        df['Kapur_Value'] = df['Kapur_Value'].apply(lambda x: f"{x:.8f}")
        df['Otsu_Value'] = df['Otsu_Value'].apply(lambda x: f"{x:.8f}")
        df['SSIM'] = df['SSIM'].apply(lambda x: f"{x:.6f}")
        df['MSE'] = df['MSE'].apply(lambda x: f"{x:.2f}")
        df['PSNR'] = df['PSNR'].apply(lambda x: f"{x:.2f}")
        df['Uniformity Measure'] = df['Uniformity Measure'].apply(lambda x: f"{x:.6f}")
        df['Execution Time (ms)'] = df['Execution Time (ms)'].apply(lambda x: f"{x:.2f}")

        df.to_excel(output_file, index=False)

        # Print summary of metric ranges for verification
        print("\n=== SAGA Metric Ranges Summary ===")
        print(f"SSIM range: [{min(r['SSIM'] for r in results):.6f}, {max(r['SSIM'] for r in results):.6f}]")
        print(f"MSE range: [{min(r['MSE'] for r in results):.2f}, {max(r['MSE'] for r in results):.2f}]")
        print(f"PSNR range: [{min(r['PSNR'] for r in results):.2f}, {max(r['PSNR'] for r in results):.2f}]")
        print(
            f"Uniformity range: [{min(r['Uniformity Measure'] for r in results):.6f}, {max(r['Uniformity Measure'] for r in results):.6f}]")

        print(f"Results saved to {output_file}")
        return df
    else:
        print("No results to save")
        return None


def get_initial_params():
    return {
        2: {'pop_size': 50, 'selection_method': 'tournament', 'crossover_rate': 0.8,
            'mutation_rate': 0.1, 'max_generations': 100, 'no_improvement_limit': 20},
        3: {'pop_size': 50, 'selection_method': 'tournament', 'crossover_rate': 0.8,
            'mutation_rate': 0.15, 'max_generations': 100, 'no_improvement_limit': 20},
        4: {'pop_size': 50, 'selection_method': 'tournament', 'crossover_rate': 0.75,
            'mutation_rate': 0.2, 'max_generations': 100, 'no_improvement_limit': 20},
        5: {'pop_size': 60, 'selection_method': 'tournament', 'crossover_rate': 0.7,
            'mutation_rate': 0.2, 'max_generations': 120, 'no_improvement_limit': 25},
        6: {'pop_size': 60, 'selection_method': 'tournament', 'crossover_rate': 0.65,
            'mutation_rate': 0.25, 'max_generations': 120, 'no_improvement_limit': 25},
        7: {'pop_size': 70, 'selection_method': 'tournament', 'crossover_rate': 0.6,
            'mutation_rate': 0.25, 'max_generations': 150, 'no_improvement_limit': 30},
        8: {'pop_size': 70, 'selection_method': 'tournament', 'crossover_rate': 0.55,
            'mutation_rate': 0.3, 'max_generations': 150, 'no_improvement_limit': 30},
        9: {'pop_size': 80, 'selection_method': 'tournament', 'crossover_rate': 0.5,
            'mutation_rate': 0.3, 'max_generations': 180, 'no_improvement_limit': 35},
        10: {'pop_size': 80, 'selection_method': 'tournament', 'crossover_rate': 0.45,
             'mutation_rate': 0.35, 'max_generations': 180, 'no_improvement_limit': 35}
    }


if __name__ == "__main__":
    folder_path = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images"
    output_path = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images\results"

    threshold_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    param_settings = get_initial_params()
    process_images_in_folder(folder_path, threshold_levels, param_settings, output_path)