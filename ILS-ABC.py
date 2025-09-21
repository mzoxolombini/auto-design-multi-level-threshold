import os
import cv2
import numpy as np
import random
import pandas as pd
import time
from skimage.metrics import structural_similarity as ssim

# Set initial seed for reproducibility
SEED = 102
random.seed(SEED)
np.random.seed(SEED)


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


def calculate_mse(image1, image2):
    return np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)


def calculate_psnr(mse_value, max_pixel=255.0):
    if mse_value == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))


def calculate_uniformity(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist_normalized = hist / hist.sum()
    return np.sum(hist_normalized ** 2)


class ArtificialBeeColonyILS:
    def __init__(self, params, num_thresholds, image, fitness_function="kapur"):
        self.params = params
        self.colony_size = self.params["colony_size"]
        self.max_iterations = self.params["max_iterations"]
        self.limit = self.params["limit"]
        self.ils_iterations = self.params["ils_iterations"]
        self.perturbation_strength = self.params["perturbation_strength"]
        self.num_thresholds = num_thresholds
        self.image = image
        self.hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
        self.fitness_function = fitness_function
        self.best_fitness_history = []

        # Initialize population
        self.food_sources = self.initialize_food_sources()
        self.fitness_values = [self.fitness(fs) for fs in self.food_sources]
        self.trial_counters = [0] * self.colony_size
        self.best_solution = None
        self.best_fitness = float('-inf')

    def initialize_food_sources(self):
        return [sorted(random.sample(range(1, 255), self.num_thresholds)) for _ in range(self.colony_size)]

    def fitness(self, individual):
        if self.fitness_function == "kapur":
            return calculate_entropy(self.hist, individual)
        elif self.fitness_function == "otsu":
            return calculate_otsu(self.hist, individual)
        else:
            raise ValueError("Unknown fitness function")

    def employed_bee_phase(self):
        for i in range(self.colony_size):
            # Generate a new candidate solution
            k = random.randint(0, self.colony_size - 1)
            while k == i:
                k = random.randint(0, self.colony_size - 1)

            j = random.randint(0, self.num_thresholds - 1)
            phi = random.uniform(-1, 1)

            new_solution = self.food_sources[i].copy()
            new_solution[j] = int(self.food_sources[i][j] + phi *
                                  (self.food_sources[i][j] - self.food_sources[k][j]))

            # Ensure boundaries
            new_solution[j] = max(1, min(254, new_solution[j]))
            new_solution = sorted(new_solution)

            new_fitness = self.fitness(new_solution)

            # Greedy selection
            if new_fitness > self.fitness_values[i]:
                self.food_sources[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1

    def onlooker_bee_phase(self):
        # Convert fitness values to probabilities with better handling
        fitness_array = np.array(self.fitness_values)

        # Handle negative fitness values (if any)
        if np.any(fitness_array < 0):
            fitness_array = fitness_array - np.min(fitness_array) + 1e-12

        # Handle case where all fitness values are zero or very small
        if np.sum(fitness_array) <= 1e-12:
            probabilities = np.ones(self.colony_size) / self.colony_size
        else:
            probabilities = fitness_array / np.sum(fitness_array)

        # Ensure probabilities sum to exactly 1 (within floating-point tolerance)
        probabilities = probabilities / np.sum(probabilities)

        for _ in range(self.colony_size):
            # Select a food source based on probability
            i = np.random.choice(range(self.colony_size), p=probabilities)

            # Generate a new candidate solution
            k = random.randint(0, self.colony_size - 1)
            while k == i:
                k = random.randint(0, self.colony_size - 1)

            j = random.randint(0, self.num_thresholds - 1)
            phi = random.uniform(-1, 1)

            new_solution = self.food_sources[i].copy()
            new_solution[j] = int(self.food_sources[i][j] + phi *
                                  (self.food_sources[i][j] - self.food_sources[k][j]))

            # Ensure boundaries
            new_solution[j] = max(1, min(254, new_solution[j]))
            new_solution = sorted(new_solution)

            new_fitness = self.fitness(new_solution)

            # Greedy selection
            if new_fitness > self.fitness_values[i]:
                self.food_sources[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1

    def scout_bee_phase(self):
        for i in range(self.colony_size):
            if self.trial_counters[i] >= self.limit:
                self.food_sources[i] = sorted(random.sample(range(1, 255), self.num_thresholds))
                self.fitness_values[i] = self.fitness(self.food_sources[i])
                self.trial_counters[i] = 0

    def local_search(self, solution):
        """Iterative Local Search: Perform local optimization around a solution"""
        current_solution = solution.copy()
        current_fitness = self.fitness(current_solution)

        for _ in range(self.ils_iterations):
            # Create a neighbor by perturbing one threshold
            neighbor = current_solution.copy()
            idx = random.randint(0, self.num_thresholds - 1)

            # Perturb the selected threshold
            perturbation = random.randint(-self.perturbation_strength, self.perturbation_strength)
            neighbor[idx] = max(1, min(254, neighbor[idx] + perturbation))
            neighbor = sorted(neighbor)

            neighbor_fitness = self.fitness(neighbor)

            # Accept if better
            if neighbor_fitness > current_fitness:
                current_solution = neighbor
                current_fitness = neighbor_fitness

        return current_solution, current_fitness

    def perturb_solution(self, solution):
        """Perturb the solution to escape local optima"""
        perturbed = solution.copy()

        # Perturb multiple thresholds
        num_to_perturb = max(1, self.num_thresholds // 2)
        indices = random.sample(range(self.num_thresholds), num_to_perturb)

        for idx in indices:
            perturbation = random.randint(-self.perturbation_strength * 2, self.perturbation_strength * 2)
            perturbed[idx] = max(1, min(254, perturbed[idx] + perturbation))

        return sorted(perturbed)

    def optimize(self):
        no_improvement_count = 0
        self.best_fitness_history = []

        # Initial ABC phase
        for iteration in range(self.max_iterations):
            # Employed bee phase
            self.employed_bee_phase()

            # Onlooker bee phase
            self.onlooker_bee_phase()

            # Update best solution
            current_best_idx = np.argmax(self.fitness_values)
            current_best_fitness = self.fitness_values[current_best_idx]
            current_best_solution = self.food_sources[current_best_idx]

            self.best_fitness_history.append(current_best_fitness)

            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_solution = current_best_solution
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Scout bee phase
            self.scout_bee_phase()

            if no_improvement_count >= self.params.get("no_improvement_limit", 20):
                print(f"  ABC phase completed at iteration {iteration}")
                break

        # ILS phase
        print("  Starting ILS phase...")
        best_ils_solution = self.best_solution
        best_ils_fitness = self.best_fitness

        for ils_iteration in range(self.params.get("ils_cycles", 5)):
            # Apply local search to the best solution
            local_optimized, local_fitness = self.local_search(best_ils_solution)

            if local_fitness > best_ils_fitness:
                best_ils_solution = local_optimized
                best_ils_fitness = local_fitness
                print(f"    ILS improved fitness to: {best_ils_fitness:.4f}")

            # Perturb the solution for the next ILS cycle
            if ils_iteration < self.params.get("ils_cycles", 5) - 1:
                best_ils_solution = self.perturb_solution(best_ils_solution)

        # Final local search on the best found solution
        final_solution, final_fitness = self.local_search(best_ils_solution)

        if final_fitness > self.best_fitness:
            self.best_solution = final_solution
            self.best_fitness = final_fitness

        return self.best_solution, self.best_fitness


def process_images_in_folder(folder_path, threshold_levels, param_settings, output_path):
    results = []

    # Create images folder
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_ILSABC")
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
                        run_seed = int((start_time * 10000) % 1000) + 1
                        random.seed(run_seed)
                        np.random.seed(run_seed)

                        print(
                            f"--- Starting ABC-ILS Experiment for {filename} {fitness_function_name} {num_thresholds} with Seed: {run_seed} ---")

                        try:
                            params = param_settings[num_thresholds]
                            abc_ils = ArtificialBeeColonyILS(
                                params=params,
                                num_thresholds=num_thresholds,
                                image=image,
                                fitness_function=fitness_function_name
                            )

                            best_solution, best_fitness = abc_ils.optimize()
                            execution_time_ms = (time.time() - start_time) * 1000

                            thresholded_image = apply_thresholds(image, best_solution)

                            image_normalized = image / 255.0
                            thresholded_normalized = thresholded_image / 255.0

                            mse_value = calculate_mse(image, thresholded_image)
                            psnr_value = calculate_psnr(mse_value)
                            ssim_value = ssim(image_normalized, thresholded_normalized, data_range=1.0)
                            uniformity_value = calculate_uniformity(thresholded_image)

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
                                f"    Completed: Thresholds {best_solution}, Fitness {best_fitness:.4f}, Time {execution_time_ms:.2f}ms")

                            # Save the thresholded image
                            image_name_base = os.path.splitext(filename)[0]
                            output_filename = f"{image_name_base}_{fitness_function_name}_{num_thresholds}_thresholds.png"
                            output_image_path = os.path.join(images_folder, output_filename)
                            cv2.imwrite(output_image_path, thresholded_image)

                        except Exception as e:
                            print(f"Error in ABC-ILS optimization for {filename}: {str(e)}")
                            continue

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    if results:
        df = pd.DataFrame(results)
        output_file = os.path.join(output_path, "ILSABC_results.xlsx")

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
        return df
    else:
        print("No results to save")
        return None


def get_initial_params():
    return {
        2: {'colony_size': 50, 'max_iterations': 50, 'limit': 15, 'ils_iterations': 10,
            'perturbation_strength': 5, 'ils_cycles': 3, 'no_improvement_limit': 15},
        3: {'colony_size': 50, 'max_iterations': 50, 'limit': 15, 'ils_iterations': 10,
            'perturbation_strength': 5, 'ils_cycles': 3, 'no_improvement_limit': 15},
        4: {'colony_size': 50, 'max_iterations': 60, 'limit': 15, 'ils_iterations': 12,
            'perturbation_strength': 4, 'ils_cycles': 4, 'no_improvement_limit': 15},
        5: {'colony_size': 60, 'max_iterations': 60, 'limit': 20, 'ils_iterations': 12,
            'perturbation_strength': 4, 'ils_cycles': 4, 'no_improvement_limit': 20},
        6: {'colony_size': 60, 'max_iterations': 70, 'limit': 20, 'ils_iterations': 15,
            'perturbation_strength': 3, 'ils_cycles': 4, 'no_improvement_limit': 20},
        7: {'colony_size': 70, 'max_iterations': 70, 'limit': 25, 'ils_iterations': 15,
            'perturbation_strength': 3, 'ils_cycles': 5, 'no_improvement_limit': 25},
        8: {'colony_size': 70, 'max_iterations': 80, 'limit': 25, 'ils_iterations': 18,
            'perturbation_strength': 3, 'ils_cycles': 5, 'no_improvement_limit': 25},
        9: {'colony_size': 80, 'max_iterations': 80, 'limit': 30, 'ils_iterations': 18,
            'perturbation_strength': 2, 'ils_cycles': 5, 'no_improvement_limit': 30},
        10: {'colony_size': 80, 'max_iterations': 90, 'limit': 30, 'ils_iterations': 20,
             'perturbation_strength': 2, 'ils_cycles': 6, 'no_improvement_limit': 30}
    }


if __name__ == "__main__":
    folder_path = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images"
    output_path = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images\results"

    threshold_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    param_settings = get_initial_params()
    process_images_in_folder(folder_path, threshold_levels, param_settings, output_path)