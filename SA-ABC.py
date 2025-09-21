import os
import cv2
import numpy as np
import random
import pandas as pd
import time
from skimage.metrics import structural_similarity as ssim
import math


# ---------------------------
# CORRECTED Thresholding utilities
# ---------------------------

def apply_thresholds(image, thresholds):
    thresholds = sorted(thresholds)
    output = np.zeros_like(image)
    prev = 0
    for t in thresholds:
        output[(image >= prev) & (image < t)] = t
        prev = t
    output[image >= prev] = 255
    return output


def kapur_entropy(image, thresholds):
    """CORRECTED Kapur's entropy calculation"""
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    total_pixels = np.sum(hist)
    prob = hist.astype(float) / total_pixels

    # Add boundaries
    thresholds = sorted(thresholds)
    class_boundaries = [0] + thresholds + [256]

    total_entropy = 0.0

    for i in range(len(class_boundaries) - 1):
        start, end = class_boundaries[i], class_boundaries[i + 1]

        # Get probabilities for this class
        class_probs = prob[start:end]
        class_probs = class_probs[class_probs > 0]  # Remove zeros

        if len(class_probs) > 0:
            # Calculate class probability sum
            class_prob_sum = np.sum(class_probs)

            # Calculate normalized entropy for this class
            if class_prob_sum > 0:
                normalized_probs = class_probs / class_prob_sum
                class_entropy = -np.sum(normalized_probs * np.log(normalized_probs))
                total_entropy += class_entropy

    return total_entropy


def otsu_variance(image, thresholds):
    """CORRECTED Otsu's between-class variance"""
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    total_pixels = np.sum(hist)
    prob = hist.astype(float) / total_pixels

    # Calculate global mean
    global_mean = np.sum(np.arange(256) * prob)

    # Add boundaries
    thresholds = sorted(thresholds)
    class_boundaries = [0] + thresholds + [256]

    between_class_variance = 0.0

    for i in range(len(class_boundaries) - 1):
        start, end = class_boundaries[i], class_boundaries[i + 1]

        # Class probability (weight)
        class_weight = np.sum(prob[start:end])

        if class_weight > 0:
            # Class mean
            class_mean = np.sum(np.arange(start, end) * prob[start:end]) / class_weight

            # Add to between-class variance
            between_class_variance += class_weight * (class_mean - global_mean) ** 2

    return between_class_variance


def calculate_mse(original, segmented):
    """Calculate Mean Squared Error"""
    return np.mean((original.astype(float) - segmented.astype(float)) ** 2)


def calculate_psnr(mse):
    """Calculate Peak Signal-to-Noise Ratio"""
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0) - 10 * math.log10(mse)


def calculate_uniformity_measure(segmented):
    """Calculate Uniformity Measure (higher is better)"""
    hist, _ = np.histogram(segmented.flatten(), bins=256, range=(0, 256))
    prob = hist.astype(float) / np.sum(hist)
    return np.sum(prob ** 2)  # Sum of squared probabilities


# ---------------------------
# Validation functions
# ---------------------------

def validate_functions():
    """Test the corrected functions with known values"""
    print("=== VALIDATING CORRECTED FUNCTIONS ===")

    # Create a simple test image
    test_image = np.random.randint(0, 256, (100, 100))

    # Test with single threshold
    thresholds = [128]

    kapur_val = kapur_entropy(test_image, thresholds)
    otsu_val = otsu_variance(test_image, thresholds)

    print(f"Single threshold [128]:")
    print(f"  Kapur entropy: {kapur_val:.6f}")
    print(f"  Otsu variance: {otsu_val:.6f}")

    # Test with multiple thresholds
    thresholds_multi = [64, 128, 192]

    kapur_multi = kapur_entropy(test_image, thresholds_multi)
    otsu_multi = otsu_variance(test_image, thresholds_multi)

    print(f"Multi thresholds [64, 128, 192]:")
    print(f"  Kapur entropy: {kapur_multi:.6f}")
    print(f"  Otsu variance: {otsu_multi:.6f}")

    # The multi-threshold values should be higher than single threshold
    print(f"Kapur improvement: {kapur_multi - kapur_val:.6f}")
    print(f"Otsu improvement: {otsu_multi - otsu_val:.6f}")

    return True


# ---------------------------
# Artificial Bee Colony (ABC) - CORRECTED
# ---------------------------

class ABC:
    def __init__(self, image, obj_func, num_thresholds,
                 pop_size=20, max_iter=50, limit=5):
        self.image = image
        self.obj_func = obj_func
        self.num_thresholds = num_thresholds
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.limit = limit
        self.dim = num_thresholds
        self.bounds = (1, 254)  # Avoid 0 and 255

    def initialize_population(self):
        """Initialize with proper spacing between thresholds"""
        population = []
        for _ in range(self.pop_size):
            while True:
                thresholds = sorted(np.random.randint(self.bounds[0], self.bounds[1], self.dim))
                # Ensure minimum spacing of 10 between thresholds
                if all(thresholds[i + 1] - thresholds[i] >= 10 for i in range(len(thresholds) - 1)):
                    population.append(thresholds)
                    break
        return population

    def evaluate(self, solution):
        return self.obj_func(self.image, solution)

    def evolve(self):
        population = self.initialize_population()
        fitness = [self.evaluate(sol) for sol in population]
        trial = [0] * self.pop_size
        best_solution = population[np.argmax(fitness)]
        best_fitness = max(fitness)

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

                # Ensure sorted and spaced
                new_sol = sorted(set(new_sol))
                if len(new_sol) < self.dim:
                    # Regenerate missing thresholds with spacing
                    available = [x for x in range(self.bounds[0], self.bounds[1])
                                 if all(abs(x - t) >= 10 for t in new_sol)]
                    if len(available) >= (self.dim - len(new_sol)):
                        new_sol.extend(np.random.choice(available, self.dim - len(new_sol), replace=False))
                        new_sol = sorted(new_sol)
                    else:
                        continue

                new_fit = self.evaluate(new_sol)

                if new_fit > fitness[i]:
                    population[i] = new_sol
                    fitness[i] = new_fit
                    trial[i] = 0
                else:
                    trial[i] += 1

            # Onlooker bees phase
            fitness_arr = np.array(fitness)
            if np.sum(fitness_arr) == 0:
                probs = np.ones(self.pop_size) / self.pop_size
            else:
                probs = fitness_arr / np.sum(fitness_arr)
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

                new_sol = sorted(set(new_sol))
                if len(new_sol) < self.dim:
                    available = [x for x in range(self.bounds[0], self.bounds[1])
                                 if all(abs(x - t) >= 10 for t in new_sol)]
                    if len(available) >= (self.dim - len(new_sol)):
                        new_sol.extend(np.random.choice(available, self.dim - len(new_sol), replace=False))
                        new_sol = sorted(new_sol)
                    else:
                        continue

                new_fit = self.evaluate(new_sol)

                if new_fit > fitness[i]:
                    population[i] = new_sol
                    fitness[i] = new_fit
                    trial[i] = 0
                else:
                    trial[i] += 1

            # Scout bees phase
            for i in range(self.pop_size):
                if trial[i] > self.limit:
                    while True:
                        new_sol = sorted(np.random.randint(self.bounds[0], self.bounds[1], self.dim))
                        if all(new_sol[i + 1] - new_sol[i] >= 10 for i in range(len(new_sol) - 1)):
                            break
                    population[i] = new_sol
                    fitness[i] = self.evaluate(new_sol)
                    trial[i] = 0

            # Update best solution
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_solution = population[current_best_idx]
                best_fitness = fitness[current_best_idx]

        return best_solution, best_fitness


# ---------------------------
# Simulated Annealing for ABC Parameter Optimization
# ---------------------------

class SA_ABC_Optimizer:
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
        return {
            'pop_size': random.randint(*self.param_bounds['pop_size']),
            'max_iter': random.randint(*self.param_bounds['max_iter']),
            'limit': random.randint(*self.param_bounds['limit'])
        }

    def generate_neighbor(self, current_solution):
        neighbor = current_solution.copy()
        param = random.choice(list(self.param_bounds.keys()))
        step = random.choice([-2, -1, 1, 2])
        new_val = neighbor[param] + step
        new_val = max(self.param_bounds[param][0], min(self.param_bounds[param][1], new_val))
        neighbor[param] = new_val
        return neighbor

    def evaluate_abc_params(self, params):
        abc = ABC(self.image, self.obj_func, self.num_thresholds,
                  pop_size=params['pop_size'],
                  max_iter=params['max_iter'],
                  limit=params['limit'])
        _, fitness = abc.evolve()
        return fitness

    def optimize(self):
        current_solution = self.generate_initial_solution()
        current_fitness = self.evaluate_abc_params(current_solution)
        best_solution = current_solution.copy()
        best_fitness = current_fitness

        temperature = self.initial_temp

        print(f"Initial parameters: {current_solution}, Fitness: {current_fitness:.6f}")

        for iteration in range(self.max_iter):
            neighbor = self.generate_neighbor(current_solution)
            neighbor_fitness = self.evaluate_abc_params(neighbor)

            delta = neighbor_fitness - current_fitness

            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness

                if neighbor_fitness > best_fitness:
                    best_solution = neighbor.copy()
                    best_fitness = neighbor_fitness

            temperature *= self.cooling_rate

        print(f"Optimized parameters: {best_solution}, Fitness: {best_fitness:.6f}")
        return best_solution, best_fitness


# ---------------------------
# Processing images with CORRECTED implementations
# ---------------------------

def process_images_in_folder(folder_path, threshold_levels, output_path):
    # Validate functions first
    validate_functions()

    # Create images folder
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_SAABC")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            # Calculate histogram once for the image
            hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
            total_pixels = np.sum(hist)
            prob = hist.astype(float) / total_pixels

            for method, obj_func in [("kapur", kapur_entropy), ("otsu", otsu_variance)]:
                for m in threshold_levels:
                    start_time = time.time()
                    run_seed = int((start_time * 10000) % 1000) + 1
                    random.seed(run_seed)
                    np.random.seed(run_seed)

                    print(f"\n=== SA-ABC {filename} {method} {m} Seed: {run_seed} ===")

                    # Optimize ABC parameters with SA
                    sa_optimizer = SA_ABC_Optimizer(
                        image, obj_func, m,
                        initial_temp=1000, cooling_rate=0.93, max_iter=15
                    )

                    optimized_params, _ = sa_optimizer.optimize()

                    # Run ABC with optimized parameters
                    abc = ABC(image, obj_func, m,
                              pop_size=optimized_params['pop_size'],
                              max_iter=optimized_params['max_iter'],
                              limit=optimized_params['limit'])

                    best_sol, best_fit = abc.evolve()
                    execution_time_ms = (time.time() - start_time) * 1000

                    # Create segmented image and calculate metrics
                    segmented = apply_thresholds(image, best_sol)
                    mse_value = calculate_mse(image, segmented)
                    psnr_value = calculate_psnr(mse_value)
                    ssim_value = ssim(image, segmented, data_range=255)
                    uniformity = calculate_uniformity_measure(segmented)

                    # Calculate both fitness values
                    kapur_value = kapur_entropy(image, best_sol)
                    otsu_value = otsu_variance(image, best_sol)

                    threshold_str = f"[{', '.join(map(str, best_sol))}]"

                    results.append({
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
                    })

                    print(
                        f"Completed: Fitness: {best_fit:.6f}, Thresholds: {best_sol}, Time: {execution_time_ms:.2f}ms")

                    # Save the thresholded image
                    image_name_base = os.path.splitext(filename)[0]
                    output_filename = f"{image_name_base}_{method}_{m}_thresholds.png"
                    output_image_path = os.path.join(images_folder, output_filename)
                    cv2.imwrite(output_image_path, segmented)

    # Save results
    df = pd.DataFrame(results)
    excel_file_path = os.path.join(output_path, "SA-ABC_Results.xlsx")

    # Format the numeric columns for better Excel display
    df['fitness_value'] = df['fitness_value'].apply(lambda x: f"{x:,.8f}")
    df['Kapur_Value'] = df['Kapur_Value'].apply(lambda x: f"{x:,.8f}")
    df['Otsu_Value'] = df['Otsu_Value'].apply(lambda x: f"{x:,.8f}")
    df['SSIM'] = df['SSIM'].apply(lambda x: f"{x:,.7f}")
    df['MSE'] = df['MSE'].apply(lambda x: f"{x:,.7f}")
    df['PSNR'] = df['PSNR'].apply(lambda x: f"{x:,.7f}")
    df['Uniformity Measure'] = df['Uniformity Measure'].apply(lambda x: f"{x:,.7f}")
    df['Execution Time (ms)'] = df['Execution Time (ms)'].apply(lambda x: f"{x:,.2f}")

    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)
        worksheet = writer.sheets['Results']
        for column in worksheet.columns:
            max_length = max(len(str(cell.value)) for cell in column)
            worksheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)

    print(f"\nResults saved to {excel_file_path}")


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    folder_path = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images"
    output_path = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images\results"
    os.makedirs(output_path, exist_ok=True)
    threshold_levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    process_images_in_folder(folder_path, threshold_levels, output_path)