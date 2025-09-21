import os
import numpy as np
import cv2
import pandas as pd
from skimage import io
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
import random
import time


# Fitness functions

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


# Utility metrics

def calculate_metrics(image, thresholds):
    # Apply thresholds using midpoints
    segmented_image = np.zeros_like(image)
    thresholds = np.sort(thresholds)
    bins = np.concatenate(([0], thresholds, [256]))

    # Calculate midpoints for each segment
    midpoints = []
    for i in range(len(bins) - 1):
        midpoint = (bins[i] + bins[i + 1]) // 2
        midpoints.append(midpoint)

    # Apply midpoints to segmented image
    for i in range(len(bins) - 1):
        mask = (image >= bins[i]) & (image < bins[i + 1])
        segmented_image[mask] = midpoints[i]

    mse_value = np.mean((image - segmented_image) ** 2)
    psnr_value = 10 * np.log10(255 ** 2 / mse_value) if mse_value != 0 else np.inf
    ssim_value = ssim(image, segmented_image, data_range=segmented_image.max() - segmented_image.min())

    histogram, _ = np.histogram(segmented_image, bins=256, range=(0, 256))
    uniformity_value = np.sum((histogram / histogram.sum()) ** 2)

    return ssim_value, mse_value, psnr_value, uniformity_value


# ABC Algorithm

def abc_algorithm(image, num_thresholds, fitness_function, colony_size=50, max_cycles=200, limit=75):
    num_food_sources = colony_size // 2
    dim = num_thresholds

    # Initialize food sources
    food_sources = [np.random.randint(1, 255, dim) for _ in range(num_food_sources)]
    fitness = [fitness_function(image, fs) for fs in food_sources]
    trial = np.zeros(num_food_sources)

    for cycle in range(max_cycles):
        # Employed bee phase
        for i in range(num_food_sources):
            k = i
            while k == i:
                k = np.random.randint(0, num_food_sources)

            phi = np.random.uniform(-1, 1, dim)
            new_solution = np.clip(food_sources[i] + phi * (food_sources[i] - food_sources[k]), 1, 255).astype(int)

            new_fitness = fitness_function(image, new_solution)
            if new_fitness > fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trial[i] = 0
            else:
                trial[i] += 1

        # Onlooker bee phase
        prob = fitness / np.sum(fitness)
        for _ in range(num_food_sources):
            i = np.random.choice(range(num_food_sources), p=prob)

            k = i
            while k == i:
                k = np.random.randint(0, num_food_sources)

            phi = np.random.uniform(-1, 1, dim)
            new_solution = np.clip(food_sources[i] + phi * (food_sources[i] - food_sources[k]), 1, 255).astype(int)

            new_fitness = fitness_function(image, new_solution)
            if new_fitness > fitness[i]:
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trial[i] = 0
            else:
                trial[i] += 1

        # Scout bee phase
        for i in range(num_food_sources):
            if trial[i] > limit:
                food_sources[i] = np.random.randint(1, 255, dim)
                fitness[i] = fitness_function(image, food_sources[i])
                trial[i] = 0

    # Best solution
    best_idx = np.argmax(fitness)
    best_thresholds = food_sources[best_idx]
    best_fitness = fitness[best_idx]

    return best_thresholds, best_fitness


# Define fitness functions
def kapur_entropy(image, thresholds):
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    return calculate_entropy(hist, thresholds)


def otsu_fitness(image, thresholds):
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    return calculate_otsu(hist, thresholds)


# Test the ABC algorithm

def main():
    input_folder = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images"
    results_folder = r"C:\Users\mzoxo\OneDrive\Documents\standard_test_images\results"

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    results = []
    threshold_levels = range(2, 11)  # Threshold levels from 2 to 6

    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)
        if not (image_file.endswith('.png') or image_file.endswith('.jpg') or image_file.endswith('.tif')):
            continue

        image = io.imread(image_path)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for fitness_function_name in ["kapur", "otsu"]:
            for num_thresholds in threshold_levels:
                # Generate and Set the Seed for this specific run
                start_time = time.time()
                run_seed = int((start_time * 10000) % 1000) + 1
                random.seed(run_seed)
                np.random.seed(run_seed)

                print(
                    f"--- Starting Experiment for {image_file} {fitness_function_name} {num_thresholds} with Seed: {run_seed} ---")

                if fitness_function_name == "kapur":
                    thresholds, fitness_value = abc_algorithm(
                        image, num_thresholds, kapur_entropy, colony_size=30, max_cycles=100, limit=50
                    )
                else:  # otsu
                    thresholds, fitness_value = abc_algorithm(
                        image, num_thresholds, otsu_fitness, colony_size=50, max_cycles=200, limit=75
                    )

                ssim_value, mse_value, psnr_value, uniformity_value = calculate_metrics(image, thresholds)

                # Convert thresholds to regular Python integers to avoid np.int64 in Excel
                thresholds_list = [int(t) for t in sorted(thresholds)]

                results.append({
                    'image_name': image_file,
                    'fitness_function': fitness_function_name.capitalize(),
                    'thresholding_level': num_thresholds,
                    'threshold_value': thresholds_list,
                    'fitness_value': fitness_value,
                    'SSIM': ssim_value,
                    'MSE': mse_value,
                    'PSNR': psnr_value,
                    'Uniformity Measure': uniformity_value,
                    'Random Seed': run_seed
                })

                print(f"Processed {image_file} with {num_thresholds} thresholds using {fitness_function_name} fitness: "
                      f"Best Thresholds {thresholds_list}, Fitness {fitness_value:.6f}, SSIM {ssim_value:.6f}, "
                      f"MSE {mse_value:.6f}, PSNR {psnr_value:.6f}, Uniformity {uniformity_value:.6f}, Seed {run_seed}")

    # Save results to an Excel file
    results_df = pd.DataFrame(results)
    excel_path = os.path.join(results_folder, "base_ABC_results.xlsx")

    # Convert threshold values to regular integers to avoid np.int64 issues
    results_df['threshold_value'] = results_df['threshold_value'].apply(lambda x: [int(val) for val in x])
    results_df['fitness_value'] = results_df['fitness_value'].apply(lambda x: f"{x:,.8f}")
    results_df['SSIM'] = results_df['SSIM'].apply(lambda x: f"{x:,.7f}")
    results_df['MSE'] = results_df['MSE'].apply(lambda x: f"{x:,.7f}")
    results_df['PSNR'] = results_df['PSNR'].apply(lambda x: f"{x:,.7f}")
    results_df['Uniformity Measure'] = results_df['Uniformity Measure'].apply(lambda x: f"{x:,.7f}")

    results_df.to_excel(excel_path, index=False)


if __name__ == "__main__":
    main()