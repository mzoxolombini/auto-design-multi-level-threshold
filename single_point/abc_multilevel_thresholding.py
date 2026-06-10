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


# 2D Rényi Entropy (order q=2) objective function
def calculate_kapur_entropy(image, thresholds):
    """Kapur's Entropy objective function for multilevel thresholding.
    Maximises the sum of class entropies across threshold segments.
    """
    thresholds = sorted(thresholds)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist / total
    boundaries = [0] + thresholds + [256]
    total_entropy = 0.0
    for i in range(len(boundaries) - 1):
        class_prob = prob[boundaries[i]:boundaries[i + 1]]
        class_sum = class_prob.sum()
        if class_sum > 1e-12:
            p_norm = class_prob / class_sum
            p_nonzero = p_norm[p_norm > 1e-12]
            total_entropy += float(-np.sum(p_nonzero * np.log(p_nonzero)))
    return total_entropy

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
    """Histogram-based uniformity measure of the segmented image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    s = hist.sum()
    if s == 0:
        return 0.0
    hist_normalised = hist / s
    return float(np.sum(hist_normalised ** 2))


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


# Utility metrics - CORRECTED VERSION
def calculate_metrics(image, thresholds):
    """Calculate all metrics with corrected formulas"""
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

    # Use corrected metric functions
    mse_value = calculate_mse(image, segmented_image)
    psnr_value = calculate_psnr(mse_value)
    ssim_value = calculate_ssim_value(image, segmented_image)
    uniformity_value = calculate_uniformity(segmented_image)

    # Validate metrics
    ssim_value, mse_value, psnr_value, uniformity_value = validate_metrics(
        ssim_value, mse_value, psnr_value, uniformity_value, "image"
    )

    return ssim_value, mse_value, psnr_value, uniformity_value, segmented_image


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
def kapur_fitness(image, thresholds):
    return calculate_kapur_entropy(image, thresholds)


def otsu_fitness(image, thresholds):
    hist, _ = np.histogram(image, bins=256, range=(0, 256))
    return calculate_otsu(hist, thresholds)


# Test the ABC algorithm

def main():
    # ── Update these two paths before running ─────────────────────────────────────
    input_folder = r"path/to/your/images"          # folder containing input images
    results_folder = r"path/to/your/results"        # folder where results will be saved
    # ──────────────────────────────────────────────────────────────────────────────
    images_folder = os.path.join(results_folder, "imagesPerThresholdLevel_ABC")

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    results = []
    threshold_levels = range(2, 11)  # Threshold levels from 2 to 10

    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)
        if not (image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'))):
            continue

        image = io.imread(image_path)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for fitness_function_name in ["kapur", "otsu"]:
            for num_thresholds in threshold_levels:
                # Generate and Set the Seed for this specific run
                start_time = time.time()
                run_seed = int((start_time * 10000) % 1000000) + 1  # More unique seed
                random.seed(run_seed)
                np.random.seed(run_seed)

                print(
                    f"--- Starting Experiment for {image_file} {fitness_function_name} {num_thresholds} with Seed: {run_seed} ---")

                if fitness_function_name == "kapur":
                    thresholds, fitness_value = abc_algorithm(
                        image, num_thresholds, kapur_fitness, colony_size=30, max_cycles=100, limit=50
                    )
                else:  # otsu
                    thresholds, fitness_value = abc_algorithm(
                        image, num_thresholds, otsu_fitness, colony_size=50, max_cycles=200, limit=75
                    )

                # Calculate execution time
                execution_time_ms = (time.time() - start_time) * 1000

                # Use corrected metrics function
                ssim_value, mse_value, psnr_value, uniformity_value, segmented_image = calculate_metrics(image,
                                                                                                         thresholds)

                # Convert thresholds to regular Python integers to avoid np.int64 in Excel
                thresholds_list = [int(t) for t in sorted(thresholds)]

                results.append({
                    'image_name': image_file,
                    'fitness_function': fitness_function_name,
                    'thresholding_level': num_thresholds,
                    'threshold_value': thresholds_list,
                    'fitness_value': fitness_value,
                    'SSIM': ssim_value,
                    'MSE': mse_value,
                    'PSNR': psnr_value,
                    'Uniformity': uniformity_value,
                    'Seed': run_seed,
                    'Execution Time (ms)': execution_time_ms
                })

                print(f"Processed {image_file} with {num_thresholds} thresholds using {fitness_function_name} fitness: "
                      f"Best Thresholds {thresholds_list}, Fitness {fitness_value:.6f}, SSIM {ssim_value:.6f}, "
                      f"MSE {mse_value:.6f}, PSNR {psnr_value:.6f}, Uniformity {uniformity_value:.6f}, Seed {run_seed}, "
                      f"Time {execution_time_ms:.2f}ms")

                # Save segmented image
                image_name_base = os.path.splitext(image_file)[0]
                output_filename = f"{image_name_base}_{fitness_function_name}_{num_thresholds}_thresholds.png"
                output_path = os.path.join(images_folder, output_filename)
                cv2.imwrite(output_path, segmented_image)

    # Save results to an Excel file
    if results:
        results_df = pd.DataFrame(results)
        excel_path = os.path.join(results_folder, "ABC_results_corrected.xlsx")

        # Convert threshold values to regular integers to avoid np.int64 issues
        results_df['threshold_value'] = results_df['threshold_value'].apply(lambda x: [int(val) for val in x])

        # Format numeric columns (optional, for better Excel presentation)
        results_df['fitness_value'] = results_df['fitness_value'].apply(lambda x: f"{x:.8f}")
        results_df['SSIM'] = results_df['SSIM'].apply(lambda x: f"{x:.6f}")
        results_df['MSE'] = results_df['MSE'].apply(lambda x: f"{x:.2f}")
        results_df['PSNR'] = results_df['PSNR'].apply(lambda x: f"{x:.2f}")
        results_df['Uniformity'] = results_df['Uniformity'].apply(lambda x: f"{x:.6f}")
        results_df['Execution Time (ms)'] = results_df['Execution Time (ms)'].apply(lambda x: f"{x:.2f}")

        results_df.to_excel(excel_path, index=False)

        # Print summary of metric ranges for verification
        print("\n=== Metric Ranges Summary ===")
        print(f"SSIM range: [{min(r['SSIM'] for r in results):.6f}, {max(r['SSIM'] for r in results):.6f}]")
        print(f"MSE range: [{min(r['MSE'] for r in results):.2f}, {max(r['MSE'] for r in results):.2f}]")
        print(f"PSNR range: [{min(r['PSNR'] for r in results):.2f}, {max(r['PSNR'] for r in results):.2f}]")
        print(
            f"Uniformity range: [{min(r['Uniformity'] for r in results):.6f}, {max(r['Uniformity'] for r in results):.6f}]")

        print(f"\nResults saved to: {excel_path}")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()