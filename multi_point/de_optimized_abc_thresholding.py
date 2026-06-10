import os
import cv2
import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed
from scipy.ndimage import correlate
from skimage.metrics import structural_similarity as ssim_skimage
from skimage.metrics import mean_squared_error
import math
import time


# ===============================
# Corrected Utility Functions
# ===============================

def calculate_histogram(image):
    """Calculate normalized histogram of an image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total == 0:
        return np.zeros_like(hist)
    return hist / total


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
    """Otsu's between-class variance for multilevel thresholding (unnormalised)."""
    thresholds = sorted(thresholds)
    total = np.sum(hist)
    if total == 0:
        return 0.0
    probs = hist / total
    thresholds_ext = [0] + thresholds + [len(hist)]
    global_mean = np.sum(probs * np.arange(len(hist)))
    between_class_variance = 0.0
    for i in range(1, len(thresholds_ext)):
        class_probs = probs[thresholds_ext[i - 1]:thresholds_ext[i]]
        class_sum = np.sum(class_probs)
        if class_sum > 0:
            class_mean = (np.sum(class_probs * np.arange(thresholds_ext[i - 1], thresholds_ext[i]))
                          / class_sum)
            between_class_variance += class_sum * (class_mean - global_mean) ** 2
    return float(between_class_variance)


def calculate_ssim(original, segmented):
    """Calculate SSIM using scikit-image."""
    min_shape = (min(original.shape[0], segmented.shape[0]),
                 min(original.shape[1], segmented.shape[1]))

    if original.shape != min_shape:
        original = cv2.resize(original, min_shape[::-1])
    if segmented.shape != min_shape:
        segmented = cv2.resize(segmented, min_shape[::-1])

    data_range = 255 if original.max() > 1.0 else 1.0
    return ssim_skimage(original, segmented, data_range=data_range)


def calculate_mse(original, segmented):
    """Calculate Mean Squared Error."""
    min_shape = (min(original.shape[0], segmented.shape[0]),
                 min(original.shape[1], segmented.shape[1]))

    if original.shape != min_shape:
        original = cv2.resize(original, min_shape[::-1])
    if segmented.shape != min_shape:
        segmented = cv2.resize(segmented, min_shape[::-1])

    return mean_squared_error(original, segmented)


def calculate_psnr(original, segmented):
    """Calculate PSNR."""
    mse = calculate_mse(original, segmented)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 10 * math.log10((max_pixel ** 2) / mse)


def calculate_uniformity(image):
    """Histogram-based uniformity measure of the segmented image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    s = hist.sum()
    if s == 0:
        return 0.0
    hist_normalised = hist / s
    return float(np.sum(hist_normalised ** 2))


def apply_thresholds(image, thresholds):
    """Apply multiple thresholds using per-segment midpoint values."""
    thresholds = sorted(thresholds)
    segmented = np.zeros_like(image, dtype=np.uint8)
    bounds = [0] + thresholds + [256]
    for i in range(len(bounds) - 1):
        mask = (image >= bounds[i]) & (image < bounds[i + 1])
        midpoint = (bounds[i] + bounds[i + 1]) // 2
        segmented[mask] = midpoint
    return segmented


# ===============================
# Artificial Bee Colony Algorithm
# ===============================

class ArtificialBeeColony:
    def __init__(self, config, num_thresholds, hist, fitness_func, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # ABC parameters from literature (typical values)
        self.colony_size = config['colony_size']  # SN (number of food sources)
        self.max_iterations = config['max_iterations']  # Maximum cycles
        self.limit = config['limit']  # Abandonment limit
        self.num_thresholds = num_thresholds
        self.hist = hist
        self.fitness_func = fitness_func

        self.fitness_cache = {}

        # Initialize food sources (solutions)
        self.food_sources = []
        for _ in range(self.colony_size):
            individual = np.sort(np.random.choice(range(1, 255), num_thresholds, replace=False))
            self.food_sources.append(individual)

        self.fitness_values = [self.evaluate(ind) for ind in self.food_sources]
        self.trial_counters = [0] * self.colony_size  # Trial counters for each food source
        self.best_solution = self.food_sources[np.argmax(self.fitness_values)].copy()
        self.best_fitness = max(self.fitness_values)

    def evaluate(self, individual):
        """Evaluate fitness of an individual."""
        key = tuple(individual)
        if key not in self.fitness_cache:
            try:
                self.fitness_cache[key] = self.fitness_func(self.hist, individual)
            except Exception as e:
                print(f"Fitness evaluation error: {e}")
                self.fitness_cache[key] = float('-inf')
        return self.fitness_cache[key]

    def employed_bee_phase(self):
        """Employed bee phase: generate new solutions and apply greedy selection."""
        for i in range(self.colony_size):
            # Generate a new solution by modifying the current one
            new_solution = self.generate_new_solution(i)
            new_fitness = self.evaluate(new_solution)

            # Greedy selection
            if new_fitness > self.fitness_values[i]:
                self.food_sources[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trial_counters[i] = 0

                # Update best solution
                if new_fitness > self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness
            else:
                self.trial_counters[i] += 1

    def onlooker_bee_phase(self):
        """Onlooker bee phase: select solutions based on fitness and generate new ones."""
        # Calculate probabilities based on fitness (roulette wheel selection)
        fitness_array = np.array(self.fitness_values)
        # Handle negative fitness values by shifting to positive range
        min_fitness = np.min(fitness_array)
        if min_fitness < 0:
            fitness_array = fitness_array - min_fitness + 1e-10

        probabilities = fitness_array / np.sum(fitness_array)

        for i in range(self.colony_size):
            # Select a food source based on probability
            selected_idx = np.random.choice(self.colony_size, p=probabilities)

            # Generate a new solution
            new_solution = self.generate_new_solution(selected_idx)
            new_fitness = self.evaluate(new_solution)

            # Greedy selection
            if new_fitness > self.fitness_values[selected_idx]:
                self.food_sources[selected_idx] = new_solution
                self.fitness_values[selected_idx] = new_fitness
                self.trial_counters[selected_idx] = 0

                # Update best solution
                if new_fitness > self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness
            else:
                self.trial_counters[selected_idx] += 1

    def scout_bee_phase(self):
        """Scout bee phase: abandon exhausted solutions and create new ones."""
        for i in range(self.colony_size):
            if self.trial_counters[i] >= self.limit:
                # Abandon this food source and create a new one
                new_solution = np.sort(np.random.choice(range(1, 255), self.num_thresholds, replace=False))
                new_fitness = self.evaluate(new_solution)

                self.food_sources[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trial_counters[i] = 0

                # Update best solution if needed
                if new_fitness > self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness

    def generate_new_solution(self, current_idx):
        """Generate a new solution by modifying a current solution."""
        current_solution = self.food_sources[current_idx].copy()

        # Select a random dimension to modify
        dim = np.random.randint(0, self.num_thresholds)

        # Select a random partner different from current_idx
        partner_idx = np.random.choice([j for j in range(self.colony_size) if j != current_idx])
        partner_solution = self.food_sources[partner_idx]

        # Generate new value using ABC mutation formula
        phi = np.random.uniform(-1, 1)
        new_val = current_solution[dim] + phi * (current_solution[dim] - partner_solution[dim])

        # Ensure the value is within bounds and integer
        new_val = int(np.clip(new_val, 1, 254))

        # Ensure uniqueness of thresholds
        while new_val in current_solution:
            new_val = np.random.randint(1, 255)

        current_solution[dim] = new_val
        return np.sort(current_solution)

    def evolve(self):
        """Run ABC algorithm evolution."""
        print(f"ABC Initial best fitness: {self.best_fitness:.6f}")

        for iteration in range(self.max_iterations):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()

            if iteration % 10 == 0:
                print(f"ABC Iteration {iteration}: Best fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness


# ===============================
# Image Processing
# ===============================

def process_single(file_name, folder_path, num_thresholds, func_name, seed=None, images_folder=None):
    """Process a single image with given parameters."""
    try:
        run_seed = (seed + hash(file_name) + num_thresholds + hash(func_name)) % 1000000
        np.random.seed(run_seed)
        random.seed(run_seed)

        start_time = time.time()

        image_path = os.path.join(folder_path, file_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[ERROR] Could not load image: {file_name}")
            return None

        print(f"[START] Processing {file_name} | {func_name} | thresholds={num_thresholds}")

        hist = calculate_histogram(image)
        if func_name == "kapur":
            _image = image
            def fitness_function(hist, thresholds, _img=_image):
                return calculate_kapur_entropy(_img, thresholds)
        else:
            fitness_function = calculate_otsu

        # ABC parameters from literature (typical values)
        config = {
            'colony_size': 20,  # Number of food sources
            'max_iterations': 100,  # Maximum cycles
            'limit': 10  # Abandonment limit
        }

        abc = ArtificialBeeColony(config, num_thresholds, hist, fitness_function, run_seed)
        best_thresholds, best_score = abc.evolve()

        execution_time_ms = (time.time() - start_time) * 1000

        segmented_image = apply_thresholds(image, best_thresholds)

        ssim_value = calculate_ssim(image, segmented_image)
        mse_value = calculate_mse(image, segmented_image)
        psnr_value = calculate_psnr(image, segmented_image)
        uniformity = calculate_uniformity(segmented_image)

        # Format threshold values exactly as requested
        threshold_value_str = "[" + ", ".join(map(str, best_thresholds)) + "]"

        if images_folder:
            image_name_base = os.path.splitext(file_name)[0]
            output_filename = f"{image_name_base}_{func_name}_{num_thresholds}_thresholds.png"
            output_image_path = os.path.join(images_folder, output_filename)
            cv2.imwrite(output_image_path, segmented_image)

        print(f"[DONE] {file_name} | {func_name} | thresholds={num_thresholds} | "
              f"fitness={best_score:.4f} | time={execution_time_ms:.2f}ms")

        # Return results in exact format as requested
        return {
            "image_name": file_name,
            "fitness_function": func_name,  # Keep as lowercase to match example
            "thresholding_level": num_thresholds,
            "threshold_value": threshold_value_str,
            "fitness_value": best_score,
            "SSIM": ssim_value,
            "MSE": mse_value,
            "PSNR": psnr_value,
            "Uniformity": uniformity,
            "Seed": run_seed,
            "Execution Time (ms)": execution_time_ms
        }

    except Exception as e:
        print(f"[ERROR] Processing {file_name} | {func_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "image_name": file_name,
            "fitness_function": func_name,
            "error": str(e)
        }


def safe_save_excel(df, filepath):
    """Safely save DataFrame to Excel with error handling."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Close any open file handles first
            import gc
            gc.collect()

            # Try to save with a temporary filename first
            temp_filepath = filepath.replace('.xlsx', f'_temp_{attempt}.xlsx')
            df.to_excel(temp_filepath, index=False)

            # If successful, rename to target filename
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass  # Ignore removal errors

            os.rename(temp_filepath, filepath)
            print(f"Successfully saved to: {filepath}")
            return True

        except PermissionError:
            print(f"Permission denied (attempt {attempt + 1}/{max_retries}). Retrying...")
            time.sleep(1)
        except Exception as e:
            print(f"Error saving to Excel (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(1)

    # If all retries fail, try CSV
    try:
        csv_path = filepath.replace('.xlsx', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved results to CSV instead: {csv_path}")
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False


def process_images_in_folder(folder_path, threshold_levels, output_file, n_jobs=1, seed=None):
    """Main function to process all images in a folder."""
    output_path = os.path.dirname(output_file)

    # images_folder is now inside the results folder
    images_folder = os.path.join(output_path, "imagesPerThresholdLevel_DEABC")

    # Create directories if they don't exist
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    if not files:
        print("No image files found in the specified folder.")
        return pd.DataFrame()

    print(f"Found {len(files)} images to process.")

    if seed is None:
        seed = int(time.time() * 1000) % 1000000
    np.random.seed(seed)
    random.seed(seed)
    print(f"Using random seed: {seed}")

    # Process images in the correct order: all threshold levels for each image-function combination
    all_results = []

    for file_name in files:
        for func_name in ["kapur", "otsu"]:
            # Process all threshold levels for this image-function combination
            for num_thresholds in threshold_levels:
                result = process_single(file_name, folder_path, num_thresholds, func_name, seed, images_folder)
                if result and 'error' not in result:
                    all_results.append(result)

    if all_results:
        df = pd.DataFrame(all_results)

        # Exact column order and names as requested
        column_order = [
            'image_name',
            'fitness_function',
            'thresholding_level',
            'threshold_value',
            'fitness_value',
            'SSIM',
            'MSE',
            'PSNR',
            'Uniformity',
            'Seed',
            'Execution Time (ms)'
        ]

        # Ensure all columns exist
        for col in column_order:
            if col not in df.columns:
                df[col] = None

        df = df[column_order]

        # Format numeric columns to match the example precision
        if 'fitness_value' in df.columns:
            df['fitness_value'] = df['fitness_value'].apply(lambda x: f"{x:.8f}" if pd.notna(x) else x)

        if 'SSIM' in df.columns:
            df['SSIM'] = df['SSIM'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else x)

        if 'MSE' in df.columns:
            df['MSE'] = df['MSE'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)

        if 'PSNR' in df.columns:
            df['PSNR'] = df['PSNR'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)

        if 'Uniformity' in df.columns:
            df['Uniformity'] = df['Uniformity'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else x)

        if 'Execution Time (ms)' in df.columns:
            df['Execution Time (ms)'] = df['Execution Time (ms)'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)

        # Sort the results to group by image and fitness function
        df = df.sort_values(['image_name', 'fitness_function', 'thresholding_level']).reset_index(drop=True)

        # Save the results
        if safe_save_excel(df, output_file):
            print(f"Results saved to {output_file}")
        else:
            # Last resort: print results to console
            print("Could not save to file. Printing results:")
            print(df.to_string(index=False))

        print(f"Successfully processed {len(all_results)} configurations")

        # Display summary statistics
        print("\nSummary Statistics:")
        print(df[['fitness_value', 'SSIM', 'PSNR', 'MSE', 'Uniformity']].describe())

    else:
        print("No valid results to save")
        df = pd.DataFrame()

    return df


# ===============================
# Main Execution
# ===============================

if __name__ == "__main__":
    # Configuration with your specific paths
    # ── Update these two paths before running ─────────────────────────────────────
    folder = r"path/to/your/images"          # folder containing input images
    output_dir = r"path/to/your/results"          # folder where results will be saved
    # ──────────────────────────────────────────────────────────────────────────────
    levels = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Create results directory
    os.makedirs(output_dir, exist_ok=True)

    # Use a unique filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"DEABC_results_{timestamp}.xlsx")

    print(f"Input folder: {folder}")
    print(f"Output file: {output_file}")
    print(f"Threshold levels: {levels}")

    try:
        df = process_images_in_folder(folder, levels, output_file, n_jobs=1, seed=42)

        if not df.empty:
            print("\nProcessing completed successfully!")
            print(f"Total results: {len(df)}")

            # Display final file locations
            images_folder = os.path.join(output_dir, "imagesPerThresholdLevel_DEABC")
            print(f"Results saved to: {output_file}")
            print(f"Segmented images saved to: {images_folder}")

            # Print first few rows to verify format
            print("\nFirst few rows of results:")
            print(df.head(10).to_string(index=False))
        else:
            print("Processing completed with errors!")
    except Exception as e:
        print(f"Critical error during processing: {e}")
        import traceback

        traceback.print_exc()