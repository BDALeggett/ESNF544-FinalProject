import os
import cv2
import numpy as np
import logging
from glob import glob
from matplotlib import pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global flags for testing and visualization
TEST_FIRST_ONLY = True    # Process only the first image per grade folder for quick testing
SHOW_VISUALIZATION = True  # Set to False to disable plotting in production

#############################
# Helper functions for border & inner region extraction
#############################

def get_yellow_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask

def get_edge_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("No contours found in the yellow mask.")
        return None, None
    largest = max(contours, key=cv2.contourArea)
    bbox = cv2.boundingRect(largest)
    return largest, bbox

def measure_thickness(mask, direction):
    h, w = mask.shape
    if direction == 'top':
        for i in range(h):
            if np.mean(mask[i, :]) < 255 * 0.9:
                return i
        return h
    elif direction == 'bottom':
        for i in range(h-1, -1, -1):
            if np.mean(mask[i, :]) < 255 * 0.9:
                return h - i - 1
        return h
    elif direction == 'left':
        for j in range(w):
            if np.mean(mask[:, j]) < 255 * 0.9:
                return j
        return w
    elif direction == 'right':
        for j in range(w-1, -1, -1):
            if np.mean(mask[:, j]) < 255 * 0.9:
                return w - j - 1
        return w

def get_inner_region(image, mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        logger.warning("No yellow border detected; cannot compute inner region.")
        return None, (0, 0, 0, 0)
    x, y, w, h = cv2.boundingRect(coords)
    top = measure_thickness(mask, 'top')
    bottom = measure_thickness(mask, 'bottom')
    left = measure_thickness(mask, 'left')
    right = measure_thickness(mask, 'right')
    if (y + top >= y + h - bottom) or (x + left >= x + w - right):
        logger.warning("Measured border thickness too large; cannot extract inner region.")
        return None, (top, bottom, left, right)
    inner = image[y+top:y+h-bottom, x+left:x+w-right]
    return inner, (top, bottom, left, right)

#############################
# Metrics Computation 
#############################

def compute_border_continuity(mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 0.0
    x, y, w, h = cv2.boundingRect(coords)
    area_box = w * h
    area_mask = np.count_nonzero(mask)
    ratio = area_mask / area_box if area_box > 0 else 0
    score = min(10, ratio * 10)
    return score

def compute_corner_quality(mask, bbox, patch_size=20):
    x, y, w, h = bbox
    patches = [
        mask[y:y+patch_size, x:x+patch_size],
        mask[y:y+patch_size, x+w-patch_size:x+w],
        mask[y+h-patch_size:y+h, x:x+patch_size],
        mask[y+h-patch_size:y+h, x+w-patch_size:x+w]
    ]
    fractions = []
    for patch in patches:
        total = patch.size
        white = np.count_nonzero(patch)
        fraction = white / total if total > 0 else 0
        fractions.append(fraction)
    avg_fraction = np.mean(fractions)
    score = min(10, avg_fraction * 10)
    return score

def compute_centering(mask, threshold_diff=5):
    top = measure_thickness(mask, 'top')
    bottom = measure_thickness(mask, 'bottom')
    left = measure_thickness(mask, 'left')
    right = measure_thickness(mask, 'right')
    diff_vert = abs(top - bottom)
    diff_horiz = abs(left - right)
    avg_diff = (diff_vert + diff_horiz) / 2.0
    score = max(0, 10 - (avg_diff / threshold_diff * 10))
    return score

def compute_inner_cleanliness(inner_region, threshold=50):
    if inner_region is None:
        return 0.0
    gray = cv2.cvtColor(inner_region, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    score = max(0, 10 - (std / threshold * 10))
    return score

def compute_grading_index(image, expected_grade=None, alpha=0.5):
    """
    Compute a grading index based on average saturation.
    Maps average saturation from [50, 200] to a 0–10 score.
    If expected_grade is provided, blends the computed score with the expected grade.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_sat = np.mean(hsv[:, :, 1])
    computed_score = (avg_sat - 50) / (200 - 50) * 10
    computed_score = np.clip(computed_score, 0, 10)
    if expected_grade is not None:
        expected_score = float(expected_grade)
        final_score = alpha * computed_score + (1 - alpha) * expected_score
        return final_score
    return computed_score

def compute_edge_density(inner_region):
    """
    Compute the edge density in the inner region using Canny edge detection.
    Returns a score on a 0–10 scale.
    """
    if inner_region is None:
        return None
    edges_inner = cv2.Canny(inner_region, 50, 150)
    density = np.count_nonzero(edges_inner) / edges_inner.size
    # Scale density to [0,10]; adjust the scaling factor as needed.
    score = min(10, density * 100)
    return score

def compute_saturation_variance(image):
    """
    Compute the variance of the saturation channel from the HSV representation of the image.
    We map the variance to a 0–10 score inversely: more variance might indicate inconsistent printing.
    Here, we assume a variance range up to about 1000.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    var_sat = np.var(hsv[:, :, 1])
    # Inverse mapping: lower variance yields higher score.
    score = 10 - min(10, var_sat / 1000 * 10)
    return score

#############################
# Main Feature Extraction Function 
#############################

def extract_all_features(image, expected_grade=None):
    """
    Extract features from the processed image (with yellow border).
    Computes seven features:
      1. Border Continuity
      2. Corner Quality
      3. Centering
      4. Inner Cleanliness
      5. Grading Index
      6. Edge Density (in inner region)
      7. Saturation Variance (from the full image)
    
    If any metric is missing, it is replaced by the average of available metrics.
    Returns:
      - features: list of 7 feature values
      - intermediate outputs: edges, bbox, inner_region, thicknesses, yellow_mask
    """
    yellow_mask = get_yellow_mask(image)
    edges = get_edge_map(image)
    _, bbox = find_largest_contour(yellow_mask)
    
    continuity = compute_border_continuity(yellow_mask) if bbox is not None else None
    corner_quality = compute_corner_quality(yellow_mask, bbox, patch_size=20) if bbox is not None else None
    centering = compute_centering(yellow_mask, threshold_diff=5) if bbox is not None else None
    inner_region, thicknesses = get_inner_region(image, yellow_mask)
    cleanliness = compute_inner_cleanliness(inner_region, threshold=50) if inner_region is not None else None
    grading_index = compute_grading_index(image, expected_grade=expected_grade)
    edge_density = compute_edge_density(inner_region)
    sat_variance = compute_saturation_variance(image)
    
    raw_features = [continuity, corner_quality, centering, cleanliness, grading_index, edge_density, sat_variance]
    # Build a validity mask: True if feature is not None, else False.
    valid_mask = [f is not None for f in raw_features]
    valid_values = [f for f in raw_features if f is not None]
    avg_value = sum(valid_values) / len(valid_values) if valid_values else 0.0
    features = [float(f) if f is not None else avg_value for f in raw_features]
    
    # Ensure feature vector has exactly 7 elements and valid_mask too.
    if len(features) != 7:
        if len(features) > 7:
            features = features[:7]
        else:
            features = features + [avg_value] * (7 - len(features))
    if len(valid_mask) != 7:
        valid_mask = valid_mask + [False] * (7 - len(valid_mask))
    
    return features, valid_mask, edges, bbox, inner_region, thicknesses, yellow_mask

#############################
# Data Cleaning Function 
#############################

def clean_feature_matrix(X, valid_masks, threshold=0.3):
    X = np.array(X)
    valid_masks = np.array(valid_masks)
    n_samples, n_features = X.shape
    keep_indices = []
    for j in range(n_features):
        valid_prop = np.mean(valid_masks[:, j])
        if valid_prop >= threshold:
            keep_indices.append(j)
        else:
            logger.info(f"Feature column {j} removed: valid proportion = {valid_prop:.2f}")
    if not keep_indices:
        logger.warning("No features met the validity threshold; returning original X.")
        return X, list(range(n_features))
    cleaned_X = X[:, keep_indices]
    return cleaned_X, keep_indices

#############################
# Visualization & Testing Functions 
#############################

def visualize_features(image, features, edges, bbox, inner_region, thicknesses, yellow_mask, grade_label=None, show=True):
    if not show:
        return
    overlay = image.copy()
    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), 2)
    summary_text = f"Expected Grade: {grade_label}\n" if grade_label is not None else ""
    summary_text += (
        f"Border Continuity: {features[0]:.1f}/10\n"
        f"Corner Quality: {features[1]:.1f}/10\n"
        f"Centering: {features[2]:.1f}/10\n"
        f"Inner Cleanliness: {features[3]:.1f}/10\n"
        f"Grading Index: {features[4]:.1f}/10\n"
        f"Edge Density: {features[5]:.1f}/10\n"
        f"Saturation Variance: {features[6]:.1f}/10\n"
    )
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Image with Detected Border")
    plt.axis("off")
    
    plt.subplot(2, 3, 2)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edge Map")
    plt.axis("off")
    
    plt.subplot(2, 3, 3)
    plt.imshow(yellow_mask, cmap="gray")
    plt.title("Yellow Mask")
    plt.axis("off")
    
    plt.subplot(2, 3, 4)
    if inner_region is not None:
        plt.imshow(cv2.cvtColor(inner_region, cv2.COLOR_BGR2RGB))
        plt.title("Extracted Inner Region")
    else:
        plt.text(0.5, 0.5, "No inner region", ha="center", fontsize=14)
    plt.axis("off")
    
    plt.subplot(2, 3, 5)
    plt.text(0.1, 0.5, summary_text, fontsize=12)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def test_feature_extraction(input_dir="data\\processed", test_first_only=TEST_FIRST_ONLY, visualize=SHOW_VISUALIZATION):
    abs_input_dir = os.path.abspath(input_dir)
    grade_folders = [os.path.join(abs_input_dir, f) for f in os.listdir(abs_input_dir)
                     if os.path.isdir(os.path.join(abs_input_dir, f))]
    logger.info(f"Found grade folders: {grade_folders}")
    for folder in grade_folders:
        img_files = sorted(glob(os.path.join(folder, "*.jpg")))
        if test_first_only and img_files:
            img_files = img_files[:1]
        grade_label = os.path.basename(folder).split("_")[-1]
        logger.info(f"Processing {len(img_files)} images in folder: {folder} (Grade: {grade_label})")
        for img_path in img_files:
            image = cv2.imread(img_path)
            if image is None:
                logger.warning(f"Could not load image: {img_path}")
                continue
            features, valid_mask, edges, bbox, inner_region, thicknesses, yellow_mask = extract_all_features(image)
            if features is None:
                logger.warning(f"Feature extraction failed for {img_path}")
                continue
            logger.info(f"{img_path} -> Features: {features}, Valid Mask: {valid_mask}")
            visualize_features(image, features, edges, bbox, inner_region, thicknesses, yellow_mask, grade_label, show=visualize)

def save_feature_dataset(input_dir="data\\processed", output_file="data\\features\\features.npy.npz"):
    """
    For a 4-class setup:
      - 0 => All grades <= 7
      - 1 => Grade 8
      - 2 => Grade 9
      - 3 => Grade 10
    Our feature vector now has 7 dimensions.
    After creating X and valid_masks, we clean the feature matrix by removing any column
    for which less than 30% of samples have a valid (original) value.
    """
    X_list, valid_mask_list, y_list = [], [], []
    abs_input_dir = os.path.abspath(input_dir)
    grade_folders = [os.path.join(abs_input_dir, f) for f in os.listdir(abs_input_dir)
                     if os.path.isdir(os.path.join(abs_input_dir, f))]
    for folder in grade_folders:
        grade_str = os.path.basename(folder)
        try:
            grade_num = int(grade_str.split("_")[-1])
        except Exception as e:
            logger.warning(f"Could not parse grade from {folder}: {e}")
            continue
        
        # 4-class logic:
        if grade_num <= 7:
            label = 0
        elif grade_num == 8:
            label = 1
        elif grade_num == 9:
            label = 2
        elif grade_num == 10:
            label = 3
        else:
            logger.warning(f"Grade {grade_num} not in [<=7,8,9,10]; skipping.")
            continue
        
        img_files = sorted(glob(os.path.join(folder, "*.jpg")))
        logger.info(f"Extracting from {len(img_files)} images in {folder} => label={label}")
        for img_path in img_files:
            image = cv2.imread(img_path)
            if image is None:
                continue
            features, valid_mask, *_ = extract_all_features(image)
            if features is None:
                continue
            features = [float(f) for f in features]
            if len(features) != 7:
                avg_val = sum(features)/len(features) if features else 0.0
                while len(features) < 7:
                    features.append(avg_val)
                features = features[:7]
            if not isinstance(valid_mask, list) or len(valid_mask) != 7:
                valid_mask = [False] * 7
            X_list.append(features)
            valid_mask_list.append(valid_mask)
            y_list.append(label)
    
    X = np.array(X_list, dtype=float)
    valid_masks = np.array([vm if len(vm)==7 else [False]*7 for vm in valid_mask_list], dtype=bool)
    y = np.array(y_list, dtype=int)
    
    cleaned_X, keep_indices = clean_feature_matrix(X, valid_masks, threshold=0.3)
    logger.info(f"After cleaning, kept feature columns: {keep_indices}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(output_file, X=cleaned_X, y=y)
    logger.info(f"Saved dataset to {output_file}: X.shape={cleaned_X.shape}, y.shape={y.shape}")

if __name__ == "__main__":
    # For testing and visualization:
    # test_feature_extraction("ENSF 544\\Final-Project\\data\\processed", test_first_only=TEST_FIRST_ONLY, visualize=SHOW_VISUALIZATION)
    
    # To save the dataset with cleaning:
    save_feature_dataset(input_dir="data\\processed", output_file="data\\features\\features.npy.npz")
