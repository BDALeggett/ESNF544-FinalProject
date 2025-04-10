# Import necessary libraries for image processing, data handling, and visualization
import os
import cv2
import numpy as np
import logging
from glob import glob
from matplotlib import pyplot as plt

# Configure logging to display informative messages
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global configuration flags
TEST_FIRST_ONLY = True  # Only test the first image in each folder
SHOW_VISUALIZATION = True  # Show visualization of feature extraction

def get_yellow_mask(image):
    """Extract a mask of yellow regions from the image using HSV color space.
    
    Args:
        image (numpy.ndarray): Input BGR image
    
    Returns:
        numpy.ndarray: Binary mask where yellow regions are white
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    return mask

def get_edge_map(image):
    """Detect edges in the image using Canny edge detection.
    
    Args:
        image (numpy.ndarray): Input BGR image
    
    Returns:
        numpy.ndarray: Binary edge map
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def find_largest_contour(mask):
    """Find the largest contour in the binary mask.
    
    Args:
        mask (numpy.ndarray): Binary mask
    
    Returns:
        tuple: (largest_contour, bounding_box) or (None, None) if no contours found
    """
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("No contours found in the yellow mask.")
        return None, None
    largest = max(contours, key=cv2.contourArea)
    bbox = cv2.boundingRect(largest)
    return largest, bbox

def measure_thickness(mask, direction):
    """Measure the thickness of the border in a specific direction.
    
    Args:
        mask (numpy.ndarray): Binary mask
        direction (str): One of 'top', 'bottom', 'left', 'right'
    
    Returns:
        int: Thickness of the border in pixels
    """
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
    """Extract the inner region of the card by removing the border.
    
    Args:
        image (numpy.ndarray): Input BGR image
        mask (numpy.ndarray): Binary mask of the card
    
    Returns:
        tuple: (inner_region, thicknesses) where thicknesses is (top, bottom, left, right)
    """
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

def compute_border_continuity(mask):
    """Compute a score for the continuity of the card's border.
    
    Args:
        mask (numpy.ndarray): Binary mask of the card
    
    Returns:
        float: Score between 0 and 10
    """
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
    """Compute a score for the quality of the card's corners.
    
    Args:
        mask (numpy.ndarray): Binary mask of the card
        bbox (tuple): Bounding box (x, y, w, h)
        patch_size (int): Size of corner patches to analyze
    
    Returns:
        float: Score between 0 and 10
    """
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
    """Compute a score for the card's centering.
    
    Args:
        mask (numpy.ndarray): Binary mask of the card
        threshold_diff (float): Maximum allowed difference in border thickness
    
    Returns:
        float: Score between 0 and 10
    """
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
    """Compute a score for the cleanliness of the card's inner region.
    
    Args:
        inner_region (numpy.ndarray): Inner region of the card
        threshold (float): Threshold for standard deviation
    
    Returns:
        float: Score between 0 and 10
    """
    if inner_region is None:
        return 0.0
    gray = cv2.cvtColor(inner_region, cv2.COLOR_BGR2GRAY)
    std = np.std(gray)
    score = max(0, 10 - (std / threshold * 10))
    return score

def compute_grading_index(image, expected_grade=None, alpha=0.5):
    """Compute a grading index based on image saturation and expected grade.
    
    Args:
        image (numpy.ndarray): Input BGR image
        expected_grade (float, optional): Expected grade of the card
        alpha (float): Weight for computed score vs expected grade
    
    Returns:
        float: Score between 0 and 10
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
    """Compute a score for the edge density in the inner region.
    
    Args:
        inner_region (numpy.ndarray): Inner region of the card
    
    Returns:
        float: Score between 0 and 10, or None if inner_region is None
    """
    if inner_region is None:
        return None
    edges_inner = cv2.Canny(inner_region, 50, 150)
    density = np.count_nonzero(edges_inner) / edges_inner.size
    score = min(10, density * 100)
    return score

def compute_saturation_variance(image):
    """Compute a score based on the variance of saturation in the image.
    
    Args:
        image (numpy.ndarray): Input BGR image
    
    Returns:
        float: Score between 0 and 10
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    var_sat = np.var(hsv[:, :, 1])
    score = 10 - min(10, var_sat / 1000 * 10)
    return score

def extract_all_features(image, expected_grade=None):
    """Extract all features from a card image.
    
    This function computes multiple features that are important for card grading:
    1. Border continuity
    2. Corner quality
    3. Centering
    4. Inner cleanliness
    5. Grading index
    6. Edge density
    7. Saturation variance
    
    Args:
        image (numpy.ndarray): Input BGR image
        expected_grade (float, optional): Expected grade of the card
    
    Returns:
        tuple: (features, valid_mask, edges, bbox, inner_region, thicknesses, yellow_mask)
    """
    yellow_mask = get_yellow_mask(image)
    edges = get_edge_map(image)
    _, bbox = find_largest_contour(yellow_mask)
    
    # Compute various feature scores
    continuity = compute_border_continuity(yellow_mask) if bbox is not None else None
    corner_quality = compute_corner_quality(yellow_mask, bbox, patch_size=20) if bbox is not None else None
    centering = compute_centering(yellow_mask, threshold_diff=5) if bbox is not None else None
    inner_region, thicknesses = get_inner_region(image, yellow_mask)
    cleanliness = compute_inner_cleanliness(inner_region, threshold=50) if inner_region is not None else None
    grading_index = compute_grading_index(image, expected_grade=expected_grade)
    edge_density = compute_edge_density(inner_region)
    sat_variance = compute_saturation_variance(image)
    
    # Combine features and handle missing values
    raw_features = [continuity, corner_quality, centering, cleanliness, grading_index, edge_density, sat_variance]
    valid_mask = [f is not None for f in raw_features]
    valid_values = [f for f in raw_features if f is not None]
    avg_value = sum(valid_values) / len(valid_values) if valid_values else 0.0
    features = [float(f) if f is not None else avg_value for f in raw_features]
    
    # Ensure consistent feature length
    if len(features) != 7:
        if len(features) > 7:
            features = features[:7]
        else:
            features = features + [avg_value] * (7 - len(features))
    if len(valid_mask) != 7:
        valid_mask = valid_mask + [False] * (7 - len(valid_mask))
    
    return features, valid_mask, edges, bbox, inner_region, thicknesses, yellow_mask

def clean_feature_matrix(X, valid_masks, threshold=0.3):
    """Clean the feature matrix by removing features with too many missing values.
    
    Args:
        X (numpy.ndarray): Feature matrix
        valid_masks (numpy.ndarray): Boolean masks indicating valid features
        threshold (float): Minimum proportion of valid values required
    
    Returns:
        tuple: (cleaned_X, keep_indices)
    """
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

def visualize_features(image, features, edges, bbox, inner_region, thicknesses, yellow_mask, grade_label=None, show=True):
    """Visualize the extracted features and processing steps.
    
    Args:
        image (numpy.ndarray): Original image
        features (list): Extracted features
        edges (numpy.ndarray): Edge map
        bbox (tuple): Bounding box
        inner_region (numpy.ndarray): Inner region
        thicknesses (tuple): Border thicknesses
        yellow_mask (numpy.ndarray): Yellow mask
        grade_label (str, optional): Expected grade label
        show (bool): Whether to display the visualization
    """
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

def test_feature_extraction(input_dir=os.path.join("data", "processed"), test_first_only=TEST_FIRST_ONLY, visualize=SHOW_VISUALIZATION):
    """Test feature extraction on images in the input directory.
    
    Args:
        input_dir (str): Directory containing processed images
        test_first_only (bool): Whether to test only the first image in each folder
        visualize (bool): Whether to show visualizations
    """
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

def save_feature_dataset(input_dir=os.path.join("data", "processed"), output_file=os.path.join("data", "features", "features.npy.npz")):
    """Save extracted features to a compressed numpy file.
    
    Args:
        input_dir (str): Directory containing processed images
        output_file (str): Path to save the feature dataset
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
        
        # Convert numeric grades to class labels
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
        
        # Process images in the folder
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
    
    # Convert lists to numpy arrays
    X = np.array(X_list, dtype=float)
    valid_masks = np.array([vm if len(vm)==7 else [False]*7 for vm in valid_mask_list], dtype=bool)
    y = np.array(y_list, dtype=int)
    
    # Clean the feature matrix
    cleaned_X, keep_indices = clean_feature_matrix(X, valid_masks, threshold=0.3)
    logger.info(f"After cleaning, kept feature columns: {keep_indices}")
    
    # Save the dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(output_file, X=cleaned_X, y=y)
    logger.info(f"Saved dataset to {output_file}: X.shape={cleaned_X.shape}, y.shape={y.shape}")
