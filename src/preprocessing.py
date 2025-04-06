import os
import cv2
import numpy as np
import logging
from glob import glob
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_border_bbox(image, margin=0.1, min_aspect_ratio=1.2, debug=False):
    """
    Detect the bounding rectangle of the card's border by combining multiple color ranges:
      1) Typical "yellow" border
      2) Pale/brownish range (approx #d4c398)
      3) Silver/gray range (for colorless/steel type cards)
      4) Light gray with low saturation (for some EX cards)
    We then pick the bounding box that meets a vertical aspect ratio requirement 
    (height >= min_aspect_ratio * width).
    
    If found, returns (x, y, w, h). If not found, returns None.
    
    Args:
        image: Input BGR image
        margin: Margin to add around detected bounding box
        min_aspect_ratio: Minimum height/width ratio for valid bounding box
        debug: If True, return debug visualization
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Check for alpha channel and handle it
    has_alpha = False
    if image.shape[2] == 4:
        has_alpha = True
        # Extract alpha channel for additional mask processing
        alpha_channel = image[:,:,3]
        # Convert to standard BGR for main processing
        image = image[:,:,:3]

    # Range #1: Typical yellow
    lower_yellow = np.array([15, 70, 70], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Range #2: Pale/brownish approx #d4c398
    lower_brown = np.array([10, 40, 80], dtype=np.uint8)
    upper_brown = np.array([40, 180, 255], dtype=np.uint8)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Range #3: Silver/gray range for colorless cards (mid-range)
    lower_gray1 = np.array([0, 0, 120], dtype=np.uint8)
    upper_gray1 = np.array([180, 30, 200], dtype=np.uint8)
    mask_gray1 = cv2.inRange(hsv, lower_gray1, upper_gray1)
    
    # Range #4: Light silver/gray range (brighter)
    lower_gray2 = np.array([0, 0, 200], dtype=np.uint8)
    upper_gray2 = np.array([180, 20, 255], dtype=np.uint8)
    mask_gray2 = cv2.inRange(hsv, lower_gray2, upper_gray2)
    
    # Range #5: Darker gray/silver (for shadowed areas)
    lower_gray3 = np.array([0, 0, 80], dtype=np.uint8)
    upper_gray3 = np.array([180, 40, 140], dtype=np.uint8)
    mask_gray3 = cv2.inRange(hsv, lower_gray3, upper_gray3)

    # Combine masks for processing
    mask_yellow_brown = cv2.bitwise_or(mask_yellow, mask_brown)
    mask_all_grays = cv2.bitwise_or(cv2.bitwise_or(mask_gray1, mask_gray2), mask_gray3)
    
    # Process each mask separately to find the best bounding box
    contour_sets = []
    
    # Try yellow+brown mask (original approach)
    contours_yellow_brown, _ = cv2.findContours(mask_yellow_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_yellow_brown:
        contour_sets.append(("yellow/brown", contours_yellow_brown))
    
    # Try combined gray masks for better detection
    contours_gray, _ = cv2.findContours(mask_all_grays, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_gray:
        contour_sets.append(("gray", contours_gray))
    
    # Try combined mask as fallback
    if not contour_sets:
        mask_combined = cv2.bitwise_or(mask_yellow_brown, mask_all_grays)
        contours_combined, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_combined:
            contour_sets.append(("combined", contours_combined))
    
    if not contour_sets:
        logger.warning("No border contours found for any color range.")
        return None
    
    # Process each set of contours to find valid bounding boxes
    all_valid_bboxes = []
    
    for name, contours in contour_sets:
        # Filter by vertical aspect ratio
        valid_bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # More permissive area filtering to catch edge cases
            if h >= min_aspect_ratio * w and w * h > 1000:  # Minimum area to avoid noise
                valid_bboxes.append((x, y, w, h))
        
        if valid_bboxes:
            # Find the largest valid bbox for this color set
            best_bbox = max(valid_bboxes, key=lambda b: b[2] * b[3])
            all_valid_bboxes.append((name, best_bbox))
    
    if not all_valid_bboxes:
        logger.warning("No contours with sufficient vertical aspect ratio found.")
        return None

    # Choose bounding box with largest area among all color sets
    color, (x, y, w, h) = max(all_valid_bboxes, key=lambda item: item[1][2] * item[1][3])
    logger.info(f"Selected bounding box from {color} color range")

    # Expand the bounding box by margin
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    x_expanded = max(0, x - margin_w)
    y_expanded = max(0, y - margin_h)
    w_expanded = min(image.shape[1] - x_expanded, w + 2 * margin_w)
    h_expanded = min(image.shape[0] - y_expanded, h + 2 * margin_h)

    # If debug is enabled, create a visualization
    if debug:
        debug_img = image.copy()
        # Draw the final bounding box
        cv2.rectangle(debug_img, (x_expanded, y_expanded), 
                     (x_expanded + w_expanded, y_expanded + h_expanded), (0, 255, 0), 2)
        
        # Create visualizations of the masks
        mask_vis = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        mask_vis[:,:,0] = mask_all_grays  # Blue channel for gray mask
        mask_vis[:,:,1] = mask_yellow_brown  # Green channel for yellow/brown mask
        
        return (x_expanded, y_expanded, w_expanded, h_expanded), debug_img, mask_vis

    return (x_expanded, y_expanded, w_expanded, h_expanded)

def segment_card(image, margin=0.1, debug=False):
    """
    Crop the image based on the bounding rectangle of the merged color ranges.
    Returns the cropped image if found; otherwise returns None (meaning skip).
    
    Args:
        image: Input image
        margin: Margin to add around detected bounding box
        debug: If True, return debug visualization
    """

    # Create a mask for the card using GrabCut algorithm
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Initialize rectangle for GrabCut
    h, w = image.shape[:2]
    rect = (10, 10, w-20, h-20)  # Assume card is roughly centered in image
    
    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create mask where definite and probable foreground are set to 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Apply the mask to the image
    image_no_bg = image.copy()
    for c in range(0, 3):
        image_no_bg[:, :, c] = image[:, :, c] * mask2

    # display_image(image_no_bg, title="Card with Background Removed")
    
    result = find_border_bbox(image_no_bg, margin=margin, debug=debug)
    
    if debug and result is not None:
        bbox, debug_img, mask_vis = result
        x, y, w, h = bbox
        cropped = image[y:y+h, x:x+w]
        return cropped, debug_img, mask_vis, image_no_bg
    
    if result is None:
        return None  # Indicate that no valid bounding box was found
    
    x, y, w, h = result
    return image[y:y+h, x:x+w]

def valid_card_dimensions(cropped, min_aspect=1.3, max_aspect=1.5, min_width=100, min_height=150):
    """
    Check if the cropped image meets expected Pokémon card dimensions.
    - Aspect ratio (height/width) must be between min_aspect and max_aspect.
    - Width must be at least min_width and height at least min_height.
    Returns True if valid, False otherwise.
    """
    h, w = cropped.shape[:2]
    aspect = h / float(w)
    if w < min_width or h < min_height:
        logger.warning(f"Cropped image too small: width={w}, height={h}")
        return False
    if aspect < min_aspect or aspect > max_aspect:
        logger.warning(f"Aspect ratio {aspect:.2f} not within acceptable range ({min_aspect}-{max_aspect}).")
        return False
    return True

def preprocess_images(input_dir="ENSF 544\\Final-Project\\data\\cards",
                      output_dir="ENSF 544\\Final-Project\\data\\processed",
                      margin=0.005):
    """
    1. Reads images from input_dir (organized by grade folders).
    2. Detects card border from color-based approach (yellow + pale brown).
    3. If found, crops with margin and saves to output_dir.
    4. If not found, we skip that image (i.e., do not write to the processed folder).
    """
    abs_input_dir = os.path.abspath(input_dir)
    abs_output_dir = os.path.abspath(output_dir)
    logger.info(f"Input directory: {abs_input_dir}")
    logger.info(f"Output directory: {abs_output_dir}")

    if not os.path.exists(abs_output_dir):
        os.makedirs(abs_output_dir)

    grade_folders = sorted(glob(os.path.join(abs_input_dir, "grade_*")))
    logger.info(f"Found grade folders: {grade_folders}")

    for grade_folder in grade_folders:
        grade_name = os.path.basename(grade_folder)
        logger.info(f"Processing folder: {grade_name}")
        out_folder = os.path.join(abs_output_dir, grade_name)
        os.makedirs(out_folder, exist_ok=True)

        img_files = sorted(glob(os.path.join(grade_folder, "*.jpg")))
        logger.info(f"Found {len(img_files)} images in {grade_folder}")

        for img_path in img_files:
            logger.info(f"Processing image: {img_path}")
            image = cv2.imread(img_path)
            if image is None:
                logger.warning(f"Could not load image: {img_path}")
                continue

            cropped = segment_card(image, margin=margin)
            if cropped is None:
                # We skip this image from the dataset
                logger.warning(f"No suitable card border found, skipping: {img_path}")
                continue

           # Check if the cropped image meets expected dimensions
            if not valid_card_dimensions(cropped):
                logger.warning(f"Cropped image does not meet card dimensions; skipping: {img_path}")
                continue

            out_path = os.path.join(out_folder, os.path.basename(img_path))
            cv2.imwrite(out_path, cropped)
            logger.info(f"Saved processed image: {out_path}")

def display_image(image, title="Image"):
    """
    Utility to display an image with matplotlib.
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    preprocess_images(
        input_dir="ENSF 544\\Final-Project\\data\\cards",
        output_dir="ENSF 544\\Final-Project\\data\\processed",
        margin=0.005
    )
    print("Preprocessing completed.")
