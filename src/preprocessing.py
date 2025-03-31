import os
import cv2
import numpy as np
import logging
from glob import glob
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_border_bbox(image, margin=0.1, min_aspect_ratio=1.2):
    """
    Detect the bounding rectangle of the card's border by combining two color ranges:
      1) Typical "yellow" border
      2) Pale/brownish range (approx #d4c398)
    We then pick the bounding box that meets a vertical aspect ratio requirement 
    (height >= min_aspect_ratio * width).
    
    If found, returns (x, y, w, h). If not found, returns None.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Range #1: Typical yellow
    lower_yellow = np.array([15, 70, 70], dtype=np.uint8)
    upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Range #2: Pale/brownish approx #d4c398
    lower_brown = np.array([10, 40, 80], dtype=np.uint8)
    upper_brown = np.array([40, 180, 255], dtype=np.uint8)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # Combine both masks
    mask = cv2.bitwise_or(mask_yellow, mask_brown)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("No border contours found for either color range.")
        return None

    # Filter by vertical aspect ratio
    valid_bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= min_aspect_ratio * w:
            valid_bboxes.append((x, y, w, h))

    if not valid_bboxes:
        logger.warning("No contours with sufficient vertical aspect ratio found.")
        return None

    # Choose bounding box with largest area
    x, y, w, h = max(valid_bboxes, key=lambda b: b[2] * b[3])

    # Expand the bounding box by margin
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    x_expanded = max(0, x - margin_w)
    y_expanded = max(0, y - margin_h)
    w_expanded = min(image.shape[1] - x_expanded, w + 2 * margin_w)
    h_expanded = min(image.shape[0] - y_expanded, h + 2 * margin_h)

    return (x_expanded, y_expanded, w_expanded, h_expanded)

def segment_card(image, margin=0.1):
    """
    Crop the image based on the bounding rectangle of the merged color ranges.
    Returns the cropped image if found; otherwise returns None (meaning skip).
    """
    bbox = find_border_bbox(image, margin=margin)
    if bbox is None:
        return None  # Indicate that no valid bounding box was found
    x, y, w, h = bbox
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
