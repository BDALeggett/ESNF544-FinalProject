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
    Detect the bounding rectangle of the card's border focusing only on the yellow color range.
    We pick the bounding box that meets a vertical aspect ratio requirement 
    (height >= min_aspect_ratio * width).
    
    If found, returns (x, y, w, h). If not found, returns None.
    
    Args:
        image: Input BGR image
        margin: Margin to add around detected bounding box
        min_aspect_ratio: Minimum height/width ratio for valid bounding box
        debug: If True, return debug visualization
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    if image.shape[2] == 4:
        image = image[:,:,:3]
    
    img_h, img_w = image.shape[:2]
    
    # Add a black border around the image to prevent detecting the entire frame
    # This helps distinguish the card from the full image frame
    bordered_image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    bordered_hsv = cv2.cvtColor(bordered_image, cv2.COLOR_BGR2HSV)

    # Expanded yellow range to catch more card borders
    lower_yellow = np.array([10, 50, 70], dtype=np.uint8)   # Broadened hue range (10-40)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(bordered_hsv, lower_yellow, upper_yellow)

    # Broader brown range
    lower_brown = np.array([5, 30, 80], dtype=np.uint8)
    upper_brown = np.array([45, 200, 255], dtype=np.uint8)
    mask_brown = cv2.inRange(bordered_hsv, lower_brown, upper_brown)
    
    # Combine yellow and brown masks
    mask_combined = cv2.bitwise_or(mask_yellow, mask_brown)
    
    # Morphological operations to improve detection
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
    mask_combined = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    
    # Find contours on the combined mask
    contours_combined, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_sets = []
    if contours_combined:
        contour_sets.append(("combined", contours_combined))
    
    # Only if combined doesn't work, try individual color ranges
    if not contour_sets or max([cv2.contourArea(cnt) for cnt in contours_combined]) < 5000:
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_yellow:
            contour_sets.append(("yellow", contours_yellow))
        
        contours_brown, _ = cv2.findContours(mask_brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_brown:
            contour_sets.append(("brown", contours_brown))
    
    if not contour_sets:
        logger.warning("No border contours found for any color range.")
        return None
    
    all_valid_bboxes = []
    
    # Filter out contours that are too close to the image size (full frame detections)
    full_image_area = img_w * img_h
    
    for name, contours in contour_sets:
        valid_bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # Skip tiny contours
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Adjust for the added border
            x -= 20
            y -= 20
            
            # Clamp to image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(img_w - x, w)
            h = min(img_h - y, h)
            
            # Skip if it's too close to the full image size (within 5%)
            contour_area = w * h
            if contour_area > 0.95 * full_image_area:
                continue
                
            if h >= min_aspect_ratio * w and w * h > 5000:
                valid_bboxes.append((x, y, w, h))
        
        if valid_bboxes:
            best_bbox = max(valid_bboxes, key=lambda b: b[2] * b[3])
            all_valid_bboxes.append((name, best_bbox))
    
    if not all_valid_bboxes:
        # Fallback: Try edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        dilated = cv2.dilate(edges, None, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            contour_area = w * h
            
            # Skip if it's too close to the full image size
            if contour_area > 0.95 * full_image_area:
                continue
                
            if h >= min_aspect_ratio * w and contour_area > 5000:
                valid_bboxes.append((x, y, w, h))
        
        if valid_bboxes:
            best_bbox = max(valid_bboxes, key=lambda b: b[2] * b[3])
            all_valid_bboxes.append(("edge", best_bbox))
    
    if not all_valid_bboxes:
        logger.warning("No contours with sufficient vertical aspect ratio found.")
        return None

    color, (x, y, w, h) = max(all_valid_bboxes, key=lambda item: item[1][2] * item[1][3])
    logger.info(f"Selected bounding box from {color} color range")

    margin_w = int(w * margin)
    margin_h = int(h * margin)
    x_expanded = max(0, x - margin_w)
    y_expanded = max(0, y - margin_h)
    w_expanded = min(image.shape[1] - x_expanded, w + 2 * margin_w)
    h_expanded = min(image.shape[0] - y_expanded, h + 2 * margin_h)

    if debug:
        debug_img = image.copy()
        cv2.rectangle(debug_img, (x_expanded, y_expanded), 
                     (x_expanded + w_expanded, y_expanded + h_expanded), (0, 255, 0), 2)
        
        mask_vis = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        mask_vis[:,:,1] = mask_combined[20:-20, 20:-20]  # Adjust for the border
        
        return (x_expanded, y_expanded, w_expanded, h_expanded), debug_img, mask_vis

    return (x_expanded, y_expanded, w_expanded, h_expanded)

def segment_card(image, margin=0.1, debug=False, remove_background=True):
    """
    Crop the image based on the bounding rectangle of the merged color ranges.
    Returns the cropped image if found; otherwise returns None (meaning skip).
    
    Args:
        image: Input image
        margin: Margin to add around detected bounding box
        debug: If True, return debug visualization
        remove_background: If True, apply GrabCut to remove background; False to skip bg removal
    """

    if remove_background:
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        h, w = image.shape[:2]
        rect = (10, 10, w-20, h-20)
        
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        image_no_bg = image.copy()
        for c in range(0, 3):
            image_no_bg[:, :, c] = image[:, :, c] * mask2
        
        # Use the version with background removed for finding borders
        result = find_border_bbox(image_no_bg, margin=margin, debug=debug)
    else:
        # Skip background removal and use original image
        result = find_border_bbox(image, margin=margin, debug=debug)
    
    if debug and result is not None:
        bbox, debug_img, mask_vis = result
        x, y, w, h = bbox
        cropped = image[y:y+h, x:x+w]
        return cropped, debug_img, mask_vis, image_no_bg if remove_background else image
    
    if result is None:
        return None
    
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

def preprocess_images(input_dir=os.path.join("data", "cards"),
                      output_dir=os.path.join("data", "processed"),
                      margin=0.005,
                      remove_background=True):
    
    abs_input_dir = os.path.abspath(input_dir)
    abs_output_dir = os.path.abspath(output_dir)
    logger.info(f"Input directory: {abs_input_dir}")
    logger.info(f"Output directory: {abs_output_dir}")
    logger.info(f"Background removal: {'enabled' if remove_background else 'disabled'}")

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

            cropped = segment_card(image, margin=margin, remove_background=remove_background)
            if cropped is None:
                logger.warning(f"No suitable card border found, skipping: {img_path}")
                continue

            if not valid_card_dimensions(cropped):
                logger.warning(f"Cropped image does not meet card dimensions; skipping: {img_path}")
                continue

            out_path = os.path.join(out_folder, os.path.basename(img_path))
            cv2.imwrite(out_path, cropped)
            logger.info(f"Saved processed image: {out_path}")

def display_image(image, title="Image"):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()
