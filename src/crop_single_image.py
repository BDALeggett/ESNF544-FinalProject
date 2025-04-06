import os
import cv2
import numpy as np
import sys
from preprocessing import find_border_bbox, segment_card, valid_card_dimensions

def crop_pokemon_card(image):
    """
    Crops a single Pokemon card image based on border detection.
    
    Args:
        image: OpenCV image object (from cv2.imread)
    
    Returns:
        cropped image if cropping was successful, None otherwise
    """
    if image is None:
        print("Error: Invalid image provided")
        return None
    
    print(f"Image loaded, shape: {image.shape}")
    
    # Try different margin values for better detection
    cropped_img = segment_card(image, margin=0.005)
    
    # If color-based segmentation failed, try edge detection approach
    if cropped_img is None:
        print("Color-based segmentation failed, trying edge detection...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        # Dilate to connect edges
        dilated = cv2.dilate(edges, None, iterations=1)
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add small margin
            margin_px = 10
            x = max(0, x - margin_px)
            y = max(0, y - margin_px)
            w = min(image.shape[1] - x, w + 2*margin_px)
            h = min(image.shape[0] - y, h + 2*margin_px)
            
            # Crop the image
            cropped_img = image[y:y+h, x:x+w]
            
            if valid_card_dimensions(cropped_img):
                print("Successfully cropped using edge detection")
            else:
                print("Edge detection produced an invalid card crop")
                cropped_img = None
    
    if cropped_img is None:
        print("Failed to crop the card - no valid crop found")
        return None
    
    # Display cropping information
    original_h, original_w = image.shape[:2]
    cropped_h, cropped_w = cropped_img.shape[:2]
    print(f"Original dimensions: {original_w}x{original_h}")
    print(f"Cropped dimensions: {cropped_w}x{cropped_h}")
    print(f"Reduction: {100 - (cropped_w*cropped_h)/(original_w*original_h)*100:.1f}%")
    
    return cropped_img

if __name__ == "__main__":
    input_image_path = "src/testImage2.jpg"
    output_image_path = "src/cropped.jpg"
    
    # Allow command line arguments to override defaults
    if len(sys.argv) > 1:
        input_image_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_image_path = sys.argv[2]
    
    # Read the image
    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Could not read image {input_image_path}")
        sys.exit(1)
        
    # Crop the image
    cropped_image = crop_pokemon_card(image)
    
    if cropped_image is not None:
        # Save the cropped image
        cv2.imwrite(output_image_path, cropped_image)
        print(f"Cropped image saved to: {output_image_path}")
        print("Card cropping completed successfully")
    else:
        print("Card cropping failed")
        sys.exit(1) 