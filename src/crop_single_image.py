import cv2
from preprocessing import segment_card, valid_card_dimensions

# Toggle for background removal - set to True to enable it, False to disable
REMOVE_BACKGROUND = False  # Disabled - will crop the card without removing background

def crop_pokemon_card(image):
    if image is None:
        print("Error: Invalid image provided")
        return None
    
    print(f"Image loaded, shape: {image.shape}")
    
    cropped_img = segment_card(image, margin=0.005, remove_background=REMOVE_BACKGROUND)
    
    if cropped_img is None:
        print("Color-based segmentation failed, trying edge detection...")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        dilated = cv2.dilate(edges, None, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            margin_px = 10
            x = max(0, x - margin_px)
            y = max(0, y - margin_px)
            w = min(image.shape[1] - x, w + 2*margin_px)
            h = min(image.shape[0] - y, h + 2*margin_px)
            
            cropped_img = image[y:y+h, x:x+w]
            
            if valid_card_dimensions(cropped_img):
                print("Successfully cropped using edge detection")
            else:
                print("Edge detection produced an invalid card crop")
                cropped_img = None
    
    if cropped_img is None:
        print("Failed to crop the card - no valid crop found")
        return None
    
    original_h, original_w = image.shape[:2]
    cropped_h, cropped_w = cropped_img.shape[:2]
    print(f"Original dimensions: {original_w}x{original_h}")
    print(f"Cropped dimensions: {cropped_w}x{cropped_h}")
    print(f"Reduction: {100 - (cropped_w*cropped_h)/(original_w*original_h)*100:.1f}%")
    
    return cropped_img