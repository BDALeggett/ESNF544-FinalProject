# Import necessary libraries for image processing, model loading, and feature extraction
import os
import cv2
import numpy as np
import joblib
from feature_extraction import extract_all_features

def load_models():
    """Load trained machine learning models from disk.
    
    This function attempts to load three different models:
    - Random Forest
    - SVM
    - Logistic Regression
    
    Returns:
        list: List of tuples containing (model_name, loaded_model) if successful,
              None if models cannot be loaded
    """
    models = []
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Define paths to model files
    rf_path = os.path.join(model_dir, "random_forest.joblib")
    svm_path = os.path.join(model_dir, "svm.joblib")
    lr_path = os.path.join(model_dir, "logistic_regression.joblib")
    
    # Check if all model files exist
    models_exist = os.path.exists(rf_path) and os.path.exists(svm_path) and os.path.exists(lr_path)
    
    if models_exist:
        try:
            # Load each model from disk
            rf_model = joblib.load(rf_path)
            svm_model = joblib.load(svm_path)
            lr_model = joblib.load(lr_path)
            
            models = [
                ("Random Forest", rf_model),
                ("SVM", svm_model),
                ("Logistic Regression", lr_model)
            ]
            print("Models loaded successfully.")
            return models
        except Exception as e:
            print(f"Error loading models: {e}")
            return None
    else:
        print("Model files not found. Please run model_training.py first to train models.")
        return None

def highlight_card_defects(image_path, predictions):
    """Highlight defects on a Pokemon card image based on predicted grade.
    
    This function analyzes the card image and highlights areas that may have contributed
    to a lower grade prediction, such as:
    - Corner wear
    - Edge wear
    - Centering issues
    
    Args:
        image_path (str): Path to the card image
        predictions (list): List of tuples containing (model_name, grade, confidence)
    
    Returns:
        numpy.ndarray: Image with highlighted defects, or None if processing fails
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    highlighted_image = image.copy()
    
    if not predictions or len(predictions) < 1:
        return highlighted_image
    
    # Get the top prediction
    top_model, top_grade, confidence = predictions[0]
    
    # Convert to HSV color space for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find the card's contour
    contours, _ = cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return highlighted_image
    
    # Get the bounding rectangle of the card
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Different highlighting strategies based on predicted grade
    if top_grade == "≤7":
        # Highlight severe defects for low grade cards
        corner_size = min(w, h) // 8
        corners = [
            (x, y, corner_size, corner_size),  # Top-left
            (x + w - corner_size, y, corner_size, corner_size),  # Top-right
            (x, y + h - corner_size, corner_size, corner_size),  # Bottom-left
            (x + w - corner_size, y + h - corner_size, corner_size, corner_size)  # Bottom-right
        ]
        
        # Check corners for wear
        for cx, cy, cw, ch in corners:
            corner_img = yellow_mask[cy:cy+ch, cx:cx+cw]
            if corner_img.size > 0:
                white_ratio = np.count_nonzero(corner_img) / corner_img.size
                if white_ratio < 0.85:  # Less than 85% yellow in corner indicates damage
                    cv2.rectangle(highlighted_image, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
        
        # Check centering
        left_border = x
        right_border = image.shape[1] - (x + w)
        top_border = y
        bottom_border = image.shape[0] - (y + h)
        
        h_diff = abs(left_border - right_border) / max(left_border, right_border)
        v_diff = abs(top_border - bottom_border) / max(top_border, bottom_border)
        
        # Highlight significant centering issues
        if h_diff > 0.2:
            if left_border < right_border:
                cv2.rectangle(highlighted_image, (0, y), (x, y+h), (0, 0, 255), 2)
            else:
                cv2.rectangle(highlighted_image, (x+w, y), (image.shape[1], y+h), (0, 0, 255), 2)
        
        if v_diff > 0.2:
            if top_border < bottom_border:
                cv2.rectangle(highlighted_image, (x, 0), (x+w, y), (0, 0, 255), 2)
            else:
                cv2.rectangle(highlighted_image, (x, y+h), (x+w, image.shape[0]), (0, 0, 255), 2)
                
    elif top_grade == "8":
        # Highlight moderate defects for grade 8 cards
        corner_size = min(w, h) // 10
        corners = [
            (x, y, corner_size, corner_size),  # Top-left
            (x + w - corner_size, y, corner_size, corner_size),  # Top-right
            (x, y + h - corner_size, corner_size, corner_size),  # Bottom-left
            (x + w - corner_size, y + h - corner_size, corner_size, corner_size)  # Bottom-right
        ]
        
        # Check corners with slightly less strict threshold
        for cx, cy, cw, ch in corners:
            corner_img = yellow_mask[cy:cy+ch, cx:cx+cw]
            if corner_img.size > 0:
                white_ratio = np.count_nonzero(corner_img) / corner_img.size
                if white_ratio < 0.9:  # Slightly less strict threshold
                    cv2.rectangle(highlighted_image, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
        
        # Check edges for wear
        edge_thickness = 5
        edges = [
            (x, y, w, edge_thickness),  # Top edge
            (x, y, edge_thickness, h),  # Left edge
            (x, y + h - edge_thickness, w, edge_thickness),  # Bottom edge
            (x + w - edge_thickness, y, edge_thickness, h)  # Right edge
        ]
        
        for ex, ey, ew, eh in edges:
            edge_img = yellow_mask[ey:ey+eh, ex:ex+ew]
            if edge_img.size > 0:
                white_ratio = np.count_nonzero(edge_img) / edge_img.size
                if white_ratio < 0.92:  # Less than 92% yellow on edge indicates wear
                    cv2.rectangle(highlighted_image, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
                    
    elif top_grade == "9":
        # Highlight minor defects for grade 9 cards
        corner_size = min(w, h) // 12
        corners = [
            (x, y, corner_size, corner_size),  # Top-left
            (x + w - corner_size, y, corner_size, corner_size),  # Top-right
            (x, y + h - corner_size, corner_size, corner_size),  # Bottom-left
            (x + w - corner_size, y + h - corner_size, corner_size, corner_size)  # Bottom-right
        ]
        
        # Check corners with very strict threshold
        for cx, cy, cw, ch in corners:
            corner_img = yellow_mask[cy:cy+ch, cx:cx+cw]
            if corner_img.size > 0:
                white_ratio = np.count_nonzero(corner_img) / corner_img.size
                if white_ratio < 0.95:  # Very strict threshold
                    cv2.rectangle(highlighted_image, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
                    
        # Check centering with stricter tolerance
        left_border = x
        right_border = image.shape[1] - (x + w)
        top_border = y
        bottom_border = image.shape[0] - (y + h)
        
        h_diff = abs(left_border - right_border) / max(left_border, right_border)
        v_diff = abs(top_border - bottom_border) / max(top_border, bottom_border)
        
        # Highlight minor centering issues
        if h_diff > 0.1:  # Only 10% difference allowed
            if left_border < right_border:
                cv2.rectangle(highlighted_image, (0, y), (x, y+h), (0, 0, 255), 2)
            else:
                cv2.rectangle(highlighted_image, (x+w, y), (image.shape[1], y+h), (0, 0, 255), 2)
        
        if v_diff > 0.1:
            if top_border < bottom_border:
                cv2.rectangle(highlighted_image, (x, 0), (x+w, y), (0, 0, 255), 2)
            else:
                cv2.rectangle(highlighted_image, (x, y+h), (x+w, image.shape[0]), (0, 0, 255), 2)
    
    return highlighted_image

def predict_grade(image_path):
    """Predict the grade of a Pokemon card using multiple machine learning models.
    
    This function:
    1. Loads the trained models
    2. Extracts features from the card image
    3. Makes predictions using each model
    4. Returns predictions sorted by confidence
    
    Args:
        image_path (str): Path to the card image
    
    Returns:
        list: List of tuples containing (model_name, grade, confidence) sorted by confidence,
              or None if prediction fails
    """
    models = load_models()
    if not models:
        return None
    
    try:
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image from {image_path}")
            return None
        
        # Extract features from the image
        features, _, _, _, _, _, _ = extract_all_features(image)
        features_array = np.array([features])
        
        # Map numeric grades to string representations
        grade_map = {
            0: "≤7",
            1: "8",
            2: "9",
            3: "10"
        }
        
        predictions = []
        
        # Make predictions using each model
        for model_name, model in models:
            try:
                # Handle different model types and their prediction methods
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features_array)[0]
                else:
                    if hasattr(model, 'decision_function'):
                        if model.classes_.shape[0] == 2:
                            decision = model.decision_function(features_array)
                            probs = np.zeros(2)
                            probs[1] = 1 / (1 + np.exp(-decision))
                            probs[0] = 1 - probs[1]
                        else:
                            decision = model.decision_function(features_array)
                            probs = np.exp(decision) / np.sum(np.exp(decision), axis=1).reshape(-1, 1)
                    else:
                        pred = model.predict(features_array)[0]
                        probs = np.zeros(len(grade_map))
                        probs[pred] = 1.0
                
                # Get the top prediction and confidence
                top_idx = np.argmax(probs)
                grade = grade_map[top_idx]
                confidence = probs[top_idx] * 100
                predictions.append((model_name, grade, confidence))
            
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
        
        # Sort predictions by confidence
        predictions.sort(key=lambda x: x[2], reverse=True)
        
        return predictions
        
    except Exception as e:
        print(f"Error predicting grade: {e}")
        return None

def save_image_with_defects(image_path, output_path, predictions):
    """Save an image with highlighted defects to disk.
    
    Args:
        image_path (str): Path to the original card image
        output_path (str): Path where the highlighted image should be saved
        predictions (list): List of tuples containing (model_name, grade, confidence)
    
    Returns:
        bool: True if successful, False otherwise
    """
    highlighted_image = highlight_card_defects(image_path, predictions)
    if highlighted_image is not None:
        try:
            cv2.imwrite(output_path, highlighted_image)
            print(f"Saved highlighted image to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving highlighted image: {e}")
    return False