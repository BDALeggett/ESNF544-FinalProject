import os
import cv2
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from feature_extraction import extract_all_features, clean_feature_matrix

def load_models():
    """
    Load the trained models. If models don't exist, train them first.
    Returns a list of tuples (model_name, model)
    """
    models = []
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Define model file paths
    rf_path = os.path.join(model_dir, "random_forest.joblib")
    svm_path = os.path.join(model_dir, "svm.joblib")
    lr_path = os.path.join(model_dir, "logistic_regression.joblib")
    
    # Check if model files exist
    models_exist = os.path.exists(rf_path) and os.path.exists(svm_path) and os.path.exists(lr_path)
    
    if models_exist:
        # Load models
        try:
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
    """
    Analyze the card image and highlight potential defects with red boxes
    Returns the original image with highlighted defects
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Create a copy for highlighting
    highlighted_image = image.copy()
    
    # Get predicted grade (use the top prediction)
    if not predictions or len(predictions) < 1:
        return highlighted_image
    
    top_model, top_grade, confidence = predictions[0]
    
    # Process the image to find the yellow border
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Find contours of the yellow border
    contours, _ = cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return highlighted_image
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Different highlighting strategies based on grade
    if top_grade == "≤7":
        # For poor grades, highlight multiple issues
        
        # Check corners
        corner_size = min(w, h) // 8
        corners = [
            (x, y, corner_size, corner_size),  # Top-left
            (x + w - corner_size, y, corner_size, corner_size),  # Top-right
            (x, y + h - corner_size, corner_size, corner_size),  # Bottom-left
            (x + w - corner_size, y + h - corner_size, corner_size, corner_size)  # Bottom-right
        ]
        
        # Highlight corners with issues
        for cx, cy, cw, ch in corners:
            corner_img = yellow_mask[cy:cy+ch, cx:cx+cw]
            if corner_img.size > 0:
                white_ratio = np.count_nonzero(corner_img) / corner_img.size
                if white_ratio < 0.85:  # Less than 85% yellow in corner indicates damage
                    cv2.rectangle(highlighted_image, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
        
        # Highlight centering issues
        left_border = x
        right_border = image.shape[1] - (x + w)
        top_border = y
        bottom_border = image.shape[0] - (y + h)
        
        # If borders differ by more than 20%, highlight the smaller border
        h_diff = abs(left_border - right_border) / max(left_border, right_border)
        v_diff = abs(top_border - bottom_border) / max(top_border, bottom_border)
        
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
        # For grade 8, highlight moderate issues
        
        # Check corners with less strict criteria
        corner_size = min(w, h) // 10
        corners = [
            (x, y, corner_size, corner_size),  # Top-left
            (x + w - corner_size, y, corner_size, corner_size),  # Top-right
            (x, y + h - corner_size, corner_size, corner_size),  # Bottom-left
            (x + w - corner_size, y + h - corner_size, corner_size, corner_size)  # Bottom-right
        ]
        
        for cx, cy, cw, ch in corners:
            corner_img = yellow_mask[cy:cy+ch, cx:cx+cw]
            if corner_img.size > 0:
                white_ratio = np.count_nonzero(corner_img) / corner_img.size
                if white_ratio < 0.9:  # Slightly less strict threshold
                    cv2.rectangle(highlighted_image, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
        
        # Check for edge wear
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
        # For grade 9, highlight minor issues
        
        # Check only specific areas for very minor issues
        # Mainly just look at corners with stricter criteria
        corner_size = min(w, h) // 12
        corners = [
            (x, y, corner_size, corner_size),  # Top-left
            (x + w - corner_size, y, corner_size, corner_size),  # Top-right
            (x, y + h - corner_size, corner_size, corner_size),  # Bottom-left
            (x + w - corner_size, y + h - corner_size, corner_size, corner_size)  # Bottom-right
        ]
        
        for cx, cy, cw, ch in corners:
            corner_img = yellow_mask[cy:cy+ch, cx:cx+cw]
            if corner_img.size > 0:
                white_ratio = np.count_nonzero(corner_img) / corner_img.size
                if white_ratio < 0.95:  # Very strict threshold
                    cv2.rectangle(highlighted_image, (cx, cy), (cx+cw, cy+ch), (0, 0, 255), 2)
                    
        # Check for subtle centering issues
        left_border = x
        right_border = image.shape[1] - (x + w)
        top_border = y
        bottom_border = image.shape[0] - (y + h)
        
        # Much stricter centering criteria for grade 9
        h_diff = abs(left_border - right_border) / max(left_border, right_border)
        v_diff = abs(top_border - bottom_border) / max(top_border, bottom_border)
        
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
    
    # For grade 10, we don't highlight anything as it's near perfect
    
    return highlighted_image

def predict_grade(image_path):
    """
    Predict the grade of a Pokemon card from an image.
    Returns a list of top 3 grade predictions with confidence scores.
    """
    # Load models
    models = load_models()
    if not models:
        return None
    
    # Load and preprocess the image
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image from {image_path}")
            return None
        
        # Extract features from the image
        features, _, _, _, _, _, _ = extract_all_features(image)
        
        # Convert the features to a numpy array
        features_array = np.array([features])
        
        # Grade mapping
        grade_map = {
            0: "≤7",
            1: "8",
            2: "9",
            3: "10"
        }
        
        predictions = []
        
        # Get predictions from each model
        for model_name, model in models:
            try:
                # Predict probabilities for each class
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features_array)[0]
                else:
                    # If model doesn't support predict_proba (like some SVMs), use decision function
                    if hasattr(model, 'decision_function'):
                        # For binary classification
                        if model.classes_.shape[0] == 2:
                            decision = model.decision_function(features_array)
                            probs = np.zeros(2)
                            probs[1] = 1 / (1 + np.exp(-decision))
                            probs[0] = 1 - probs[1]
                        else:
                            # One-vs-rest strategy for multiclass
                            decision = model.decision_function(features_array)
                            probs = np.exp(decision) / np.sum(np.exp(decision), axis=1).reshape(-1, 1)
                    else:
                        # Just use the prediction as a single probability of 1.0
                        pred = model.predict(features_array)[0]
                        probs = np.zeros(len(grade_map))
                        probs[pred] = 1.0
                
                # Get top 3 predictions
                top_indices = np.argsort(probs)[::-1][:3]
                
                for idx in top_indices:
                    grade = grade_map[idx]
                    confidence = probs[idx] * 100
                    predictions.append((model_name, grade, confidence))
            
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
        
        # Sort all predictions by confidence
        predictions.sort(key=lambda x: x[2], reverse=True)
        
        # Return top 3 overall predictions
        return predictions[:3]
        
    except Exception as e:
        print(f"Error predicting grade: {e}")
        return None

def save_image_with_defects(image_path, output_path, predictions):
    """
    Save a new image with card defects highlighted
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

def main():
    """
    Simple CLI interface to predict Pokemon card grades.
    """
    print("Pokemon Card Grade Predictor")
    print("============================")
    
    while True:
        print("\nEnter the path to your Pokemon card image (or 'q' to quit):")
        image_path = input("> ").strip()
        
        if image_path.lower() == 'q':
            break
        
        if not os.path.exists(image_path):
            print(f"Error: File '{image_path}' not found.")
            continue
        
        print(f"Processing image: {image_path}")
        predictions = predict_grade(image_path)
        
        if predictions:
            print("\nTop 3 Grade Predictions:")
            print("-----------------------")
            for i, (model_name, grade, confidence) in enumerate(predictions, 1):
                print(f"{i}. Grade {grade} ({model_name}) - Confidence: {confidence:.2f}%")
            
            # Ask if user wants to save an image with defects highlighted
            print("\nWould you like to see the card defects highlighted? (y/n)")
            show_defects = input("> ").strip().lower()
            
            if show_defects == 'y':
                # Get the output filename
                base_name = os.path.splitext(image_path)[0]
                output_path = f"{base_name}_defects.jpg"
                
                # Save the highlighted image
                if save_image_with_defects(image_path, output_path, predictions):
                    # Show the image if on a system with display capability
                    try:
                        highlighted_img = cv2.imread(output_path)
                        cv2.imshow("Card Defects", highlighted_img)
                        print("Press any key to close the image window...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    except Exception as e:
                        print(f"Unable to display image: {e}")
                        print(f"The highlighted image has been saved to: {output_path}")
        else:
            print("Failed to predict card grade. Make sure the image contains a clear view of a Pokemon card.")

if __name__ == "__main__":
    main() 