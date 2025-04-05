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
        else:
            print("Failed to predict card grade. Make sure the image contains a clear view of a Pokemon card.")

if __name__ == "__main__":
    main() 