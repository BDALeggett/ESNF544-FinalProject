import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_feature_dataset(feature_file=None):
    """
    Load the feature dataset from a .npz file
    """
    if feature_file is None:
        # Try to find the feature file in the data/features directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        feature_file = os.path.join(base_dir, "data", "features", "features.npy.npz")
    
    if not os.path.exists(feature_file):
        logger.error(f"Feature file not found: {feature_file}")
        return None, None
    
    try:
        data = np.load(feature_file)
        X = data["X"]
        y = data["y"]
        logger.info(f"Loaded dataset: X.shape={X.shape}, y.shape={y.shape}")
        return X, y
    except Exception as e:
        logger.error(f"Error loading feature dataset: {e}")
        return None, None

def train_models(X, y):
    """
    Train the three models (Random Forest, SVM, Logistic Regression)
    and return them as a list of tuples (model_name, model)
    """
    models = []
    
    logger.info("Resampling data using SMOTE...")
    # Use SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    logger.info(f"Original data shape: {X.shape}, Resampled data shape: {X_resampled.shape}")
    
    # Train Random Forest
    logger.info("Training Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_resampled, y_resampled)
    models.append(("Random Forest", rf_model))
    
    # Train SVM with StandardScaler in a pipeline
    logger.info("Training SVM classifier...")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            random_state=42
        ))
    ])
    svm_pipeline.fit(X_resampled, y_resampled)
    models.append(("SVM", svm_pipeline))
    
    # Train Logistic Regression with StandardScaler in a pipeline
    logger.info("Training Logistic Regression classifier...")
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C=1.0,
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42
        ))
    ])
    lr_pipeline.fit(X_resampled, y_resampled)
    models.append(("Logistic Regression", lr_pipeline))
    
    return models

def save_models(models):
    """
    Save the trained models to disk
    """
    # Create models directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    for model_name, model in models:
        # Create safe filename from model name
        filename = model_name.lower().replace(" ", "_") + ".joblib"
        filepath = os.path.join(model_dir, filename)
        
        logger.info(f"Saving {model_name} to {filepath}...")
        try:
            joblib.dump(model, filepath)
            logger.info(f"Successfully saved {model_name}")
        except Exception as e:
            logger.error(f"Error saving {model_name}: {e}")

def main():
    """
    Main function to train and save models
    """
    logger.info("Starting model training and saving process...")
    
    # Load feature dataset
    X, y = load_feature_dataset()
    if X is None or y is None:
        logger.error("Failed to load feature dataset. Exiting.")
        return
    
    # Train models
    models = train_models(X, y)
    
    # Save models
    save_models(models)
    
    logger.info("Model training and saving process completed.")

if __name__ == "__main__":
    main() 