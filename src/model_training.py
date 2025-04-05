import os
import numpy as np
import logging
import time
import warnings
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, 
    precision_score, recall_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import sys

# Suppress all warnings
warnings.filterwarnings('ignore')

# Advanced resampling techniques for handling class imbalance
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

# Configure logging to exclude warnings
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Filter out warning messages from logger
logging.getLogger('py.warnings').setLevel(logging.ERROR)

# Custom callback class for GridSearchCV progress tracking
class ProgressCallback:
    def __init__(self, total_iters, desc="Progress"):
        self.total_iters = total_iters
        self.current_iter = 0
        self.start_time = time.time()
        self.pbar = tqdm(total=total_iters, desc=desc, file=sys.stdout)
        
    def __call__(self, params, score, n_iter):
        self.current_iter += 1
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.current_iter if self.current_iter > 0 else 0
        eta = avg_time * (self.total_iters - self.current_iter)
        
        self.pbar.set_postfix({
            'best_score': f"{score:.3f}", 
            'eta': f"{eta:.1f}s"
        })
        self.pbar.update(1)
        
    def close(self):
        self.pbar.close()

def load_feature_dataset(feature_file="/data/features/features.npy.npz"):
    if not os.path.exists(feature_file):
        logger.error(f"Feature file not found: {feature_file}")
        return None, None
    data = np.load(feature_file)
    X = data["X"]
    y = data["y"]
    logger.info(f"Loaded dataset: X.shape={X.shape}, y.shape={y.shape}")
    return X, y

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def tune_model(model, param_grid, X, y, cv=5, scoring="f1_macro"):
    """
    Tune model hyperparameters using GridSearchCV with progress tracking.
    """
    # Calculate total number of fits
    n_fits = cv * np.prod([len(values) for values in param_grid.values()])
    logger.info(f"Starting {model.__class__.__name__} tuning: {n_fits} total fits")
    
    # Display progress message for user
    logger.info(f"Hyperparameter search for {model.__class__.__name__}: fitting {cv} folds for {n_fits//cv} candidates")
    start_time = time.time()
    
    # Create progress callback 
    progress_cb = ProgressCallback(total_iters=n_fits, desc=f"Tuning {model.__class__.__name__}")
    
    # Run grid search with higher verbosity and callback
    grid = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring, n_jobs=-1, 
        verbose=2, error_score='raise',
        pre_dispatch='2*n_jobs'
    )
    
    # Use a progress bar for the overall fitting process
    with tqdm(total=1, desc=f"GridSearchCV ({model.__class__.__name__})", file=sys.stdout) as pbar:
        grid.fit(X, y)
        pbar.update(1)
    
    # Close the callback progress bar
    if hasattr(progress_cb, 'close'):
        progress_cb.close()
    
    elapsed = time.time() - start_time
    logger.info(f"Completed {model.__class__.__name__} tuning in {elapsed:.1f} seconds")
    logger.info(f"Best params for {model.__class__.__name__}: {grid.best_params_}")
    logger.info(f"Best {scoring} score: {grid.best_score_:.3f}")
    
    return grid.best_estimator_

def evaluate_model(model, X, y, folds=5, resampling_method='smote'):
    """
    Evaluate the given model using StratifiedKFold cross-validation.
    Returns:
      - mean accuracy
      - mean macro F1 score
      - mean macro precision
      - mean macro recall
      - averaged confusion matrix over folds
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    accs, f1s, precs, recs = [], [], [], []
    confusion_mats = []
    
    # Create resampling strategy based on method parameter
    if resampling_method == 'smote':
        resampler = SMOTE(random_state=42, sampling_strategy='auto')
    elif resampling_method == 'adasyn':
        resampler = ADASYN(random_state=42, sampling_strategy='auto', n_neighbors=5)
    elif resampling_method == 'smotetomek':
        resampler = SMOTETomek(random_state=42, sampling_strategy='auto')
    elif resampling_method == 'none':
        resampler = None  # No resampling
    else:  # Default to SMOTE
        resampler = SMOTE(random_state=42, sampling_strategy='auto')
    
    # Show progress bar for cross-validation
    with tqdm(total=folds, desc=f"Cross-validation ({model.__class__.__name__})", file=sys.stdout) as pbar:
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Standardize features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            # Oversample training set to balance classes (if resampler is specified)
            if resampler is not None:
                try:
                    X_train, y_train = resampler.fit_resample(X_train, y_train)
                    logger.info(f"Applied {resampling_method} resampling, new shape: {X_train.shape}")
                except ValueError as e:
                    logger.warning(f"Resampling with {resampling_method} failed: {str(e)}")
                    logger.warning(f"Proceeding without resampling for fold {fold}")
            
            # Log progress
            logger.info(f"Fitting fold {fold}/{folds} for {model.__class__.__name__}")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            elapsed = time.time() - start_time
            logger.info(f"Fold {fold} fitting completed in {elapsed:.1f} seconds")
            
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            confusion_mats.append(cm)
            accs.append(acc)
            f1s.append(f1)
            precs.append(prec)
            recs.append(rec)
            
            logger.info(f"Fold {fold} Accuracy: {acc:.3f}")
            logger.info(f"Fold {fold} Macro F1: {f1:.3f}")
            logger.info(f"Fold {fold} Macro Precision: {prec:.3f}")
            logger.info(f"Fold {fold} Macro Recall: {rec:.3f}")
            logger.info(f"Fold {fold} Classification Report:\n"
                        f"{classification_report(y_test, y_pred, zero_division=0)}")
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({'acc': f"{acc:.3f}", 'f1': f"{f1:.3f}"})

    avg_cm = np.mean(np.array(confusion_mats), axis=0)
    return np.mean(accs), np.mean(f1s), np.mean(precs), np.mean(recs), avg_cm

def find_best_resampling_method(X, y, model):
    """
    Find the best resampling method for a given model.
    """
    methods = ['smote', 'adasyn', 'smotetomek', 'none']  # Added 'none' as an option
    best_method = None
    best_f1 = 0
    
    # Show progress for resampling method evaluation
    with tqdm(total=len(methods), desc=f"Testing resampling methods ({model.__class__.__name__})", file=sys.stdout) as pbar:
        for method in methods:
            logger.info(f"Testing resampling method: {method}")
            start_time = time.time()
            
            try:
                acc, f1, prec, rec, _ = evaluate_model(model, X, y, folds=3, resampling_method=method)
                elapsed = time.time() - start_time
                logger.info(f"{method} evaluation completed in {elapsed:.1f} seconds")
                logger.info(f"{method} => Accuracy: {acc:.3f}, F1: {f1:.3f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_method = method
            except Exception as e:
                logger.error(f"Error evaluating {method}: {str(e)}")
                logger.error(f"Skipping {method} due to error")
                
            pbar.update(1)
            pbar.set_postfix({'method': method, 'f1': f"{best_f1:.3f}"})
    
    # If all methods failed, default to 'none'
    if best_method is None:
        logger.warning("All resampling methods failed, using 'none' as fallback")
        best_method = 'none'
    
    logger.info(f"Best resampling method: {best_method} with F1: {best_f1:.3f}")
    return best_method

def train_and_evaluate_models(X, y):
    """
    4-class problem:
      - 0 => <=7
      - 1 => 8
      - 2 => 9
      - 3 => 10
    Uses class_weight='balanced' and hyperparameter tuning to improve performance.
    """
    # Show overall progress
    logger.info("Starting model training and evaluation process")
    overall_start = time.time()
    
    # Track liblinear parameters globally
    global best_liblinear_params, liblinear_best_score
    
    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create feature selector
    selector = SelectKBest(f_classif, k='all')
    X_selected = selector.fit_transform(X_scaled, y)
    
    # Get feature importance scores
    if X_selected.shape[1] < X.shape[1]:
        scores = selector.scores_
        feature_indices = selector.get_support(indices=True)
        logger.info(f"Selected {len(feature_indices)} features out of {X.shape[1]}")
        logger.info(f"Top 5 feature scores: {sorted(scores, reverse=True)[:5]}")
    
    # Use the selected features for model training
    X_processed = X_selected
    
    # Enhanced hyperparameter grids:
    param_grid_lr = {
        "C": [0.001],
        "solver": ["lbfgs"],
        "multi_class": ["ovr"]
    }

    # Best LogisticRegression parameters:
    # 2025-04-04 23:24:15,836 - INFO -   C: 0.001
    # 2025-04-04 23:24:15,836 - INFO -   solver: lbfgs
    # 2025-04-04 23:24:15,837 - INFO -   multi_class: ovr
    # 2025-04-04 23:24:15,837 - INFO - Best score during tuning: 0.000
    
    # Alternative grid for liblinear solver (which only supports 'ovr')
    # param_grid_lr_liblinear = {
    #     "C": [0.001, 0.01, 0.1, 0.5, 1, 2, 3, 5, 10],
    #     "solver": ["liblinear"],
    #     "multi_class": ["ovr"]  # liblinear only supports ovr
    # }

    param_grid_lr_liblinear = {
        "C": [0.001],
        "solver": ["liblinear"],
        "multi_class": ["ovr"]  # liblinear only supports ovr
    }
    
    # param_grid_rf = {
    #     "n_estimators": [375, 400, 425, 450, 500, 550, 600],
    #     "max_depth": [None, 10, 15, 20, 30],
    #     "min_samples_split": [2, 5, 10],
    #     "min_samples_leaf": [1, 2, 4],
    #     "max_features": [None, "sqrt", "log2"],
    #     "criterion": ["gini", "entropy"]
    # }

    param_grid_rf = {
        "n_estimators": [600],
        "max_depth": [10],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_features": [None],
        "criterion": ["entropy"]
    }
    
    # param_grid_svm = {
    #     "C": [0.1, 0.5, 1, 2, 5, 10, 100],
    #     "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 0.5, 1, 2, 5],
    #     "kernel": ["rbf", "poly", "sigmoid"]
    # }

    param_grid_svm = {
        "C": [10],
        "gamma": [1],
        "kernel": ["rbf"]
    }
    
    # Initialize models with balanced class weights
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    svm = SVC(class_weight="balanced", random_state=42, probability=True)
    
    # Show progress bar for overall model training
    models_to_train = ['Logistic Regression', 'Random Forest', 'SVM']
    with tqdm(total=len(models_to_train), desc="Model training progress", file=sys.stdout) as pbar:
        # Train models with progress tracking
        logger.info("Tuning Logistic Regression...")
        # First try with solvers that support multinomial
        best_lr = tune_model(lr, param_grid_lr, X_processed, y)
        best_lr_params = best_lr.get_params()
        lr_best_score = getattr(best_lr, 'best_score_', 0)
        
        # Then try with liblinear solver
        lr_liblinear = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
        best_lr_liblinear = tune_model(lr_liblinear, param_grid_lr_liblinear, X_processed, y)
        best_lr_liblinear_params = best_lr_liblinear.get_params()
        lr_liblinear_best_score = getattr(best_lr_liblinear, 'best_score_', 0)
        
        # Store liblinear parameters separately
        liblinear_params = {
            "C": best_lr_liblinear_params["C"],
            "solver": best_lr_liblinear_params["solver"],
            "multi_class": best_lr_liblinear_params["multi_class"]
        }
        
        # Save to global variables for final printing
        best_liblinear_params = liblinear_params
        liblinear_best_score = lr_liblinear_best_score
        
        # Compare and keep the better model
        if hasattr(best_lr_liblinear, 'best_score_') and hasattr(best_lr, 'best_score_'):
            if best_lr_liblinear.best_score_ > best_lr.best_score_:
                best_lr = best_lr_liblinear
                logger.info("Liblinear solver performed better and was selected")
                final_lr_params = best_lr_liblinear_params
                final_lr_score = lr_liblinear_best_score
            else:
                final_lr_params = best_lr_params
                final_lr_score = lr_best_score
        else:
            final_lr_params = best_lr_params
            final_lr_score = lr_best_score
        
        pbar.update(1)
        
        logger.info("Tuning Random Forest...")
        best_rf = tune_model(rf, param_grid_rf, X_processed, y)
        # Store best RF parameters
        rf_params = best_rf.get_params()
        rf_best_score = getattr(best_rf, 'best_score_', 0)
        pbar.update(1)
        
        logger.info("Tuning SVM...")
        best_svm = tune_model(svm, param_grid_svm, X_processed, y)
        # Store best SVM parameters
        svm_params = best_svm.get_params()
        svm_best_score = getattr(best_svm, 'best_score_', 0)
        pbar.update(1)
    
    models = {
        "Logistic Regression": best_lr,
        "Random Forest": best_rf,
        "SVM": best_svm
    }
    
    # Find best resampling method for each model
    resampling_methods = {}
    for name, model in models.items():
        logger.info(f"Finding best resampling method for {name}")
        resampling_methods[name] = find_best_resampling_method(X_processed, y, model)
    
    results = {}
    # Show progress for final evaluation
    with tqdm(total=len(models), desc="Final model evaluation", file=sys.stdout) as pbar:
        for name, model in models.items():
            logger.info(f"Evaluating model: {name} with {resampling_methods[name]} resampling")
            acc, f1, prec, rec, cm = evaluate_model(model, X_processed, y, 
                                                   folds=5, 
                                                   resampling_method=resampling_methods[name])
            if name == "Logistic Regression":
                best_params = {
                    "C": final_lr_params["C"],
                    "solver": final_lr_params["solver"],
                    "multi_class": final_lr_params["multi_class"]
                }
                results[name] = {"mean_accuracy": acc, "mean_f1": f1,
                                "mean_precision": prec, "mean_recall": rec,
                                "confusion_matrix": cm, "best_params": best_params,
                                "best_score": final_lr_score}
            elif name == "Random Forest":
                best_params = {
                    "n_estimators": rf_params["n_estimators"],
                    "max_depth": rf_params["max_depth"],
                    "min_samples_split": rf_params["min_samples_split"],
                    "min_samples_leaf": rf_params["min_samples_leaf"],
                    "max_features": rf_params["max_features"],
                    "criterion": rf_params["criterion"]
                }
                results[name] = {"mean_accuracy": acc, "mean_f1": f1,
                                "mean_precision": prec, "mean_recall": rec,
                                "confusion_matrix": cm, "best_params": best_params,
                                "best_score": rf_best_score}
            elif name == "SVM":
                best_params = {
                    "C": svm_params["C"],
                    "gamma": svm_params["gamma"],
                    "kernel": svm_params["kernel"]
                }
                results[name] = {"mean_accuracy": acc, "mean_f1": f1,
                                "mean_precision": prec, "mean_recall": rec,
                                "confusion_matrix": cm, "best_params": best_params,
                                "best_score": svm_best_score}
            else:
                results[name] = {"mean_accuracy": acc, "mean_f1": f1,
                                "mean_precision": prec, "mean_recall": rec,
                                "confusion_matrix": cm}
            logger.info(f"{name} => Accuracy: {acc:.3f}, Macro F1: {f1:.3f}, "
                        f"Macro Precision: {prec:.3f}, Macro Recall: {rec:.3f}")
            class_names = ["≤7", "8", "9", "10"]
            plot_confusion_matrix(cm, class_names, title=f"{name} Average Confusion Matrix")
            pbar.update(1)
            pbar.set_postfix({'model': name, 'f1': f"{f1:.3f}"})
    
    # Show total elapsed time
    overall_elapsed = time.time() - overall_start
    logger.info(f"Total model training and evaluation completed in {overall_elapsed:.1f} seconds")
    
    return results

def main():
    X, y = load_feature_dataset("data/features/features.npy.npz")
    if X is None or y is None:
        logger.error("Dataset not loaded.")
        return
    
    # Initialize global variables to store liblinear results
    global best_liblinear_params, liblinear_best_score
    best_liblinear_params = None
    liblinear_best_score = 0
    
    results = train_and_evaluate_models(X, y)
    logger.info("Model evaluation completed.")
    for model_name, metrics in results.items():
        logger.info(f"{model_name} => Accuracy: {metrics['mean_accuracy']:.2f}, "
                    f"F1: {metrics['mean_f1']:.2f}, Precision: {metrics['mean_precision']:.2f}, "
                    f"Recall: {metrics['mean_recall']:.2f}")
        
        # Print best parameters for all models
        if "best_params" in metrics:
            logger.info(f"Best {model_name} parameters:")
            for param, value in metrics["best_params"].items():
                logger.info(f"  {param}: {value}")
            logger.info(f"Best score during tuning: {metrics['best_score']:.3f}")
    
    # Add a visual separator
    logger.info("="*50)
    logger.info("FINAL RESULT SUMMARY")
    logger.info("="*50)
    
    # Print liblinear parameters as the absolute final output
    if best_liblinear_params:
        logger.info(f"Best parameters for LogisticRegression with liblinear solver:")
        for param, value in best_liblinear_params.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Best score during liblinear tuning: {liblinear_best_score:.3f}")

if __name__ == "__main__":
    main()
    
    # Absolutely final message to ensure liblinear params are last
    if 'best_liblinear_params' in globals() and best_liblinear_params:
        print("\n" + "="*50)
        print("LIBLINEAR SOLVER BEST PARAMETERS (FINAL OUTPUT)")
        print("="*50)
        for param, value in best_liblinear_params.items():
            print(f"  {param}: {value}")
        print(f"Best score: {liblinear_best_score:.3f}")
        print("="*50)
