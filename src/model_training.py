import os
import numpy as np
import logging
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


# Oversampling to handle class imbalance
from imblearn.over_sampling import RandomOverSampler

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_feature_dataset(feature_file="ENSF 544/Final-Project/data/features/features.npy.npz"):
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
    Tune model hyperparameters using GridSearchCV.
    """
    grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    grid.fit(X, y)
    logger.info(f"Best params for {model.__class__.__name__}: {grid.best_params_}")
    logger.info(f"Best {scoring} score: {grid.best_score_:.3f}")
    return grid.best_estimator_

def evaluate_model(model, X, y, folds=5):
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
    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Oversample training set to balance classes
        oversampler = RandomOverSampler(random_state=42)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        
        model.fit(X_train, y_train)
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
        fold += 1

    avg_cm = np.mean(np.array(confusion_mats), axis=0)
    return np.mean(accs), np.mean(f1s), np.mean(precs), np.mean(recs), avg_cm

def train_and_evaluate_models(X, y):
    """
    4-class problem:
      - 0 => <=7
      - 1 => 8
      - 2 => 9
      - 3 => 10
    Uses class_weight='balanced' and hyperparameter tuning to improve performance.
    """
    # Expanded hyperparameter grids:
    param_grid_lr = {"C": [0.001, 0.01, 0.1, 1, 2, 3]} #says 1 is best
    param_grid_rf = {
        "n_estimators": [375, 380, 390,  400, 425, 450, 500, 525, 550, 600, 625], #best rn is 500, then 450, then 400
        "max_depth": [None], #best as none
        "min_samples_split": [2], #best is 2
        "max_features": [None] #best was none
    }
    param_grid_svm = {
        "C": [0.1, 0.5, 1, 1.5, 5, 10, 100, 1000], #best is 1
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5] # best is 2, then 1.5
    }
    
    # Initialize models with balanced class weights
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    rf = RandomForestClassifier(random_state=42, class_weight="balanced")
    svm = SVC(kernel='rbf', class_weight="balanced")
    
    logger.info("Tuning Logistic Regression...")
    best_lr = tune_model(lr, param_grid_lr, X, y)
    
    logger.info("Tuning Random Forest...")
    best_rf = tune_model(rf, param_grid_rf, X, y)
    
    logger.info("Tuning SVM (RBF kernel)...")
    best_svm = tune_model(svm, param_grid_svm, X, y)
    
    models = {
        "Logistic Regression": best_lr,
        "Random Forest": best_rf,
        "SVM (RBF kernel)": best_svm
    }
    
    results = {}
    for name, model in models.items():
        logger.info(f"Evaluating model: {name}")
        acc, f1, prec, rec, cm = evaluate_model(model, X, y, folds=5)
        results[name] = {"mean_accuracy": acc, "mean_f1": f1,
                         "mean_precision": prec, "mean_recall": rec,
                         "confusion_matrix": cm}
        logger.info(f"{name} => Accuracy: {acc:.3f}, Macro F1: {f1:.3f}, "
                    f"Macro Precision: {prec:.3f}, Macro Recall: {rec:.3f}")
        class_names = ["≤7", "8", "9", "10"]
        plot_confusion_matrix(cm, class_names, title=f"{name} Average Confusion Matrix")
    
    return results

def main():
    X, y = load_feature_dataset("ENSF 544/Final-Project/data/features/features.npy.npz")
    if X is None or y is None:
        logger.error("Dataset not loaded.")
        return
    
    results = train_and_evaluate_models(X, y)
    logger.info("Model evaluation completed.")
    for model_name, metrics in results.items():
        logger.info(f"{model_name} => Accuracy: {metrics['mean_accuracy']:.2f}, "
                    f"F1: {metrics['mean_f1']:.2f}, Precision: {metrics['mean_precision']:.2f}, "
                    f"Recall: {metrics['mean_recall']:.2f}")

if __name__ == "__main__":
    main()
