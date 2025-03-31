# ESNF544-FinalProject: Pokémon Card Grading Machine Learning Pipeline

This project automates the grading of Pokémon cards using machine learning. It implements a complete pipeline—from image preprocessing and feature extraction to model training, hyperparameter tuning, and evaluation. The goal is to predict card grades (e.g., on a PSA scale) based on discriminative features extracted from card images.


## Dataset

This project uses a dataset created by converting uploaded images of cards and their grades from popular online auctions, processes them using computer vision techniques, and stores the graded card image in the corresonding folder "grade_#".

## Overview

This project follows a structured machine learning pipeline:

1. **Image Preprocessing & Feature Extraction:**  
   - The raw images of Pokémon cards are processed to extract only the relevant card region.
   - Multiple discriminative features are calculated from each image. These include measures of border continuity, corner quality, centering, inner cleanliness, grading index (with bias based on expected grade), edge density, saturation variance, and additional texture or color consistency features.
   - The resulting feature vectors are cleaned (columns with insufficient valid data are removed) and saved as a compressed NumPy archive.

2. **Model Training & Evaluation:**  
   - The saved features are loaded and used to train classifiers on a 4-class problem:
     - Class 0: Cards graded ≤7
     - Class 1: Cards graded 8
     - Class 2: Cards graded 9
     - Class 3: Cards graded 10
   - Models used include Logistic Regression, Random Forest, and SVM (with an RBF kernel). For models sensitive to feature scales (Logistic Regression and SVM), pipelines with scaling are used.
   - Hyperparameter tuning is performed using GridSearchCV, and the training loop uses stratified cross-validation with oversampling to balance classes.
   - Evaluation metrics (accuracy, macro F1, macro precision, and macro recall) are computed for each fold, and an averaged confusion matrix is plotted.


## Requirements and Installation

### Requirements

- Python 3.11 (or later)
- [OpenCV-Python](https://pypi.org/project/opencv-python/)
- [NumPy](https://pypi.org/project/numpy/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [seaborn](https://pypi.org/project/seaborn/)
- [imbalanced-learn](https://pypi.org/project/imbalanced-learn/)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd <repository-folder>
   ```


## Usage: 

** Note some folder paths may need to be updated from the placeholders for input-output paths in the feature_extraction, model_training an dpreprocessing files. **

### Step 1: Feature Extraction

1. **Testing & Visualization:**

   To quickly test feature extraction and visualize the outputs (need to uncomment the test and evaluate line in the main function to visualize the features - which includes the computed feature values, the detected yellow border, Canny edge map, and inner region), run:

   ```bash
   python src/feature_extraction.py
   ```

2. **Saving the Feature Dataset:**

   Once you are satisfied with feature extraction, comment out the testing function call and uncomment the dataset saving function call in the main block of `feature_extraction.py`:

   ```python
   # test_feature_extraction(..., visualize=SHOW_VISUALIZATION)
   save_feature_dataset(input_dir="ENSF 544\\Final-Project\\data\\processed", output_file="your-root-file-path-name\\data\\features\\features.npz")
   ```

   Then run:

   ```bash
   python src/feature_extraction.py
   ```

   This will generate a compressed file `features.npy.npz` in the `data/features/` directory containing your feature matrix and labels.

### Step 2: Model Training & Evaluation

1. **Model Training:**

   With the feature dataset generated, run the model training and evaluation module:

   ```bash
   python src/model_training.py
   ```

   This script will:
   - Load the feature dataset.
   - Perform hyperparameter tuning for Logistic Regression, Random Forest, and SVM (using pipelines for models that require scaling).
   - Evaluate each tuned model using stratified cross-validation with oversampling.
   - Log metrics (accuracy, macro F1, precision, and recall) for each fold and print an averaged confusion matrix plot for each model.

---

## Machine Learning Pipeline Flow

1. **Data Preprocessing & Feature Extraction:**
   - The images are processed to detect the yellow border and extract the inner region.
   - Several features are computed from each image (including features such as Edge Density, Saturation Variance, and Expected Deviation Score).
   - The feature matrix is cleaned by discarding columns that are not valid for at least 30% of the samples.
   - The resulting features (and corresponding class labels) are saved in a compressed file.

2. **Model Training & Evaluation:**
   - The feature dataset is loaded.
   - Each classifier is tuned using GridSearchCV (with appropriate scaling for Linear Regression and SVM).
   - The training is performed with oversampling to balance the classes.
   - Evaluation is done using stratified k-fold cross-validation (5 folds), and performance metrics are logged and plotted.

---

## Our Further Improvements


- **Hyperparameter Tuning:**  
  Experimented with various hyperparamter values across the three chosen models. Provided ranges of values to ensure more variations in finding a best combination.


- **Feature Engineering:**  
  The initail feature set included 4 features, but additional determinitive features were added until performance started to decrease.

- **Ensemble Methods:**  
  Considered using ensemble methods (voting classifiers, stacking) to combine multiple model predictions.

---


