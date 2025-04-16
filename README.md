# Breast Cancer Detection Using Machine Learning

This project focuses on classifying breast cancer using various machine learning models trained on medical image and tabular data. It aims to compare the performance of traditional algorithms and implement best practices in preprocessing, resampling, and evaluation.

## 📁 Dataset

The dataset consists of clinical and image-level annotations for breast cancer. It includes:
- BI-RADS category (converted to binary class)
- Image metadata (view, laterality, dimensions)
- Corresponding image files (grayscale mammograms)

Target variable:
- `class = 0` → BI-RADS 1 (Normal)
- `class = 1` → BI-RADS 2 or above (Abnormal)

## 🛠 Technologies Used

- Python
- NumPy, Pandas
- Scikit-learn
- imbalanced-learn (SMOTE)
- PIL / OpenCV
- PCA
- Matplotlib / Seaborn

## ⚙️ Pipeline Overview

1. **Data Preprocessing**
   - Load and process metadata from CSV
   - Convert BI-RADS to binary classification
   - Split into training and testing sets

2. **Image Processing**
   - Load grayscale mammogram images
   - Resize to 256x256
   - Histogram equalization
   - Normalize pixel values

3. **Dimensionality Reduction**
   - Apply PCA to reduce image dimensionality to 100 components

4. **Handling Class Imbalance**
   - Use SMOTE to oversample minority class in the training set

5. **Model Training & Evaluation**
   Applied and evaluated the following models:
   - **Logistic Regression**
   - **Support Vector Machine (SVM)**
   - **Decision Tree**
   - **Random Forest**
   - **k-Nearest Neighbors (KNN)**

6. **Hyperparameter Optimization**
   - Performed Grid Search CV to tune each model using recall as the scoring metric

7. **Model Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Confusion Matrix visualization

## 🔍 Best Performing Model

| Model              | Best Recall (CV) |
|-------------------|------------------|
| Logistic Regression | 0.5592 |
| SVM                 | 0.4733 |
| Decision Tree       | 0.7444 |
| Random Forest       | 0.8155 |
| KNN                 | **0.9030** ✅ |

## 📊 Example Output

Each model prints:
- Classification metrics
- Best hyperparameters
- Confusion matrix plot

