# 🩺 HFR-MADM Based Healthcare Disease Prediction System
A Streamlit-based healthcare analytics and disease prediction system that automatically evaluates multiple medical datasets using **Hesitant Fuzzy Rough – Multi Attribute Decision Making (HFR-MADM)** and selects the best dataset for **Logistic Regression–based risk prediction**.
## 📌 Project Overview
This project helps in:
- Comparing multiple healthcare datasets
- Ranking datasets based on quality using HFR-MADM
- Automatically selecting the best dataset
- Training a machine learning model for disease prediction
- Providing an interactive **patient risk prediction interface**
## 🚀 Features
-  Supports **multiple healthcare datasets**
-  HFR-MADM dataset quality ranking with uncertainty-aware scoring
-  Class-weight-balanced Logistic Regression model for prediction
-  Interactive charts and metrics
-  Feature importance visualization
-  Confusion matrix, classification report & 5-fold cross-validation
-  Individual patient risk prediction form with dynamic risk labeling
-  Advanced UI with sidebar navigation
## 🗂️ Datasets Used
- `brain_stroke.csv`
- `diabetes.csv`
- `heartdisease.csv`
- `hypertension_dataset.csv`
- `indian_liver_patient.csv`
- `kidney_disease.csv`
- `healthcare-dataset-stroke-data.csv`
##  Methodology
### 1. Dataset Preprocessing
- Remove missing values
- Remove duplicates
- Encode categorical features using `LabelEncoder`
- Separate features (X) and target (y)
### 2. Dataset Ranking (HFR-MADM)
Each dataset is evaluated across 4 criteria — sample size, class balance, feature richness, and data reliability (share of rows retained after cleaning). The "hesitant fuzzy" part: size, balance, and feature scores are each recomputed across multiple bootstrap resamples of the dataset, producing a range of plausible values rather than one fixed number. The final ranking score is the weighted average of these criteria **minus** a penalty for how much that value swings across resamples — so a dataset with a volatile signal ranks below an equally-scored but more stable one.
### 3. Model Training
- Train/Test split (80/20), stratified by class
- Feature scaling using `StandardScaler`
- Class-weight-balanced Logistic Regression classifier, to avoid bias toward the majority class on imbalanced medical data
- Model evaluation using test-split accuracy, 5-fold cross-validation accuracy, F1-score, and confusion matrix — CV accuracy is treated as the more reliable estimate, since a single train/test split can vary with the random seed
### 4. Risk Prediction
- User inputs patient values
- Model predicts risk class based on the dataset's actual minority class (not hardcoded to a fixed 0/1 label), so it correctly adapts across datasets with different label encodings
- Displays prediction confidence
## 🛠️ Technologies Used
- **Python 3**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
## 🌐 Live Demo
Interactive web application deployed using Streamlit Cloud.
👉 https://healthcare-ml-project-gvskcbxdfc37z84spyouwd.streamlit.app/
