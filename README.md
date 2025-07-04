# autism-prediction-model


 Project Overview
AutiScan is a machine learning-based application developed to predict the likelihood of Autism Spectrum Disorder (ASD) in individuals. It utilizes a dataset of behavioral and demographic features and applies preprocessing and classification techniques to build a reliable prediction system.

 Technologies Used
- Python
- Scikit-learn
- pandas
- NumPy
- Streamlit

Objectives
- Predict the likelihood of autism in individuals based on behavioral and demographic data.
- Improve model performance using preprocessing and resampling techniques.
- Evaluate the model with standard classification metrics.
- Deploy an interactive Streamlit interface for real-time predictions.

 Dataset
- Source: Kaggle – Autism Screening for Adults Dataset
- Records: 1,000+ entries
- Features include: Age, Gender, Ethnicity, Family relation to ASD.
 Features
- Trained a Random Forest classifier achieving 93% cross-validation accuracy and 83% test accuracy.
- Performed data preprocessing including:
  - Label encoding for categorical variables
  - SMOTE for class balancing
  - Outlier detection and treatment
- Model evaluated using:
  - Confusion matrix
  - Classification report
  - Weighted F1-score: 0.83
  - Macro F1-score: 0.76

Preprocessing Steps
- Data cleaning and handling missing values
- Label encoding for categorical features
- SMOTE to balance class distribution
- Outlier treatment
- Feature selection 
