# Diabetes Prediction App

An interactive web application to predict the likelihood of diabetes based on medical parameters using machine learning models.

## Overview

The **Diabetes Prediction App** allows users to input medical parameters (such as glucose level, BMI, and insulin level) and receive a prediction of whether they are likely to be diabetic. The app is built using machine learning models and provides a user-friendly interface with real-time visualizations.

## Features

- User-friendly sliders for entering medical data.
- Predicts the likelihood of diabetes using trained machine learning models (Logistic Regression and Random Forest).
- Displays results as "Diabetic" or "Non-Diabetic" with a probability score.
- Visualizes predictions with a probability pie chart.
- Option to download prediction results as a CSV file.
- Displays model performance metrics such as accuracy and ROC-AUC score.

## Technologies Used

- **Programming Language**: Python
- **Framework**: Streamlit
- **Libraries**:
  - Scikit-learn (for model training and evaluation)
  - Pandas and NumPy (for data manipulation)
  - Matplotlib (for visualizations)
  - Joblib (for model serialization)

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd diabetes_prediction
