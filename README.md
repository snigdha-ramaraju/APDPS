APDPS – Autonomous ML Readiness & Data Intelligence Platform
Overview

APDPS (Autonomous ML Readiness & Data Intelligence Platform) is an intelligent machine learning workflow system designed to bridge the gap between raw datasets and machine learning model development.

Real-world datasets are often messy, inconsistent, and difficult to use directly for machine learning. APDPS automates the process of analyzing datasets, identifying potential data issues, preparing the data, training models, and generating clear insights about model performance and generalization.

The platform is built to help researchers, students, analysts, and developers quickly understand whether a dataset is suitable for machine learning and how well models perform on it.

APDPS acts as a Pre-ML Intelligence Layer between raw data and machine learning workflows.

Key Features
1. Intelligent Data Profiling

APDPS automatically analyzes the uploaded dataset and generates insights about its structure.

Features include:

Dataset shape detection

Automatic schema detection

Correlation matrix generation

Correlation heatmap visualization

Identification of highly correlated features

Detection of multicollinearity risks

2. Automated Data Cleaning

Real-world datasets often contain problematic columns. APDPS automatically handles them.

Capabilities:

Automatic identifier detection (CustomerID, RowNumber, etc.)

Removal of high-cardinality categorical columns

Missing target value handling

Missing feature value imputation

Median for numeric features

Mode for categorical features

3. Intelligent Problem Type Detection

The system automatically determines whether the task is:

Binary Classification

Multi-Class Classification

Regression

This removes the need for manual configuration.

4. Automated Model Training

APDPS automatically trains machine learning models after analyzing the dataset.

Supported tasks:

Classification

Regression

Users can also control the train-test split ratio to evaluate model performance under different training conditions.

5. Comprehensive Model Evaluation

The platform provides detailed evaluation metrics depending on the detected problem type.

For Classification:

Test Accuracy

Train Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Classification Report

For Regression:

R² Score

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

These metrics help users understand model performance clearly.

6. Feature Importance Analysis

APDPS identifies which features contribute most to the model's predictions.

Capabilities:

Feature importance ranking

Visualization of top predictive features

Identification of influential variables

This helps users understand which dataset attributes matter most.

7. Model Generalization Intelligence

The system evaluates how well the model performs on unseen data.

It compares:

Training performance

Testing performance

Based on the difference between them, APDPS detects:

Stable models

Moderate overfitting

Severe overfitting

This ensures users understand the reliability of the trained model.

8. Interactive Prediction Interface

APDPS allows users to test the trained model interactively.

Users can manually enter feature values and generate predictions directly within the interface.

For classification tasks, prediction probabilities are also displayed.

9. Model Export

The trained model can be downloaded as a serialized file for future use or deployment.

Supported format:

.pkl (joblib serialized model)

System Workflow

Upload dataset

Automatic data cleaning

Data intelligence analysis

Problem type detection

Model training

Model evaluation

Feature importance analysis

Generalization intelligence

Manual prediction

Model export

Technology Stack

Programming Language
Python

Libraries

Streamlit

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Joblib

Installation

Clone the repository:

git clone https://github.com/yourusername/APDPS.git

Navigate to the project folder:

cd APDPS

Create a virtual environment:

python -m venv apdps_env

Activate environment:

Windows

apdps_env\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py
Use Cases

APDPS can be used by:

Data Science students learning machine learning workflows

Researchers analyzing experimental datasets

Analysts exploring structured business data

Developers building ML pipelines

Hackathon teams developing AI tools

Future Roadmap

Upcoming features planned for APDPS include:

Automated feature selection engine

Data leakage detection

Class imbalance intelligence

ML readiness scoring system

Unsupervised learning support

Text dataset analysis

Hybrid tabular + NLP modeling

Author

Developed as part of an advanced machine learning system design project focused on automated ML readiness analysis and intelligent data processing.
