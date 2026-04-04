# APDPS - Automated Predictive Data Processing System

## Overview
APDPS (Automated Predictive Data Processing System) is an end-to-end machine learning pipeline that automates data preprocessing, model training, evaluation, and model selection. The system is designed to simplify machine learning workflows and enable users to obtain insights with minimal manual effort.

---

## Objectives
- Automate the complete machine learning pipeline
- Reduce manual effort in preprocessing and model selection
- Automatically detect problem type (classification or regression)
- Train and compare multiple machine learning models
- Provide reliable evaluation metrics for decision-making

---

## Features
- Upload CSV datasets
- Automatic data preprocessing
- Intelligent problem type detection
- Multi-model training (Logistic Regression, Random Forest)
- Performance evaluation (Accuracy, Precision, Recall, F1-score)
- Confusion Matrix and Classification Report
- Interactive user interface using Streamlit

---

## Workflow

1. Upload dataset  
2. Data preprocessing  
3. Problem type detection  
4. Model training  
5. Model evaluation  
6. Best model selection  
7. Results visualization  

---

## Project Structure
APDPS/
│
├── app.py
├── main.py
├── pipeline.py
│
├── core/
│ ├── modeling/
│ ├── profiling/
│ ├── scalability/
│
├── data/
│ └── sample.csv
│
├── experiment_log.json
├── requirements.txt
└── README.md

---

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib  

---

## Results
- Dataset: Retail Customer Segmentation  
- Problem Type: Classification  
- Best Model: Random Forest Classifier  
- Accuracy: Approximately 75%  

---

## How to Run

1. Clone the repository:
   git clone https://github.com/snigdha-ramaraju/APDPS.git


2. Install dependencies:

pip install -r requirements.txt


3. Run the application:

streamlit run app.py


---

## Future Enhancements
- Integration of advanced models (XGBoost, LightGBM)
- Hyperparameter tuning
- Explainable AI integration
- Cloud deployment
- Support for hybrid models (text + tabular data)

---

## Contributors
- Snigdha Ramaraju (Team Lead)
- Lakshmi Priya Addagulla
- Sushmitha Mabbu
- Varsha Voundekoti
