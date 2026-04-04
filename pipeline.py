import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


class APDPSPipeline:

    def __init__(self, df, target_column, test_size=0.2):
        self.df = df.copy()
        self.target_column = target_column
        self.test_size = test_size

    # -------------------------------
    # Problem Detection
    # -------------------------------
    def detect_problem(self):
        target = self.df[self.target_column]

        if target.dtype == "object":
            return "classification"

        if pd.api.types.is_numeric_dtype(target):
            if target.nunique() <= 20:
                return "classification"
            return "regression"

        return "unknown"

    # -------------------------------
    # Preprocessing
    # -------------------------------
    def preprocess(self):

        # Remove NaN target
        self.df = self.df[self.df[self.target_column].notna()]

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        # Fill missing values
        num_cols = X.select_dtypes(include=np.number).columns
        for col in num_cols:
            X[col] = X[col].fillna(X[col].median())

        cat_cols = X.select_dtypes(exclude=np.number).columns
        for col in cat_cols:
            X[col] = X[col].fillna(X[col].mode()[0])

        # Encoding
        X = pd.get_dummies(X, drop_first=True)

        return X, y

    # -------------------------------
    # Training
    # -------------------------------
    def train(self):

        X, y = self.preprocess()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        problem_type = self.detect_problem()

        results = {}
        model_comparison = {}

        # ---------------------------
        # Classification
        # ---------------------------
        if problem_type == "classification":

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest Classifier": RandomForestClassifier()
            }

            for name, model in models.items():

                cv_scores = cross_val_score(model, X_train, y_train, cv=5)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)

                model_comparison[name] = {
                    "CV Score": np.mean(cv_scores),
                    "Test Accuracy": acc
                }

            best_model_name = max(
                model_comparison,
                key=lambda x: model_comparison[x]["Test Accuracy"]
            )

            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            results = {
                "problem_type": "classification",
                "best_model": best_model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "classification_report": classification_report(y_test, y_pred, output_dict=True),
                "confusion_matrix": confusion_matrix(y_test, y_pred)
            }

        # ---------------------------
        # Regression
        # ---------------------------
        elif problem_type == "regression":

            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results = {
                "problem_type": "regression",
                "best_model": "Random Forest Regressor",
                "r2_score": r2_score(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred)
            }

            model_comparison["Random Forest"] = results

        else:
            raise ValueError("Unsupported problem type")

        return results, model_comparison, "Preprocessing Completed Successfully"