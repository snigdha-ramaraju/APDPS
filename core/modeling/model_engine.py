# core/modeling/model_engine.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)


class ModelEngine:
    def __init__(self, df, target_column, problem_type):
        self.df = df.copy()
        self.target_column = target_column
        self.problem_type = problem_type
        self.model = None

    def _encode_features(self, X):
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        return X

    def _scale_features(self, X):
        scaler = StandardScaler()
        return scaler.fit_transform(X)

    def train(self):

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        X = self._encode_features(X)
        X = self._scale_features(X)

        results = {}

        # SMALL DATASET
        if len(self.df) < 20:

            if self.problem_type in [
                "Binary Classification",
                "Multi-Class Classification",
            ]:

                self.model = LogisticRegression(max_iter=1000)
                self.model.fit(X, y)
                preds = self.model.predict(X)

                acc = accuracy_score(y, preds)

                results["accuracy"] = acc

            else:

                self.model = RandomForestRegressor(random_state=42)
                self.model.fit(X, y)
                preds = self.model.predict(X)

                results["mse"] = mean_squared_error(y, preds)
                results["r2"] = r2_score(y, preds)

        # NORMAL DATASET
        else:

            if self.problem_type in [
                "Binary Classification",
                "Multi-Class Classification",
            ]:

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

                self.model = LogisticRegression(max_iter=1000)
                self.model.fit(X_train, y_train)
                preds = self.model.predict(X_test)

                acc = accuracy_score(y_test, preds)

                results["accuracy"] = acc

            else:

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                self.model = RandomForestRegressor(random_state=42)
                self.model.fit(X_train, y_train)
                preds = self.model.predict(X_test)

                results["mse"] = mean_squared_error(y_test, preds)
                results["r2"] = r2_score(y_test, preds)

        joblib.dump(self.model, "trained_model.pkl")

        return results