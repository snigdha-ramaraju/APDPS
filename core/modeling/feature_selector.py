# core/modeling/feature_selector.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureSelector:

    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column

    def run(self):

        print("\n========== FEATURE SELECTION STARTED ==========\n")

        df_copy = self.df.copy()

        for col in df_copy.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col])

        correlations = df_copy.corr()[self.target_column].abs().sort_values(ascending=False)

        print("========== FEATURE IMPORTANCE RANKING ==========\n")
        print(correlations.drop(self.target_column, errors="ignore"))

        print("\n========== FEATURE SELECTION COMPLETED ==========\n")