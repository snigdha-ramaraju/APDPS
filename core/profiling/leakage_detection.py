# core/profiling/leakage_detection.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataLeakageDetector:

    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column

    def run(self):

        print("\n========== DATA LEAKAGE ANALYSIS STARTED ==========\n")

        df_copy = self.df.copy()

        print("[LeakageDetector] Encoding categorical columns for correlation...")

        for col in df_copy.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df_copy[col] = le.fit_transform(df_copy[col])

        print("[LeakageDetector] Encoding complete.\n")

        print("[LeakageDetector] Checking for duplicate target columns...")

        duplicate_cols = [
            col for col in df_copy.columns
            if col != self.target_column and df_copy[col].equals(df_copy[self.target_column])
        ]

        if duplicate_cols:
            print("Potential leakage detected in columns:", duplicate_cols)
        else:
            print("No duplicate target columns found.")

        print("\n[LeakageDetector] Checking for high correlation with target...")

        correlations = df_copy.corr()[self.target_column].abs().sort_values(ascending=False)

        high_corr = correlations[correlations > 0.9]
        high_corr = high_corr.drop(self.target_column, errors="ignore")

        if not high_corr.empty:
            print("High correlation detected with target:")
            print(high_corr)
        else:
            print("No strong leakage correlation detected.")

        print("\n========== DATA LEAKAGE ANALYSIS COMPLETED ==========\n")