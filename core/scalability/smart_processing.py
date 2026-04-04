# core/scalability/smart_processing.py

import pandas as pd
import os


class SmartProcessingLayer:

    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):

        print("[SmartProcessingLayer] Loading dataset...")

        # File size detection
        file_size = os.path.getsize(self.file_path) / (1024 * 1024)
        print(f"[SmartProcessingLayer] File Size: {file_size:.2f} MB")

        # Load dataset
        df = pd.read_csv(self.file_path)

        print("[SmartProcessingLayer] Dataset loaded normally.")
        print(f"[SmartProcessingLayer] Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

        # Categorize dataset
        if df.shape[0] < 1000:
            category = "Small"
            mode = "Full Processing"
        elif df.shape[0] < 100000:
            category = "Medium"
            mode = "Optimized Processing"
        else:
            category = "Large"
            mode = "Chunk Processing"

        print(f"[SmartProcessingLayer] Category: {category}")
        print(f"[SmartProcessingLayer] Processing Mode: {mode}")

        print("[SmartProcessingLayer] Optimizing memory...")

        # Basic memory optimization
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        print("[SmartProcessingLayer] Memory optimization complete.")

        return df