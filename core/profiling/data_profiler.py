# core/profiling/data_profiler.py

class IntelligentDataProfiler:

    def __init__(self, df):
        self.df = df

    def run(self):

        print("\n========== DATA PROFILING STARTED ==========\n")

        print(f"Dataset Shape: {self.df.shape}\n")

        print("[Profiling] Detecting schema...")
        schema = {}

        for col in self.df.columns:
            if self.df[col].dtype == "object":
                schema[col] = "categorical"
            else:
                schema[col] = "numeric"

        print("[Profiling] Schema detected.\n")
        print("[Profiling] Schema Summary:")

        for col, dtype in schema.items():
            print(f"{col} → {dtype}")

        print("\n[Profiling] Missing Value Analysis:")
        print(self.df.isnull().sum())

        print("\n[Profiling] Duplicate Rows:", self.df.duplicated().sum())

        print("\n========== DATA PROFILING COMPLETED ==========\n")