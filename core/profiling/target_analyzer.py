# core/profiling/target_analyzer.py


class TargetAnalyzer:

    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column

    def run(self):

        print("\n========== TARGET ANALYSIS STARTED ==========\n")

        unique_values = self.df[self.target_column].nunique()

        if unique_values == 2:
            problem_type = "Binary Classification"
        elif unique_values > 2 and unique_values < 20:
            problem_type = "Multi-Class Classification"
        else:
            problem_type = "Regression"

        print(f"[TargetAnalyzer] Detected Problem Type: {problem_type}")
        print("\n[TargetAnalyzer] Target Distribution:")
        print(self.df[self.target_column].value_counts())

        print("\n========== TARGET ANALYSIS COMPLETED ==========\n")

        return problem_type