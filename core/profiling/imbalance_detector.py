# core/profiling/imbalance_detector.py


class ClassImbalanceDetector:

    def __init__(self, df, target_column):
        self.df = df
        self.target_column = target_column

    def run(self):

        print("\n========== CLASS IMBALANCE ANALYSIS STARTED ==========\n")

        value_counts = self.df[self.target_column].value_counts()
        print("[ImbalanceDetector] Class Distribution:")
        print(value_counts)

        if len(value_counts) > 1:
            ratio = value_counts.min() / value_counts.max()
            print(f"\n[ImbalanceDetector] Imbalance Ratio: {ratio:.3f}")

            if ratio < 0.5:
                print("[ImbalanceDetector] Warning: Dataset appears imbalanced.")
            else:
                print("[ImbalanceDetector] Class distribution appears balanced.")

        print("\n========== CLASS IMBALANCE ANALYSIS COMPLETED ==========\n")