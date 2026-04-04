from core.scalability.smart_processing import SmartProcessingLayer
from core.profiling.data_profiler import IntelligentDataProfiler
from core.profiling.leakage_detection import DataLeakageDetector
from core.profiling.imbalance_detector import ClassImbalanceDetector
from core.profiling.target_analyzer import TargetAnalyzer
from core.modeling.feature_selector import FeatureSelector
from core.modeling.model_engine import ModelEngine


def main():

    file_path = "data/sample.csv"

    # 1. Smart Processing
    smart_layer = SmartProcessingLayer(file_path)
    df = smart_layer.load_dataset()

    if df is None:
        print("Dataset loading failed.")
        return

    smart_layer.categorize_dataset()
    smart_layer.optimize_memory()

    df_for_analysis = smart_layer.apply_sampling_if_needed()

    # 2. Data Profiling
    profiler = IntelligentDataProfiler(df_for_analysis)
    profiler.profile()

    # 3. Leakage Detection
    leakage_detector = DataLeakageDetector(
        df_for_analysis,
        target_column="target"
    )
    leakage_detector.analyze()

    # 4. Class Imbalance Detection
    imbalance_detector = ClassImbalanceDetector(
        df_for_analysis,
        target_column="target"
    )
    imbalance_detector.analyze_imbalance()

    # 5. Target Intelligence
    target_analyzer = TargetAnalyzer(
        df_for_analysis,
        target_column="target"
    )
    problem_type = target_analyzer.analyze()

    if problem_type is None:
        problem_type = "Binary Classification"

    # 6. Feature Selection
    feature_selector = FeatureSelector(
        df_for_analysis,
        target_column="target"
    )
    feature_selector.analyze()

    # 7. Model Training
    model_engine = ModelEngine(
        df_for_analysis,
        target_column="target",
        problem_type=problem_type
    )
    model_engine.train()


if __name__ == "__main__":
    main()