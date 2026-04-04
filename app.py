import streamlit as st
import pandas as pd
from pipeline import APDPSPipeline

st.set_page_config(page_title="APDPS", layout="wide")
st.title("APDPS – Automated Predictive Data Processing System")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded Successfully")
    st.write(f"Dataset Shape: {df.shape}")

    target_column = st.selectbox("Select Target Column", df.columns)

    if target_column:

        pipeline = APDPSPipeline(df, target_column)

        problem = pipeline.detect_problem()

        st.subheader("Detected Problem Type")
        st.write(problem)

        if st.button("Run Full Pipeline"):

            results, model_comparison, preprocessing_report = pipeline.train()

            st.header("Model Results")

            st.write("Best Model:", results["best_model"])

            if results["problem_type"] == "classification":

                st.write("Accuracy:", results["accuracy"])

                st.subheader("Classification Report")
                st.dataframe(pd.DataFrame(results["classification_report"]).transpose())

                st.subheader("Confusion Matrix")
                st.write(results["confusion_matrix"])

            else:

                st.write("R2 Score:", results["r2_score"])
                st.write("MSE:", results["mse"])
                st.write("MAE:", results["mae"])

            st.header("Model Comparison")
            st.write(model_comparison)

            st.success(preprocessing_report)