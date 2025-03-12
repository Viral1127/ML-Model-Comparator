import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from utils import load_csv, plot_correlation_matrix
from model_evaluator import train_and_evaluate_models, get_feature_importance
from fpdf import FPDF

st.set_page_config(
    page_title="ML Model Comparator",  # Change the tab name
    page_icon="📊",  # Change the favicon (You can use an emoji or a custom icon)
)
# Header
st.markdown("""
    <h1 style='text-align: center;'>📊 ML Model Comparator</h1>
    <hr>
""", unsafe_allow_html=True)

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load Data
    df = load_csv(uploaded_file)

    target_column = st.selectbox("Select the Target Column", df.columns)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Create Tabs
    tab1, tab2 = st.tabs(["📊 Overview", "⚖️ Comparison"])

    with tab1:
        st.write("### Dataset Preview")
        st.write(df.head())

        # Show Dataset Info
        st.write("### Dataset Summary")
        st.write(df.describe())

        # Missing Values
        st.write("### 🚨 Missing Values")
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
        if missing_values.empty:
            st.success("No missing values in the dataset! ✅")
        else:
            st.warning("Dataset contains missing values!")
            st.table(pd.DataFrame({"Column": missing_values.index, "Missing Count": missing_values.values}))

        # Data Distribution
        st.write("### 📊 Data Distribution")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            fig, axes = plt.subplots(nrows=1, ncols=len(num_cols), figsize=(15, 5))
            if len(num_cols) == 1:
                sns.histplot(df[num_cols[0]], ax=axes)
            else:
                for col, ax in zip(num_cols, axes):
                    sns.histplot(df[col], ax=ax, kde=True)
                    ax.set_title(col)
            st.pyplot(fig)
        else:
            st.info("No numerical columns available for distribution plots.")

        # Correlation Heatmap
        st.write("### 🔥 Correlation Heatmap")
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Not enough numerical features for a correlation heatmap.")

    with tab2:
        # Model Selection
        st.write("### 🤖 Select Models to Train")
        available_models = ["Decision Tree", "Random Forest", "SVM", "Logistic Regression", "KNN"]
        selected_models = st.multiselect("Choose ML Models:", available_models, default=available_models)

        if st.button("Train Selected Models"):
            if not selected_models:
                st.warning("Please select at least one model!")
            else:
                with st.spinner("Training selected models..."):
                    start_time = time.time()
                    accuracies, best_model, training_times, metrics = train_and_evaluate_models(df, target_column, selected_models)
                    end_time = time.time()

                # Display Best Model
                st.markdown(f"""
                    <h2 style='text-align: center; color: green;'>✅ Best Model: <strong>{best_model}</strong></h2>
                    <hr>
                """, unsafe_allow_html=True)

                # Accuracy Table
                accuracy_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
                st.write("### 📊 Model Accuracy Table")
                st.table(accuracy_df)

                # Bar Chart for Accuracy
                st.write("### 📈 Model Accuracy Comparison")
                st.bar_chart(pd.DataFrame({"Accuracy": accuracies}))

                # Training Time Table
                time_df = pd.DataFrame(list(training_times.items()), columns=["Model", "Training Time (s)"])
                st.write("### ⏳ Model Training Time")
                st.table(time_df)

                # Performance Metrics Table (Precision, Recall, F1-Score)
                metrics_df = pd.DataFrame({
                    "Model": list(metrics.keys()),
                    "Precision": [metrics[m]["Precision"] for m in metrics],
                    "Recall": [metrics[m]["Recall"] for m in metrics],
                    "F1-Score": [metrics[m]["F1-Score"] for m in metrics],
                })
                st.write("### 🔍 Performance Metrics")
                st.table(metrics_df)

                # Feature Importance
                if "Random Forest" in selected_models:
                    st.write("### 🔥 Feature Importance (Random Forest)")
                    feature_importance = get_feature_importance(df, target_column)
                    feature_df = pd.DataFrame({"Feature": feature_importance.keys(), "Importance": feature_importance.values()})
                    feature_df = feature_df.set_index("Feature")
                    st.bar_chart(feature_df)
                
                # Download CSV
                st.write("### 📥 Download Results")
                csv = accuracy_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV 📂",
                    data=csv,
                    file_name="model_results.csv",
                    mime="text/csv",
                )

                # Generate and Download PDF
                def generate_pdf():
                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.add_page()

                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(200, 10, "Comparative ML Study - Report", ln=True, align="C")

                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(200, 10, f"Best Model: {best_model}", ln=True, align="C")

                    pdf.set_font("Arial", "B", 10)
                    pdf.cell(200, 10, "Model Accuracy:", ln=True)
                    for model, acc in accuracies.items():
                        pdf.cell(200, 10, f"{model}: {acc:.4f}", ln=True)

                    pdf.cell(200, 10, "Training Time (seconds):", ln=True)
                    for model, time in training_times.items():
                        pdf.cell(200, 10, f"{model}: {time:.2f}s", ln=True)

                    pdf.cell(200, 10, "Performance Metrics:", ln=True)
                    for model, metric in metrics.items():
                        pdf.cell(200, 10, f"{model}: Precision={metric['Precision']:.4f}, Recall={metric['Recall']:.4f}, F1-Score={metric['F1-Score']:.4f}", ln=True)

                    pdf.cell(200, 10, "Feature Importance (Random Forest):", ln=True)
                    for feature, importance in feature_importance.items():
                        pdf.cell(200, 10, f"{feature}: {importance:.4f}", ln=True)

                    return pdf

                pdf_report = generate_pdf()
                pdf_bytes = pdf_report.output(dest="S").encode("latin1")
                st.download_button(
                    label="Download PDF 📄",
                    data=pdf_bytes,
                    file_name="ML_Comparison_Report.pdf",
                    mime="application/pdf",
                )

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center;'>
        Developed by <a href="https://www.linkedin.com/in/viral-chauhan-73b67024b/?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">
        Viral Chauhan</a>  | Powered by Streamlit
    </p>
""", unsafe_allow_html=True)

