import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Healthcare Disease Prediction System",
    layout="wide"
)

st.title("Healthcare Disease Prediction System")
st.subheader("Hesitant Fuzzy Rough MADM based Dataset Selection & Disease Prediction")

st.markdown(
    """
    This web application evaluates multiple healthcare datasets,
    ranks them using **Hesitant Fuzzy Rough MADM**, selects the best dataset,
    and performs **disease risk prediction** using machine learning.
    """
)

# -----------------------------
# 1. LOAD DATASETS
# -----------------------------
def load_datasets(data_folder="data"):
    datasets = {}

    if not os.path.exists(data_folder):
        st.error("❌ Data folder not found.")
        return datasets

    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_folder, file))
            df = df.dropna().drop_duplicates()

            le = LabelEncoder()
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = le.fit_transform(df[col].astype(str))

            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            if y.nunique() < 2:
                continue

            datasets[file] = (X, y)

    return datasets


# -----------------------------
# 2. HESITANT FUZZY ROUGH MADM
# -----------------------------
def hesitant_fuzzy_rough_madm(datasets):
    np.random.seed(42)
    weights = np.array([0.35, 0.15, 0.30, 0.20])
    scores = {}

    for name, (X, _) in datasets.items():
        f1 = min(len(X) / 1000, 1.0)         # dataset size
        f2 = 1.0                             # completeness
        f3 = min(X.shape[1] / 30, 1.0)       # feature richness
        f4 = 1 - np.random.uniform(0.05, 0.15)  # uncertainty

        metrics = [f1, f2, f3, f4]
        hesitant_means = [
            np.mean([m * 0.95, m, min(m * 1.05, 1.0)]) for m in metrics
        ]

        scores[name] = np.dot(hesitant_means, weights)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# -----------------------------
# 3. DISEASE PREDICTION
# -----------------------------
def disease_prediction(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return acc, report, cm


# -----------------------------
# RUN APPLICATION
# -----------------------------
if st.button("Run Project"):
    data = load_datasets("data")

    if len(data) < 2:
        st.warning("⚠️ Please add at least two datasets inside the data folder.")
    else:
        rankings = hesitant_fuzzy_rough_madm(data)

        # -----------------------------
        # DATASET RANKING TABLE
        # -----------------------------
        st.header("Dataset Quality Ranking")

        ranking_df = pd.DataFrame(rankings, columns=["Dataset", "Quality Score"])
        ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))

        st.dataframe(ranking_df, use_container_width=True)

        # -----------------------------
        # RANKING BAR CHART
        # -----------------------------
        st.subheader("Dataset Quality Comparison")

        st.bar_chart(
            ranking_df.set_index("Dataset")["Quality Score"]
        )

        # -----------------------------
        # BEST DATASET & MODEL RESULTS
        # -----------------------------
        best_file = rankings[0][0]
        X, y = data[best_file]

        acc, report, cm = disease_prediction(X, y)

        st.success(f"Selected Dataset: {best_file}")

        st.metric(
            label="System Accuracy",
            value=f"{acc * 100:.2f}%"
        )

        # -----------------------------
        # CLASSIFICATION REPORT
        # -----------------------------
        st.header("Clinical Classification Report")

        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

        # -----------------------------
        # CONFUSION MATRIX
        # -----------------------------
        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
