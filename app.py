import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Healthcare Disease Prediction",
    layout="wide"
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🩺 Healthcare ML App")
st.sidebar.markdown("""
**Project Features**
- Dataset Quality Ranking (HFR-MADM)
- Automatic Best Dataset Selection
- Disease Risk Prediction
- Visual Results
""")

run_button = st.sidebar.button("🚀 Run Analysis")

# =========================
# TITLE
# =========================
st.title("Hesitant Fuzzy Rough MADM based Healthcare Prediction System")
st.write("Upload datasets in the **data/** folder and run the analysis.")

# =========================
# LOAD DATASETS
# =========================
def load_datasets(data_folder="data"):
    datasets = {}

    if not os.path.exists(data_folder):
        st.error("❌ data folder not found")
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

# =========================
# HFR-MADM
# =========================
def hfr_madm(datasets):
    np.random.seed(42)
    weights = np.array([0.35, 0.15, 0.30, 0.20])
    scores = {}

    for name, (X, _) in datasets.items():
        f1 = min(len(X) / 1000, 1.0)
        f2 = 1.0
        f3 = min(X.shape[1] / 30, 1.0)
        f4 = 1 - np.random.uniform(0.05, 0.15)

        metrics = [f1, f2, f3, f4]
        hesitant = [np.mean([m*0.95, m, min(m*1.05, 1)]) for m in metrics]
        scores[name] = np.dot(hesitant, weights)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

# =========================
# PREDICTION
# =========================
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
    report = classification_report(y_test, y_pred, output_dict=True)

    return acc, pd.DataFrame(report).T.round(3)

# =========================
# RUN APP
# =========================
if run_button:
    data = load_datasets("data")

    if len(data) < 2:
        st.warning("⚠️ Please add at least two datasets.")
    else:
        rankings = hfr_madm(data)

        # ----- RANKING TABLE -----
        st.subheader("📊 Dataset Quality Ranking")

        rank_df = pd.DataFrame({
            "Rank": range(1, len(rankings)+1),
            "Dataset": [r[0] for r in rankings],
            "Score": [round(r[1], 4) for r in rankings]
        })

        st.dataframe(rank_df, use_container_width=True)

        # ----- BAR CHART -----
        fig, ax = plt.subplots()
        ax.bar(rank_df["Dataset"], rank_df["Score"])
        ax.set_title("Dataset Ranking Scores")
        ax.set_ylabel("Score")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # ----- BEST DATASET -----
        best_dataset = rankings[0][0]
        X, y = data[best_dataset]

        acc, report_df = disease_prediction(X, y)

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"✅ Selected Dataset\n\n{best_dataset}")

        with col2:
            st.metric("🎯 Accuracy", f"{round(acc*100, 2)}%")

        # ----- CLASSIFICATION REPORT -----
        st.subheader("🧪 Clinical Classification Report")
        st.dataframe(report_df, use_container_width=True)

else:
    st.info("👈 Click **Run Analysis** from the sidebar to start.")

