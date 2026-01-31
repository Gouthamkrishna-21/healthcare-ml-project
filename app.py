import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Intelligent Healthcare Dataset Selection",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Intelligent Healthcare Dataset Selection & Disease Risk Prediction")

# ==========================================
# DATA LOADING
# ==========================================
def load_datasets(folder="data"):
    datasets = {}
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file)).dropna()
            le = LabelEncoder()
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = le.fit_transform(df[col].astype(str))
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            datasets[file] = (X, y)
    return datasets

# ==========================================
# HFR-MADM DATASET RANKING
# ==========================================
def hfr_madm(datasets):
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    scores = []

    for name, (X, y) in datasets.items():
        f1 = min(len(X)/1500, 1)
        f2 = 1 - abs(0.5 - y.value_counts(normalize=True).iloc[0])
        f3 = min(X.shape[1]/25, 1)
        f4 = 0.95
        score = np.dot([f1, f2, f3, f4], weights)
        scores.append([name, score])

    return pd.DataFrame(scores, columns=["Dataset", "Score"]).sort_values("Score", ascending=False)

# ==========================================
# MODEL TRAINING (LR + RF)
# ==========================================
def train_models(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    # Logistic Regression
    lr = LogisticRegression(max_iter=5000)
    lr.fit(Xtr, ytr)
    lr_pred = lr.predict(Xte)
    lr_acc = accuracy_score(yte, lr_pred)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(Xtr, ytr)
    rf_pred = rf.predict(Xte)
    rf_acc = accuracy_score(yte, rf_pred)

    if rf_acc >= lr_acc:
        model, preds, name = rf, rf_pred, "Random Forest"
    else:
        model, preds, name = lr, lr_pred, "Logistic Regression"

    report = classification_report(yte, preds, output_dict=True)
    cm = confusion_matrix(yte, preds)

    return model, name, lr_acc, rf_acc, preds, report, cm, scaler, X.columns

# ==========================================
# LOAD DATA
# ==========================================
datasets = load_datasets("data")
ranking = hfr_madm(datasets)
best_dataset = ranking.iloc[0]["Dataset"]
X, y = datasets[best_dataset]

model, model_name, lr_acc, rf_acc, preds, report, cm, scaler, features = train_models(X, y)

low = np.sum(preds == 0)
high = np.sum(preds == 1)
overall = "HIGH RISK" if high > low else "LOW RISK"

# ==========================================
# LAYOUT
# ==========================================
left, right = st.columns([1, 1])

# ---------- LEFT PANEL ----------
with left:
    st.subheader("📌 Final Results")

    st.metric("Best Dataset", best_dataset)
    st.metric("Selected Model", model_name)

    st.metric("Logistic Regression Accuracy", f"{lr_acc*100:.2f}%")
    st.metric("Random Forest Accuracy", f"{rf_acc*100:.2f}%")

    st.markdown("### 🩺 Risk Summary")
    st.write(f"Low Risk Patients: *{low}*")
    st.write(f"High Risk Patients: *{high}*")
    st.success(f"Overall Prediction: *{overall}*")

    st.markdown("### 📋 Clinical Report")
    st.write(pd.DataFrame(report).transpose()[[
        "precision", "recall", "f1-score", "support"
    ]])

# ---------- RIGHT PANEL ----------
with right:
    st.subheader("📊 Visual Analytics")

    # Dataset Ranking Graph
    fig1, ax1 = plt.subplots()
    sns.barplot(x="Score", y="Dataset", data=ranking, ax=ax1)
    ax1.set_title("HFR-MADM Dataset Ranking")
    st.pyplot(fig1)

    # Risk Pie Chart
    fig2, ax2 = plt.subplots()
    ax2.pie([low, high], labels=["Low Risk", "High Risk"], autopct="%1.1f%%", startangle=90)
    ax2.set_title("Risk Distribution")
    st.pyplot(fig2)

    # Feature Importance
    if model_name == "Random Forest":
        importance = model.feature_importances_
    else:
        importance = np.abs(model.coef_[0])

    fi = pd.Series(importance, index=features).sort_values(ascending=False)[:10]
    fig3, ax3 = plt.subplots()
    fi.plot(kind="barh", ax=ax3)
    ax3.set_title("Top 10 Feature Importance")
    st.pyplot(fig3)

    # Confusion Matrix
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4)
    ax4.set_title("Confusion Matrix")
    st.pyplot(fig4)
