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

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="HFR-MADM Healthcare System",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.main { background-color: #f0f2f6; }
.stTabs [data-baseweb="tab"] {
    background-color: white;
    padding: 10px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATASETS
# ==========================================
def load_datasets(data_folder="data"):
    datasets = {}

    if not os.path.exists(data_folder):
        st.error("Data folder not found")
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

            if y.nunique() >= 2:
                datasets[file] = (X, y, df)

    return datasets

# ==========================================
# 3. HFR-MADM DATASET RANKING
# ==========================================
def hfr_madm_logic(datasets):
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    results = []

    for name, (X, y, _) in datasets.items():
        f1 = min(len(X) / 1500, 1.0)
        f2 = 1 - abs(0.5 - y.value_counts(normalize=True).iloc[0])
        f3 = min(X.shape[1] / 25, 1.0)
        f4 = 0.95

        metrics = [f1, f2, f3, f4]
        hesitant = [np.mean([m*0.9, m, min(m*1.1, 1.0)]) for m in metrics]
        score = np.dot(hesitant, weights)

        results.append({
            "Dataset": name,
            "Score": round(score, 4),
            "Samples": len(X),
            "Features": X.shape[1]
        })

    return pd.DataFrame(results).sort_values("Score", ascending=False)

# ==========================================
# 4. LOGISTIC REGRESSION TRAINING
# ==========================================
def train_logistic_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    return model, acc, cm, report, scaler, X.columns

# ==========================================
# 5. SIDEBAR
# ==========================================
st.sidebar.title("Clinical Dashboard")

all_data = load_datasets("data")
if not all_data:
    st.stop()

rankings = hfr_madm_logic(all_data)

dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    list(all_data.keys())
)

X_sel, y_sel, raw_df = all_data[dataset_choice]
model, acc, cm, report, scaler, feature_names = train_logistic_model(X_sel, y_sel)

# ==========================================
# 6. MAIN UI
# ==========================================
st.title("🩺 HFR-MADM Healthcare Prediction")
st.info("Model Used: **Logistic Regression**")

tab1, tab2, tab3 = st.tabs(["📊 Evaluation", "📈 Metrics", "🔍 Risk Predictor"])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("Dataset Quality Ranking")
    st.dataframe(rankings, use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Weighted F1 Score", f"{report['weighted avg']['f1-score']:.2%}")
    col3.metric("Test Samples", int(report['macro avg']['support']))

    st.markdown("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.markdown("### Feature Impact (Logistic Coefficients)")
    coef = pd.Series(abs(model.coef_[0]), index=feature_names)
    fig2, ax2 = plt.subplots()
    coef.nlargest(10).plot(kind="barh", ax=ax2)
    st.pyplot(fig2)

# ---------- TAB 3 ----------
with tab3:
    st.subheader("Patient Risk Prediction")

    user_input = []
    cols = st.columns(3)

    for i, col in enumerate(feature_names):
        with cols[i % 3]:
            val = st.number_input(col, float(raw_df[col].median()))
            user_input.append(val)

    if st.button("Predict Risk"):
        input_scaled = scaler.transform(np.array(user_input).reshape(1, -1))
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled).max()

        if prediction == 1:
            st.error(f"⚠️ HIGH RISK (Confidence: {probability:.2%})")
        else:
            st.success(f"✅ LOW RISK (Confidence: {probability:.2%})")
