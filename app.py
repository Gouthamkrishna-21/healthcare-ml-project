import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. PAGE & THEME CONFIG
# ==========================================
st.set_page_config(
    page_title="HFR-MADM Healthcare System",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] { background-color: #e1f5fe; border-bottom: 2px solid #01579b; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOGIC & PROCESSING
# ==========================================

def load_datasets(data_folder="data"):
    datasets = {}
    if not os.path.exists(data_folder):
        st.error(f"Directory '{data_folder}' not found.")
        return datasets

    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            try:
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
            except Exception as e:
                st.sidebar.error(f"Error loading {file}: {e}")
    return datasets

def hfr_madm_logic(datasets):
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    ranking_data = []
    for name, (X, y, _) in datasets.items():
        f1 = min(len(X) / 1500, 1.0)
        f2 = 1.0 - abs(0.5 - y.value_counts(normalize=True).iloc[0])
        f3 = min(X.shape[1] / 25, 1.0)
        f4 = 0.95
        metrics = [f1, f2, f3, f4]
        hesitant_values = [np.mean([m*0.9, m, min(m*1.1, 1.0)]) for m in metrics]
        score = np.dot(hesitant_values, weights)
        ranking_data.append({
            "Dataset": name, 
            "Score": round(score, 4), 
            "Samples": len(X), 
            "Features": X.shape[1]
        })
    return pd.DataFrame(ranking_data).sort_values(by="Score", ascending=False)

def train_best_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    # Get the dictionary version for the table
    report = classification_report(y_test, preds, output_dict=True)
    
    return model, acc, cm, report, scaler, X.columns

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
st.sidebar.title("Clinical Dashboard")
st.sidebar.markdown("---")

all_data = load_datasets("data")

if not all_data:
    st.sidebar.warning("Please add CSV datasets to the 'data' folder.")
    st.stop()

rankings = hfr_madm_logic(all_data)

dataset_choice = st.sidebar.selectbox(
    "📂 Choose Dataset to Analyze", 
    options=list(all_data.keys()),
    index=0
)

st.sidebar.info(f"Currently Analyzing: {dataset_choice}")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.title("🩺 Hesitant Fuzzy Rough Healthcare Prediction")
st.caption(f"Analyzing Source: {dataset_choice}")

# Dynamic Training based on Selection
X_sel, y_sel, raw_df = all_data[dataset_choice]
model, acc, cm, report, scaler, feature_names = train_best_model(X_sel, y_sel)

tab1, tab2, tab3 = st.tabs(["📊 Global Evaluation", "🧪 Clinical Metrics", "🔍 Risk Predictor"])

with tab1:
    st.subheader("HFR-MADM Quality Ranking")
    st.write("Ranking results based on Volume, Balance, Features, and Reliability.")
    st.dataframe(rankings.style.highlight_max(axis=0, subset=['Score']), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**HFR-MADM Quality Scores**")
        fig_rank, ax_rank = plt.subplots()
        sns.barplot(data=rankings, x="Score", y="Dataset", palette="Blues_d", ax=ax_rank)
        st.pyplot(fig_rank)
    
    with col2:
        st.markdown("**Dataset Size Comparison (Samples)**")
        fig_size, ax_size = plt.subplots()
        sns.barplot(data=rankings, x="Samples", y="Dataset", palette="Greens_d", ax=ax_size)
        st.pyplot(fig_size)

with tab2:
    st.subheader(f"Clinical Performance Summary: {dataset_choice}")
    
    # Top Row Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Accuracy", f"{acc:.2%}")
    # Extracting weighted F1-Score from the report
    f1_weighted = report['weighted avg']['f1-score']
    m2.metric("Weighted F1-Score", f"{f1_weighted:.2%}")
    m3.metric("Samples Tested", report['macro avg']['support'])

    st.markdown("---")
    
    # Precision, Recall, F1 Table
    st.markdown("#### 📋 Comprehensive Classification Report")
    report_df = pd.DataFrame(report).transpose().iloc[:-3, :] # Remove averages for class-wise view
    st.table(report_df.style.background_gradient(cmap='PuBu', axis=0).format("{:.2%}"))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        st.pyplot(fig_cm)
    with col_b:
        st.markdown("**Feature Importance (Top 10)**")
        feat_importances = pd.Series(model.feature_importances_, index=feature_names)
        fig_fi, ax_fi = plt.subplots()
        feat_importances.nlargest(10).plot(kind='barh', color='#01579b', ax=ax_fi)
        st.pyplot(fig_fi)

with tab3:
    st.subheader(f"🔍 Patient Risk Diagnosis")
    st.info(f"Target Feature Structure: **{dataset_choice}**")
    
    if "active_dataset" not in st.session_state or st.session_state.active_dataset != dataset_choice:
        st.session_state.active_dataset = dataset_choice
        st.session_state.diagnosis_result = None

    with st.form(f"form_{dataset_choice}"):
        cols = st.columns(3)
        user_input = []
        for i, col_name in enumerate(feature_names):
            with cols[i % 3]:
                val = st.number_input(
                    f"{col_name}", 
                    value=float(raw_df[col_name].median()),
                    key=f"input_{dataset_choice}_{col_name}"
                )
                user_input.append(val)
        
        submit = st.form_submit_button("Generate Diagnosis", use_container_width=True)
        
        if submit:
            input_scaled = scaler.transform(np.array(user_input).reshape(1, -1))
            prediction = model.predict(input_scaled)
            prob = model.predict_proba(input_scaled)
            st.session_state.diagnosis_result = {
                "risk": "High" if prediction[0] == 1 else "Low",
                "confidence": max(prob[0])
            }

    if st.session_state.diagnosis_result:
        res = st.session_state.diagnosis_result
        st.markdown("---")
        if res["risk"] == "High":
            st.error(f"### ⚠️ Result: HIGH RISK\n**Model Confidence:** {res['confidence']:.2%}")
        else:
            st.success(f"### ✅ Result: LOW RISK\n**Model Confidence:** {res['confidence']:.2%}")
