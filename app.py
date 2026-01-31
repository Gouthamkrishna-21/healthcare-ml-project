import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. PAGE & THEME CONFIG
# ==========================================
st.set_page_config(
    page_title="HFR-MADM Intelligent Healthcare",
    page_icon="🩺",
    layout="wide"
)

# Custom Styling
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
# 2. LOGIC & PROCESSING (Combined)
# ==========================================

def load_datasets(data_folder="data"):
    datasets = {}
    if not os.path.exists(data_folder):
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
            except:
                continue
    return datasets

def hfr_madm_logic(datasets):
    # Hesitant Fuzzy Weights
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    ranking_data = []
    for name, (X, y, _) in datasets.items():
        f1 = min(len(X) / 1500, 1.0)
        f2 = 1.0 - abs(0.5 - y.value_counts(normalize=True).iloc[0])
        f3 = min(X.shape[1] / 25, 1.0)
        f4 = 0.95
        metrics = [f1, f2, f3, f4]
        # Hesitant Range Logic
        hesitant_values = [np.mean([m*0.9, m, min(m*1.1, 1.0)]) for m in metrics]
        score = np.dot(hesitant_values, weights)
        ranking_data.append({"Dataset": name, "Score": round(score, 4), "Samples": len(X), "Features": X.shape[1]})
    return pd.DataFrame(ranking_data).sort_values(by="Score", ascending=False)

def train_competitive_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_s))
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test_s))
    
    # Selection Logic (Choose the winner)
    if rf_acc >= lr_acc:
        best_model, best_name, best_acc = rf, "Random Forest", rf_acc
    else:
        best_model, best_name, best_acc = lr, "Logistic Regression", lr_acc
        
    preds = best_model.predict(X_test_s)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    
    return best_model, best_name, best_acc, lr_acc, rf_acc, cm, report, scaler, X.columns

# ==========================================
# 3. SIDEBAR & DATA LOADING
# ==========================================
all_data = load_datasets("data")
if not all_data:
    st.error("No data found in 'data' folder.")
    st.stop()

rankings = hfr_madm_logic(all_data)
st.sidebar.title("🩺 Clinical Control")
dataset_choice = st.sidebar.selectbox("📂 Select Dataset", options=list(all_data.keys()))
st.sidebar.success(f"Top Ranked: {rankings.iloc[0]['Dataset']}")

# ==========================================
# 4. MAIN INTERFACE (Combined)
# ==========================================
st.title("🩺 HFR-MADM Intelligent Healthcare System")
st.caption(f"Competitive Analysis for: {dataset_choice}")

X_sel, y_sel, raw_df = all_data[dataset_choice]
model, model_name, acc, lr_acc, rf_acc, cm, report, scaler, feature_names = train_competitive_models(X_sel, y_sel)

tab1, tab2, tab3 = st.tabs(["📊 Dataset Ranking", "🧪 Model Competition", "🔍 Patient Predictor"])

with tab1:
    st.subheader("HFR-MADM Evaluation Results")
    st.dataframe(rankings.style.highlight_max(subset=['Score']), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        fig1, ax1 = plt.subplots()
        sns.barplot(data=rankings, x="Score", y="Dataset", palette="viridis", ax=ax1)
        st.pyplot(fig1)
    with c2:
        # Pie Chart from Logic 2
        fig2, ax2 = plt.subplots()
        counts = y_sel.value_counts()
        ax2.pie(counts, labels=["Low Risk", "High Risk"], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
        st.pyplot(fig2)

with tab2:
    st.subheader("Algorithm Competition: LR vs RF")
    m1, m2, m3 = st.columns(3)
    m1.metric("Logistic Regression", f"{lr_acc:.2%}")
    m2.metric("Random Forest", f"{rf_acc:.2%}")
    m3.metric("WINNING MODEL", model_name)
    
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Confusion Matrix (Winner)**")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        st.pyplot(fig_cm)
    with col_b:
        st.markdown("**Performance Report**")
        st.table(pd.DataFrame(report).transpose().iloc[:-3, :].style.format("{:.2%}"))

with tab3:
    st.subheader("🔍 Real-Time Patient Diagnosis")
    st.info(f"Currently powered by: **{model_name}**")
    
    with st.form("diag_form"):
        cols = st.columns(3)
        user_input = []
        for i, col_name in enumerate(feature_names):
            with cols[i % 3]:
                val = st.number_input(f"{col_name}", value=float(raw_df[col_name].median()))
                user_input.append(val)
        
        btn = st.form_submit_button("Generate Prediction", use_container_width=True)
        if btn:
            input_s = scaler.transform(np.array(user_input).reshape(1, -1))
            res = model.predict(input_s)[0]
            prob = model.predict_proba(input_s)[0]
            
            if res == 1:
                st.error(f"### ⚠️ HIGH RISK DETECTED\nConfidence: {max(prob):.2%}")
            else:
                st.success(f"### ✅ LOW RISK DETECTED\nConfidence: {max(prob):.2%}")
