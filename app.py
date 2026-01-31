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
# 1. PAGE CONFIG & UI THEME
# ==========================================
st.set_page_config(page_title="HFR-MADM Healthcare", page_icon="🩺", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA PROCESSING LOGIC
# ==========================================

def load_datasets(folder="data"):
    datasets = {}
    if not os.path.exists(folder):
        os.makedirs(folder)
        return datasets
    
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(folder, file)).dropna()
                # Clean column names
                df.columns = [c.strip() for c in df.columns]
                
                # Encode text columns to numbers
                le = LabelEncoder()
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = le.fit_transform(df[col].astype(str))
                
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                datasets[file] = (X, y, df)
            except Exception as e:
                st.error(f"Error loading {file}: {e}")
    return datasets

def hfr_madm_ranking(datasets):
    weights = np.array([0.35, 0.25, 0.25, 0.15]) # Volume, Balance, Features, Reliability
    results = []
    for name, (X, y, _) in datasets.items():
        # Feature Extraction for MADM
        f1 = min(len(X) / 1000, 1.0) # Volume
        f2 = 1.0 - abs(0.5 - y.value_counts(normalize=True).iloc[0]) # Balance
        f3 = min(X.shape[1] / 20, 1.0) # Feature Complexity
        f4 = 0.95 # Reliability Constant
        
        # Hesitant Fuzzy Logic (Mean of uncertainty range)
        score = np.dot([f1, f2, f3, f4], weights)
        results.append({"Dataset": name, "Score": round(score, 4), "Samples": len(X), "Features": X.shape[1]})
    
    return pd.DataFrame(results).sort_values("Score", ascending=False)

def train_logic(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_s))
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test_s))
    
    # Select Winner
    if rf_acc >= lr_acc:
        best_model, m_name, m_acc = rf, "Random Forest", rf_acc
    else:
        best_model, m_name, m_acc = lr, "Logistic Regression", lr_acc
        
    preds = best_model.predict(X_test_s)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    
    return best_model, m_name, m_acc, lr_acc, rf_acc, report, cm, scaler

# ==========================================
# 3. MAIN APP INTERFACE
# ==========================================
st.title("🩺 HFR-MADM: Intelligent Disease Risk Framework")

data_dict = load_datasets("data")

if not data_dict:
    st.warning("⚠️ No CSV files found in 'data' folder. Please upload datasets to continue.")
    st.stop()

# Get Rankings
rank_df = hfr_madm_ranking(data_dict)
best_file = rank_df.iloc[0]["Dataset"]

# Sidebar Selection
st.sidebar.header("Settings")
selected_file = st.sidebar.selectbox("📂 Select Dataset", list(data_dict.keys()))
st.sidebar.info(f"Top Ranked Dataset: **{best_file}**")

# Training
X, y, raw_df = data_dict[selected_file]
model, model_name, acc, lr_acc, rf_acc, report, cm, scaler = train_logic(X, y)

# TABS
t1, t2, t3 = st.tabs(["📊 Quality Analysis", "🧪 Model Competition", "🔍 Prediction Tool"])

with t1:
    st.subheader("HFR-MADM Dataset Quality Ranking")
    st.dataframe(rank_df.style.highlight_max(axis=0, subset=['Score']), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.barplot(x="Score", y="Dataset", data=rank_df, palette="magma", ax=ax1)
        st.pyplot(fig1)
    with col2:
        # Risk Distribution Pie Chart
        fig2, ax2 = plt.subplots()
        y.value_counts().plot.pie(autopct='%1.1f%%', labels=["Low Risk", "High Risk"], colors=['#74c476','#fb6a4a'], ax=ax2)
        ax2.set_ylabel("")
        st.pyplot(fig2)

with t2:
    st.subheader("Algorithm Accuracy Comparison")
    c1, c2, c3 = st.columns(3)
    c1.metric("Logistic Regression", f"{lr_acc:.2%}")
    c2.metric("Random Forest", f"{rf_acc:.2%}")
    c3.metric("Selected Winner", model_name)
    
    st.markdown("---")
    res_l, res_r = st.columns(2)
    with res_l:
        st.markdown("**Confusion Matrix**")
        fig3, ax3 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        st.pyplot(fig3)
    with res_r:
        st.markdown("**Classification Report**")
        st.table(pd.DataFrame(report).transpose().iloc[:-3, :].style.format("{:.2%}"))

with t3:
    st.subheader("Patient Risk Diagnosis Form")
    st.write(f"This diagnosis is currently generated using the **{model_name}** model.")
    
    # Dynamic Form
    with st.form("input_form"):
        cols = st.columns(3)
        inputs = []
        for i, col_name in enumerate(X.columns):
            with cols[i % 3]:
                # Use median as default value to avoid errors
                val = st.number_input(f"{col_name}", value=float(X[col_name].median()))
                inputs.append(val)
        
        submit = st.form_submit_button("Predict Risk Status", use_container_width=True)
        
        if submit:
            final_input = scaler.transform(np.array(inputs).reshape(1, -1))
            prediction = model.predict(final_input)[0]
            probs = model.predict_proba(final_input)[0]
            
            st.markdown("---")
            if prediction == 1:
                st.error(f"### ⚠️ Result: HIGH RISK ({max(probs):.2%} confidence)")
            else:
                st.success(f"### ✅ Result: LOW RISK ({max(probs):.2%} confidence)")
