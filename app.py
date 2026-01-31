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
# 1. ADVANCED UI CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="HFR-MADM Clinical Portal",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS for Uniform Chart Containers
st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }
.main-header {
    background: linear-gradient(90deg, #004e92 0%, #000428 100%);
    padding: 2rem; border-radius: 15px; color: white;
    margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
div[data-testid="stMetric"] {
    background-color: white; padding: 20px; border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #edf2f7;
}
/* Ensure all chart areas have consistent white backing and height */
.chart-container {
    background-color: white; padding: 15px; border-radius: 12px;
    border: 1px solid #e2e8f0; margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOGIC (Unchanged Internal Logic)
# ==========================================
def load_datasets(folder="data"):
    datasets = {}
    if not os.path.exists(folder): return datasets
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file)).dropna().drop_duplicates()
            le = LabelEncoder()
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = le.fit_transform(df[col].astype(str))
            X, y = df.iloc[:, :-1], df.iloc[:, -1]
            if y.nunique() >= 2: datasets[file] = (X, y, df)
    return datasets

def hfr_madm_logic(datasets):
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    results = []
    for name, (X, y, _) in datasets.items():
        f1, f2, f3, f4 = min(len(X)/1500, 1.0), 1-abs(0.5-y.value_counts(normalize=True).iloc[0]), min(X.shape[1]/25, 1.0), 0.95
        hesitant = [np.mean([m*0.9, m, min(m*1.1, 1.0)]) for m in [f1, f2, f3, f4]]
        results.append({"Dataset": name, "Score": round(np.dot(hesitant, weights), 4), "Samples": len(X), "Features": X.shape[1]})
    return pd.DataFrame(results).sort_values("Score", ascending=False)

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
    return model, accuracy_score(y_test, model.predict(X_test_s)), confusion_matrix(y_test, model.predict(X_test_s)), classification_report(y_test, model.predict(X_test_s), output_dict=True), scaler, X.columns

# ==========================================
# 3. DATA PREPARATION
# ==========================================
all_data = load_datasets("data")
if not all_data:
    st.error("Please ensure the 'data' folder contains CSV files.")
    st.stop()

rankings = hfr_madm_logic(all_data).reset_index(drop=True)
rankings.insert(0, "Rank", rankings.index + 1)
dataset_choice = st.sidebar.selectbox("📂 Select Database", list(all_data.keys()))
X_sel, y_sel, raw_df = all_data[dataset_choice]
model, acc, cm, report, scaler, feature_names = train_model(X_sel, y_sel)

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.markdown(f'<div class="main-header"><h1>🩺 Predictive Healthcare Decision System</h1><p>HFR-MADM Optimized Analysis | Active Source: {dataset_choice}</p></div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Data Intelligence", "🧪 Model Performance", "🔍 Risk Diagnosis"])

with tab1:
    st.markdown("### **HFR-MADM Quality Ranking**")
    st.dataframe(rankings.style.background_gradient(cmap="Blues", subset=["Score"]).format({"Score": "{:.3f}"}), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Dataset Quality Scores**")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=rankings, x="Score", y="Dataset", palette="Blues_d", ax=ax1)
        st.pyplot(fig1)
    with c2:
        st.write("**Dataset Size Comparison**")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=rankings, x="Samples", y="Dataset", palette="Blues_d", ax=ax2)
        st.pyplot(fig2)

with tab2:
    st.markdown("### **Model Metrics**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Diagnostic Accuracy", f"{acc:.2%}")
    m2.metric("Weighted F1-Score", f"{report['weighted avg']['f1-score']:.2%}")
    m3.metric("Processed Samples", f"{int(report['macro avg']['support'])}")
    st.markdown("---")
    l, r = st.columns(2)
    with l:
        st.write("**Confusion Matrix**")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        st.pyplot(fig_cm)
    with r:
        st.write("**Clinical Feature Weights**")
        coef = pd.Series(abs(model.coef_[0]), index=feature_names)
        fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
        coef.nlargest(10).plot(kind="barh", color="#004e92", ax=ax_fi)
        ax_fi.invert_yaxis()
        st.pyplot(fig_fi)

with tab3:
    st.markdown("### **Patient Risk Predictor**")
    st.info("Input clinical parameters to generate an individual risk probability.")
    
    with st.form("clinical_form"):
        cols = st.columns(3)
        inputs = []
        for i, col in enumerate(feature_names):
            with cols[i % 3]:
                # Added min_value=0.0 to allow for younger ages (e.g. 36)
                inputs.append(st.number_input(col, value=float(raw_df[col].median()), min_value=0.0))
        
        submitted = st.form_submit_button("Generate Prediction", use_container_width=True)

    if submitted:
        input_scaled = scaler.transform(np.array(inputs).reshape(1, -1))
        res = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0] # Individual probabilities

        # Results Display
        res_col, chart_col = st.columns([1, 1])
        
        with res_col:
            st.write("**Diagnosis Result**")
            if res == 1:
                st.error(f"### ⚠️ HIGH RISK\nConfidence: {probs[1]:.2%}")
            else:
                st.success(f"### ✅ LOW RISK\nConfidence: {probs[0]:.2%}")
            
            st.write(f"""
            **Explanation:** Based on the clinical parameters provided, the model indicates a **{max(probs):.2%}** probability that the patient belongs to the **{'High' if res==1 else 'Low'} Risk** category.
            """)

        with chart_col:
            st.write("**Individual Risk Distribution**")
            # --- THE NEW PIE CHART (Same size as others) ---
            fig_p, ax_p = plt.subplots(figsize=(6, 4))
            ax_p.pie(probs, labels=["Low Risk", "High Risk"], autopct='%1.1f%%', 
                    colors=['#4A90E2', '#D0021B'], startangle=90, explode=(0.05, 0.05))
            st.pyplot(fig_p)
