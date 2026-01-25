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

# Custom Medical CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #e1f5fe; border-bottom: 2px solid #01579b; }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
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
                
                # Auto-encoding for non-numeric clinical data
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
    """
    Calculates Dataset Quality Score using 
    Hesitant Fuzzy Rough Multi-Attribute Decision Making.
    """
    # Criteria Weights: [Volume, Class Balance, Feature Richness, Reliability]
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    ranking_data = []

    for name, (X, y, _) in datasets.items():
        # Normalized Attributes
        f1 = min(len(X) / 1500, 1.0) # Size factor
        f2 = 1.0 - abs(0.5 - y.value_counts(normalize=True).iloc[0]) # Balance factor
        f3 = min(X.shape[1] / 25, 1.0) # Feature factor
        f4 = 0.95 # Reliability constant
        
        metrics = [f1, f2, f3, f4]
        # Hesitant Fuzzy calculation (applying fuzzy bounds)
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Using Random Forest for better clinical feature analysis
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    
    return model, acc, report, cm, scaler, X.columns

# ==========================================
# 3. SIDEBAR & NAVIGATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80)
st.sidebar.title("Clinical Dashboard")
st.sidebar.markdown("---")
analyze_btn = st.sidebar.button("🚀 Run System Analysis", use_container_width=True)
st.sidebar.markdown("### System Info")
st.sidebar.info("This system uses HFR-MADM to select the most reliable healthcare dataset before training prediction models.")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.title("🩺 Hesitant Fuzzy Rough Healthcare Prediction")
st.caption("Advanced Dataset Selection & Disease Risk Classification System")

if analyze_btn:
    data_store = load_datasets("data")
    
    if len(data_store) < 2:
        st.warning("⚠️ Please ensure at least 2 CSV files are in the 'data' folder for comparison.")
    else:
        # Step 1: Rankings
        rank_df = hfr_madm_logic(data_store)
        best_dataset_name = rank_df.iloc[0]['Dataset']
        X_best, y_best, raw_df = data_store[best_dataset_name]
        
        # Step 2: Training
        model, acc, report, cm, scaler, feature_names = train_best_model(X_best, y_best)
        
        # UI TABS
        tab1, tab2, tab3 = st.tabs(["📊 Dataset Evaluation", "🧪 Clinical Metrics", "🔍 Risk Predictor"])
        
        with tab1:
            st.subheader("HFR-MADM Ranking Results")
            c1, c2 = st.columns([1, 1.5])
            with c1:
                st.write("Quality scores based on volume, balance, and feature richness.")
                st.dataframe(rank_df.style.highlight_max(axis=0, subset=['Score']), use_container_width=True)
            with c2:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=rank_df, x="Score", y="Dataset", palette="Blues_d")
                st.pyplot(fig)
            
            st.success(f"🏆 The system has selected **{best_dataset_name}** as the most reliable source.")

        with tab2:
            st.subheader("Model Performance Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction Accuracy", f"{acc:.2%}")
            m2.metric("Best Source", best_dataset_name)
            m3.metric("Features Analyzed", len(feature_names))
            
            
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Confusion Matrix**")
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                st.pyplot(fig_cm)
            with col_b:
                st.markdown("**Feature Importance**")
                feat_importances = pd.Series(model.feature_importances_, index=feature_names)
                fig_fi, ax_fi = plt.subplots()
                feat_importances.nlargest(10).plot(kind='barh', color='#01579b')
                st.pyplot(fig_fi)

        with tab3:
            st.subheader("🔍 Real-time Patient Risk Assessment")
            st.info("The form below is automatically generated based on the features of the selected dataset.")
            
            # 1. Initialize session state for the result so it persists after clicking
            if "diagnosis_result" not in st.session_state:
                st.session_state.diagnosis_result = None

            # 2. The Input Form
            with st.form("prediction_form"):
                cols = st.columns(3)
                user_input = []
                
                # Dynamic input fields based on the dataset
                for i, col_name in enumerate(feature_names):
                    with cols[i % 3]:
                        # Using a unique key for each input is best practice in Streamlit
                        val = st.number_input(
                            f"{col_name}", 
                            value=float(raw_df[col_name].median()),
                            key=f"input_{col_name}"
                        )
                        user_input.append(val)
                
                submit = st.form_submit_button("Generate Diagnosis", use_container_width=True)
                
                if submit:
                    # Prepare the data
                    input_array = np.array(user_input).reshape(1, -1)
                    input_scaled = scaler.transform(input_array)
                    
                    # Run Prediction
                    prediction = model.predict(input_scaled)
                    prob = model.predict_proba(input_scaled)
                    
                    # Save results to session state so they stay visible
                    st.session_state.diagnosis_result = {
                        "risk": "High" if prediction[0] == 1 else "Low",
                        "confidence": max(prob[0])
                    }

            # 3. Display the result OUTSIDE the form block for better visibility
            if st.session_state.diagnosis_result:
                res = st.session_state.diagnosis_result
                st.markdown("---")
                st.markdown("### 📋 Clinical Result")
                
                if res["risk"] == "High":
                    st.error(f"### ⚠️ High Risk Detected\n**Confidence Level:** {res['confidence']:.2%}")
                    st.markdown("> **Recommendation:** Urgent clinical follow-up and further diagnostic testing advised.")
                else:
                    st.success(f"### ✅ Low Risk Detected\n**Confidence Level:** {res['confidence']:.2%}")
                    st.markdown("> **Observation:** Patient metrics are within normal bounds for this specific model.")    

else:
    # Landing Page State
    st.info("👈 Use the sidebar to trigger the HFR-MADM analysis engine.")
    
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 1. Data Selection")
    c1.write("The system scans the /data folder for clinical CSV files.")
    
    c2.markdown("### 2. Fuzzy Logic")
    c2.write("HFR-MADM ranks datasets by calculating hesitant fuzzy uncertainty.")
    
    c3.markdown("### 3. Prediction")
    c3.write("A Random Forest model is trained on the best source for high accuracy.")

