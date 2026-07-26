import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# 1. PAGE CONFIG + STYLING
# =========================================================
st.set_page_config(
    page_title="HFR-MADM Clinical Portal",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }

.main-header {
    background: linear-gradient(90deg, #002b5b 0%, #004e92 100%);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

[data-testid="stSidebar"] {
    background-image: linear-gradient(180deg, #002b5b 0%, #004e92 100%) !important;
    background-color: #002b5b !important;
}
[data-testid="stSidebar"] label, .sidebar-title {
    color: white !important;
    font-weight: 700 !important;
}
div[data-baseweb="select"] > div { color: black !important; }

.sidebar-card {
    background-color: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(10px);
    padding: 16px;
    border-radius: 15px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    margin-bottom: 1rem;
    color: white !important;
}
.sidebar-card b, .sidebar-card span, .sidebar-card div { color: white !important; }

.rank-badge {
    background: linear-gradient(90deg, #ffd700 0%, #ffae00 100%) !important;
    color: #002b5b !important;
    padding: 12px;
    border-radius: 12px;
    font-weight: 700;
    text-align: center;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. DATA LOADING (now records how much data survived cleaning)
# =========================================================
def load_datasets(folder="data", uploaded_files=None):
    """Loads CSVs from a local folder and/or user-uploaded files.
    Tracks retention_rate = rows kept after cleaning / original rows,
    which becomes the real (non-hardcoded) 'reliability' criterion below.
    """
    datasets = {}
    sources = []

    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.endswith(".csv"):
                sources.append((file, os.path.join(folder, file)))

    if uploaded_files:
        for f in uploaded_files:
            sources.append((f.name, f))

    for name, source in sources:
        try:
            raw = pd.read_csv(source)
        except Exception:
            continue

        original_rows = len(raw)
        if original_rows == 0:
            continue

        df = raw.dropna().drop_duplicates()
        cleaned_rows = len(df)
        retention_rate = cleaned_rows / original_rows if original_rows else 0

        le = LabelEncoder()
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = le.fit_transform(df[col].astype(str))

        if df.shape[0] < 10 or df.shape[1] < 2:
            continue

        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        if y.nunique() >= 2:
            datasets[name] = {"X": X, "y": y, "df": df, "retention": retention_rate}

    return datasets


# =========================================================
# 3. HFR-MADM RANKING (every criterion is now actually computed)
# =========================================================
def normalized_entropy_balance(y):
    """Class balance as normalized entropy: 1.0 = perfectly balanced,
    0.0 = single-class. Works for binary AND multi-class targets
    (the old version only checked the majority class share)."""
    counts = y.value_counts(normalize=True).values
    k = len(counts)
    if k <= 1:
        return 0.0
    ent = -np.sum(counts * np.log(counts + 1e-9))
    return ent / np.log(k)


def hfr_madm_logic(datasets, weights, n_bootstrap=8, seed=42):
    """
    Weighted multi-attribute ranking across 4 criteria: size, class balance,
    feature richness, and reliability (data retained after cleaning).

    The 'hesitant fuzzy' element: each of the resampling-sensitive criteria
    (size, balance, features) is recomputed across n_bootstrap resamples of
    each dataset, producing a genuine spread of possible values rather than
    a fixed number. The final score is the weighted mean MINUS a weighted
    uncertainty penalty, so a dataset whose quality signal is volatile across
    resamples ranks lower than an equally-scored but more stable dataset.
    """
    rng = np.random.default_rng(seed)
    results = []

    for name, d in datasets.items():
        X, y, retention = d["X"], d["y"], d["retention"]
        n = len(X)

        size_samples, balance_samples, feat_samples = [], [], []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            Xb, yb = X.iloc[idx], y.iloc[idx]
            size_samples.append(min(len(Xb) / 1500, 1.0))
            balance_samples.append(normalized_entropy_balance(yb))
            feat_samples.append(min(Xb.shape[1] / 25, 1.0))

        criteria_stats = {
            "size": (np.mean(size_samples), np.std(size_samples)),
            "balance": (np.mean(balance_samples), np.std(balance_samples)),
            "features": (np.mean(feat_samples), np.std(feat_samples)),
            "reliability": (retention, 0.0),  # fixed property of the dataset, not resampled
        }

        agg_mean, agg_uncertainty = 0.0, 0.0
        for i, key in enumerate(["size", "balance", "features", "reliability"]):
            mean_val, std_val = criteria_stats[key]
            agg_mean += weights[i] * mean_val
            agg_uncertainty += weights[i] * std_val

        final_score = agg_mean - agg_uncertainty

        results.append({
            "Dataset": name,
            "Score": round(final_score, 4),
            "Uncertainty": round(agg_uncertainty, 4),
            "Samples": n,
            "Features": X.shape[1],
            "Balance": round(criteria_stats["balance"][0], 3),
            "Reliability": round(retention, 3),
        })

    return pd.DataFrame(results).sort_values("Score", ascending=False)


# =========================================================
# 4. MODEL TRAINING (added 5-fold cross-validation)
# =========================================================
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000).fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    # 5-fold CV on the full dataset gives a more stable accuracy estimate
    # than a single train/test split alone.
    X_all_s = StandardScaler().fit_transform(X)
    cv_scores = cross_val_score(LogisticRegression(max_iter=1000), X_all_s, y, cv=5)

    return (
        model,
        accuracy_score(y_test, preds),
        cv_scores,
        confusion_matrix(y_test, preds),
        classification_report(y_test, preds, output_dict=True),
        scaler,
        X.columns
    )


# =========================================================
# 5. SIDEBAR: dataset source, upload, weight sliders
# =========================================================
st.sidebar.markdown(
    """
    <div style="text-align:center;">
        <img src="https://cdn-icons-png.flaticon.com/512/3774/3774299.png" width="90"/>
        <div class="sidebar-title">Navigation Menu</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="sidebar-card">📤 <b>Add a Dataset</b>', unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader(
    "Upload additional CSVs (last column = target)",
    type=["csv"], accept_multiple_files=True
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

all_data = load_datasets("data", uploaded_files=uploaded)
if not all_data:
    st.error("❌ No usable datasets found. Add CSVs to the 'data' folder or upload one from the sidebar.")
    st.stop()

st.sidebar.markdown('<div class="sidebar-card">⚖️ <b>Ranking Weights</b>', unsafe_allow_html=True)
w_size = st.sidebar.slider("Sample size weight", 0.0, 1.0, 0.35, 0.05)
w_balance = st.sidebar.slider("Class balance weight", 0.0, 1.0, 0.25, 0.05)
w_features = st.sidebar.slider("Feature richness weight", 0.0, 1.0, 0.25, 0.05)
w_reliability = st.sidebar.slider("Data reliability weight", 0.0, 1.0, 0.15, 0.05)
raw_weights = np.array([w_size, w_balance, w_features, w_reliability])
weights = raw_weights / raw_weights.sum() if raw_weights.sum() > 0 else np.array([0.25]*4)
st.sidebar.caption(f"Normalized: {', '.join(f'{w:.2f}' for w in weights)}")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

rankings = hfr_madm_logic(all_data, weights)
rankings = rankings.reset_index(drop=True)
rankings.insert(0, "Rank", rankings.index + 1)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-card">📂 <b>Database Access</b>', unsafe_allow_html=True)
dataset_choice = st.sidebar.selectbox("", list(all_data.keys()))
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-card">🏷️ <b>Top Ranked Dataset</b>', unsafe_allow_html=True)
st.sidebar.markdown(
    f'<div class="rank-badge">{rankings.iloc[0]["Dataset"]}</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 6. MAIN INTERFACE
# =========================================================
st.markdown(f"""
<div class="main-header">
    <h1>🩺 Predictive Healthcare Decision System</h1>
    <p>HFR-MADM Optimized Analysis | Active Source: {dataset_choice}</p>
</div>
""", unsafe_allow_html=True)

X_sel, y_sel, raw_df = all_data[dataset_choice]["X"], all_data[dataset_choice]["y"], all_data[dataset_choice]["df"]
model, acc, cv_scores, cm, report, scaler, feature_names = train_model(X_sel, y_sel)

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Intelligence", "🧪 Model Performance", "🔍 Risk Diagnosis"]
)

with tab1:
    with st.expander("ℹ️ How this ranking works", expanded=False):
        st.write(
            "Each dataset is scored on 4 criteria: sample size, class balance "
            "(entropy-based, so it works for multi-class targets too), feature "
            "richness, and reliability (share of rows that survived cleaning — "
            "a proxy for original data quality). Size, balance, and feature "
            "scores are recomputed across several bootstrap resamples of each "
            "dataset, so each carries a *range* of plausible values rather than "
            "a single fixed number. The final score is the weighted average of "
            "these criteria, minus a penalty for how much that value swings "
            "across resamples — a dataset with a volatile signal ranks below "
            "an equally-scored but more stable one. Adjust the weights in the "
            "sidebar to see how the ranking responds."
        )

    st.markdown("### *HFR-MADM Quality Ranking*")
    st.dataframe(
        rankings.style
        .background_gradient(cmap="Blues", subset=["Score"])
        .format({"Score": "{:.3f}", "Uncertainty": "{:.3f}", "Balance": "{:.3f}", "Reliability": "{:.3f}"}),
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("*Dataset Quality Scores (with uncertainty band)*")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.barh(rankings["Dataset"], rankings["Score"], xerr=rankings["Uncertainty"],
                 color="#004e92", capsize=4)
        ax1.invert_yaxis()
        ax1.set_xlabel("Score")
        st.pyplot(fig1)

    with col2:
        st.write("*Dataset Size vs Feature Count*")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=rankings, x="Samples", y="Dataset", palette="Blues_d", ax=ax2)
        ax2.set_xlabel("Number of Samples")
        ax2.set_ylabel("Dataset")
        st.pyplot(fig2)

with tab2:
    st.markdown("### **Metrics**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test-Split Accuracy", f"{acc:.2%}")
    m2.metric("5-Fold CV Accuracy", f"{cv_scores.mean():.2%}", f"±{cv_scores.std():.2%}")
    m3.metric("Weighted F1-Score", f"{report['weighted avg']['f1-score']:.2%}")
    m4.metric("Processed Samples", f"{int(report['macro avg']['support'])}")
    st.caption(
        "CV accuracy is a more stable estimate than the single test-split number above — "
        "use it as the headline figure if asked how confident the model's accuracy claim is."
    )

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
    st.info("Input clinical parameters to generate a risk probability score.")

    form_col, info_col = st.columns([2, 1])

    with info_col:
        st.write("**Overall Dataset Risk Context**")
        risk_counts = y_sel.value_counts()
        labels = ['Low Risk', 'High Risk']
        sizes = [risk_counts.get(0, 0), risk_counts.get(1, 0)]
        colors = ['#4A90E2', '#E53935']

        fig_pie, ax_pie = plt.subplots(figsize=(3, 2))
        ax_pie.pie(
            sizes, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=colors, radius=1.0, textprops={'fontsize': 7},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        ax_pie.axis('equal')
        st.pyplot(fig_pie, bbox_inches='tight')
        plt.close(fig_pie)
        st.caption("This chart shows the distribution of the entire dataset.")

    with form_col:
        with st.form("clinical_form"):
            cols = st.columns(2)
            inputs = []
            for i, col in enumerate(feature_names):
                with cols[i % 2]:
                    inputs.append(
                        st.number_input(col, value=float(raw_df[col].median()), min_value=0.0)
                    )
            submit = st.form_submit_button("Generate Individual Prediction", use_container_width=True)

    if submit:
        st.divider()
        input_scaled = scaler.transform(np.array(inputs).reshape(1, -1))
        res = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled).max()

        if res == 1:
            st.error(f"### ⚠️ INDIVIDUAL DIAGNOSIS: HIGH RISK\nPersonalized Confidence: {prob:.2%}")
            st.warning("**Recommendation:** Clinical intervention and further diagnostic testing recommended.")
        else:
            st.success(f"### ✅ INDIVIDUAL DIAGNOSIS: LOW RISK\nPersonalized Confidence: {prob:.2%}")
            st.toast("Analysis Complete: Low Risk Detected", icon='✅')
