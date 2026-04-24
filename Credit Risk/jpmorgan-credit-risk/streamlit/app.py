# streamlit/app.py
# Run: streamlit run streamlit/app.py

import streamlit as st
import joblib, json, math, os
import numpy as np
import plotly.graph_objects as go
import shap

st.set_page_config(
    page_title="JPMorgan Credit Risk Analyzer",
    page_icon="🏦",
    layout="wide"
)

# ---------------- PATH FIX ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "credit_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_names.json")
METRICS_PATH = os.path.join(BASE_DIR, "..", "models", "metrics.json")

# ---------------- LOAD MODEL ---------------- #
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)   # ✅ joblib.load added correctly
        feats = json.load(open(FEATURE_PATH))

        try:
            metrics = json.load(open(METRICS_PATH))
        except:
            metrics = {}

        return model, feats, metrics, None

    except Exception as e:
        return None, None, {}, str(e)

model, features, metrics, load_error = load_model()

# ---------------- SHAP SAFE ---------------- #
explainer = None
if model:
    try:
        explainer = shap.TreeExplainer(model)
    except:
        explainer = None

# ---------------- HELPERS ---------------- #
def prob_to_score(pd_prob, base=600, pdo=20):
    pd_prob = max(min(pd_prob, 0.999), 0.001)
    return int(max(300, min(900, round(base + pdo * math.log2((1 - pd_prob) / pd_prob)))))

def engineer_features(d):
    try:
        d["total_dti"] = d["dti"] + (d["installment"] / (d["annual_inc"] / 12 + 1)) * 100
        d["loan_income_ratio"] = d["loan_amnt"] / (d["annual_inc"] + 1)
        d["high_utilization"] = int(d["revol_util"] > 75)
        d["has_derogatory"] = int(d["pub_rec"] > 0 or d["pub_rec_bankruptcies"] > 0)
        d["high_inquiries"] = int(d["inq_last_6mths"] >= 3)
        d["payment_to_income"] = d["installment"] / (d["annual_inc"] / 12 + 1)

        return [d.get(f, 0) for f in features]

    except Exception as e:
        st.error(f"Feature engineering error: {e}")
        return None

# ---------------- UI ---------------- #
st.title("🏦 JPMorgan Credit Risk Analyzer")

# ---------------- ERROR CHECK ---------------- #
if load_error:
    st.error(f"❌ Model loading failed: {load_error}")
    st.stop()

# ---------------- METRICS ---------------- #
metrics = metrics or {}

c1, c2, c3, c4 = st.columns(4)

c1.metric("Model AUC-ROC", metrics.get("auc_roc", "N/A"))
c2.metric("Training Samples", f'{metrics.get("training_samples", 0):,}')
c3.metric("Default Rate", f'{metrics.get("default_rate", 0)*100:.1f}%')
c4.metric("Model Version", "v1.0")

st.markdown("---")

# ---------------- INPUT ---------------- #
col1, col2, col3 = st.columns(3)

with col1:
    loan_amnt = st.number_input("Loan Amount", 1000, 40000, 15000)
    int_rate = st.slider("Interest Rate", 5.0, 30.0, 12.5)
    annual_inc = st.number_input("Annual Income", 10000, 500000, 65000)

with col2:
    dti = st.slider("DTI", 0.0, 60.0, 18.5)
    installment = st.number_input("Installment", 50, 2000, 350)
    revol_util = st.slider("Credit Utilization", 0.0, 100.0, 35.0)

with col3:
    delinq_2yrs = st.number_input("Delinquencies", 0, 10, 0)
    inq_last_6mths = st.number_input("Inquiries", 0, 10, 1)
    pub_rec = st.number_input("Public Records", 0, 5, 0)

# ---------------- BUTTON ---------------- #
if st.button("🚀 Predict", use_container_width=True):

    data = dict(
        loan_amnt=loan_amnt,
        int_rate=int_rate,
        installment=installment,
        annual_inc=annual_inc,
        dti=dti,
        delinq_2yrs=int(delinq_2yrs),
        inq_last_6mths=int(inq_last_6mths),
        open_acc=8,
        pub_rec=int(pub_rec),
        revol_bal=5000,
        revol_util=revol_util,
        total_acc=15,
        mort_acc=1,
        pub_rec_bankruptcies=0
    )

    X = engineer_features(data)

    if X is None:
        st.stop()

    try:
        prob = model.predict_proba([X])[0][1]
        score = prob_to_score(prob)

        st.success(f"Credit Score: {score}")
        st.info(f"Default Probability: {prob*100:.2f}%")

        # SHAP
        if explainer:
            st.subheader("🔍 Feature Impact")
            shap_vals = explainer.shap_values(np.array([X]))

            for f, v in sorted(zip(features, shap_vals[0]), key=lambda x: abs(x[1]), reverse=True)[:5]:
                st.write(f"{f}: {v:.3f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")