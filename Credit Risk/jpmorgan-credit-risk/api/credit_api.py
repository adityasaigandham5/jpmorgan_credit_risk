# api/credit_api.py
# Run from the PROJECT ROOT with:
#   uvicorn api.credit_api:app --reload --port 8001
# OR run from inside the api/ folder with:
#   uvicorn credit_api:app --reload --port 8001

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import math
import time
import os

# ── Path resolution ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATH    = os.path.join(MODELS_DIR, "credit_model.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="JPMorgan Credit Risk API",
    description="Predicts loan default probability and generates a 300-900 credit score",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ──────────────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r") as f:
        FEATURES = json.load(f)
    print(f"[OK] Model loaded — {len(FEATURES)} features")
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model files not found: {e}\n"
        "Run notebooks/03_model_training.ipynb first."
    )

# ── Input schema (Pydantic v2 compatible) ─────────────────────────────────────
class LoanApplication(BaseModel):
    model_config = {"json_schema_extra": {
        "example": {
            "loan_amnt": 15000, "int_rate": 12.5, "installment": 350.0,
            "annual_inc": 65000, "dti": 18.5, "delinq_2yrs": 0,
            "inq_last_6mths": 1, "open_acc": 8, "pub_rec": 0,
            "revol_bal": 5000.0, "revol_util": 35.0, "total_acc": 15,
            "mort_acc": 1, "pub_rec_bankruptcies": 0,
        }
    }}

    loan_amnt:            float = Field(...,    description="Loan amount ($)")
    int_rate:             float = Field(...,    description="Interest rate (%)")
    installment:          float = Field(...,    description="Monthly payment ($)")
    annual_inc:           float = Field(...,    description="Annual income ($)")
    dti:                  float = Field(...,    description="Debt-to-income ratio")
    delinq_2yrs:          int   = Field(0,      description="Late payments in last 2 years")
    inq_last_6mths:       int   = Field(0,      description="Credit inquiries in last 6 months")
    open_acc:             int   = Field(8,      description="Open credit accounts")
    pub_rec:              int   = Field(0,      description="Public derogatory records")
    revol_bal:            float = Field(5000.0, description="Revolving balance ($)")
    revol_util:           float = Field(35.0,   description="Credit utilization (%)")
    total_acc:            int   = Field(15,     description="Total credit accounts")
    mort_acc:             int   = Field(1,      description="Mortgage accounts")
    pub_rec_bankruptcies: int   = Field(0,      description="Bankruptcies")

# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(data: dict) -> list:
    d = data.copy()
    monthly_income         = d["annual_inc"] / 12 + 1      # +1 avoids division by zero
    d["total_dti"]         = d["dti"] + (d["installment"] / monthly_income) * 100
    d["loan_income_ratio"] = d["loan_amnt"] / (d["annual_inc"] + 1)
    d["high_utilization"]  = int(d["revol_util"] > 75)
    d["has_derogatory"]    = int(d["pub_rec"] > 0 or d["pub_rec_bankruptcies"] > 0)
    d["high_inquiries"]    = int(d["inq_last_6mths"] >= 3)
    d["payment_to_income"] = d["installment"] / monthly_income
    try:
        return [d[feat] for feat in FEATURES]
    except KeyError as missing:
        raise ValueError(f"Feature {missing} missing after engineering.")

# ── Scoring helpers ────────────────────────────────────────────────────────────
def prob_to_score(pd_prob: float, base: int = 600, pdo: int = 20) -> int:
    pd_prob = max(min(pd_prob, 0.999), 0.001)
    return int(max(300, min(900, round(base + pdo * math.log2((1 - pd_prob) / pd_prob)))))

def get_decision(score: int) -> tuple:
    if score >= 700: return "AUTO_APPROVE",  "GREEN"
    if score >= 500: return "MANUAL_REVIEW", "AMBER"
    return                   "AUTO_REJECT",  "RED"

def get_risk_factors(app: LoanApplication, prob: float) -> list:
    reasons = []
    if prob > 0.5:
        if app.dti > 30:
            reasons.append(f"High debt-to-income ratio ({app.dti:.1f}% > 30% threshold)")
        if app.int_rate > 20:
            reasons.append(f"High interest rate ({app.int_rate:.1f}%)")
        if app.delinq_2yrs > 0:
            reasons.append(f"Recent delinquencies ({app.delinq_2yrs} in last 2 years)")
        if app.revol_util > 75:
            reasons.append(f"High credit utilization ({app.revol_util:.1f}%)")
        if app.inq_last_6mths >= 3:
            reasons.append(f"Multiple credit inquiries ({app.inq_last_6mths} in 6 months)")
        if app.pub_rec > 0 or app.pub_rec_bankruptcies > 0:
            reasons.append("Derogatory public records or bankruptcies on file")
    return reasons[:3]

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "running", "model": "JPMorgan Credit Risk v1.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None, "features": len(FEATURES)}

@app.post("/predict")
def predict(application: LoanApplication):
    start = time.time()
    try:
        raw             = application.model_dump()
        feat_vector     = engineer_features(raw)
        prob            = float(model.predict_proba([feat_vector])[0][1])
        score           = prob_to_score(prob)
        decision, color = get_decision(score)
        reasons         = get_risk_factors(application, prob)
        latency_ms      = round((time.time() - start) * 1000, 2)

        return {
            "credit_score":        score,
            "default_probability": round(prob, 4),
            "decision":            decision,
            "risk_color":          color,
            "top_risk_factors":    reasons,
            "latency_ms":          latency_ms,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")