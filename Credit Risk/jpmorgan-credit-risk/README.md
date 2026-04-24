# 🏦 JPMorgan Credit Risk Analyzer

An end-to-end **Machine Learning + Full Stack Application** that predicts loan default risk and generates a credit score using real-world financial features.

This project simulates a **fintech-grade credit risk system**, combining a trained ML model, REST API, and interactive dashboard.

---

## 🚀 Features

* 🔮 **Credit Risk Prediction** (Default Probability)
* 💳 **Credit Score Calculation (300–900 range)**
* ⚙️ **FastAPI Backend (Production-ready API)**
* 🎨 **Streamlit Frontend Dashboard**
* 📊 **Feature Engineering Pipeline**
* 🔍 **SHAP Explainability (Model Interpretability)**
* 📈 **Interactive Visualizations (Plotly Gauges)**

---

## 🧠 Tech Stack

* **Machine Learning:** Scikit-learn / XGBoost
* **Backend:** FastAPI
* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly
* **Model Explainability:** SHAP
* **Deployment Ready:** Uvicorn

---

## 📂 Project Structure

```
jpmorgan-credit-risk/
│
├── api/                # FastAPI backend
│   └── credit_api.py
│
├── streamlit/          # Frontend UI
│   └── app.py
│
├── models/             # Trained model & metadata
│   ├── credit_model.pkl
│   ├── feature_names.json
│   └── metrics.json
│
├── notebooks/          # Model training & analysis
├── data/               # Dataset
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/jpmorgan-credit-risk.git
cd jpmorgan-credit-risk
```

### 2️⃣ Create virtual environment

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Application

### ▶️ Start Backend API

```bash
python -m uvicorn api.credit_api:app --reload --port 8001
```

👉 Open API docs:
http://127.0.0.1:8001/docs

---

### 🎨 Start Frontend UI

```bash
streamlit run streamlit/app.py
```

👉 Open UI:
http://localhost:8501

---

## 📊 Example Input

```json
{
  "loan_amnt": 15000,
  "int_rate": 12.5,
  "installment": 350,
  "annual_inc": 65000,
  "dti": 18.5
}
```

---

## 📈 Output

* Credit Score (300–900)
* Default Probability
* Decision (Approve / Review / Reject)
* Feature Impact (SHAP)

---

## 💡 Key Highlights

* Built a **real-world fintech credit scoring system**
* Designed **scalable API architecture**
* Integrated **ML model with UI + backend**
* Implemented **explainable AI (XAI)** using SHAP

---

## 🎯 Future Improvements

* Deploy on AWS / Render / Docker
* Add real dataset (LendingClub)
* Improve model accuracy with advanced tuning
* Add authentication system

---

## 👨‍💻 Author

**Aditya Sai**
Aspiring Software Engineer | ML Engineer | Full Stack Developer

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share!
