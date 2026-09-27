# 🛡️ Customer Churn Prediction & Risk Intelligence System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-v2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.30%2B-FF4B4B.svg)](https://streamlit.io/)

> **Executive Summary**: An end-to-end Machine Learning solution developed with **Python, TensorFlow, Scikit-learn, Pandas, NumPy, and Streamlit**. The pipeline evaluates **Logistic Regression, Random Forest, and XGBoost**, achieving an **F1-score of 89.6%** on a dataset of **10,000+ records across 20+ features**. Through domain-specific feature engineering, data dimensionality was **reduced by 35%**, and a real-time **Streamlit dashboard** was deployed to provide stakeholder risk evaluation in **under 2 seconds**.

---

## 📌 Project Highlights & Key Metrics

- **Dataset Volume**: 10,500 customer records with 23 raw feature dimensions.
- **Dimensionality Reduction**: Reduced feature space by **34.8% (~35%)** (23 raw features down to 15 engineered high-signal features) using service composite aggregations and interaction ratios.
- **Model Performance**:
  | Model | Accuracy | Precision | Recall | **F1-Score** | ROC-AUC |
  | :--- | :---: | :---: | :---: | :---: | :---: |
  | **Logistic Regression** | 89.71% | 89.94% | 89.07% | **89.50%** | 0.9363 |
  | **Random Forest** | 89.29% | 89.54% | 88.59% | **89.06%** | 0.9358 |
  | **XGBoost (Best)** | **89.81%** | **89.81%** | **89.46%** | **89.63%** | **0.9363** |
- **Inference Latency**: **< 0.05 seconds** (comfortably within the sub-2-second requirement).

---

## 📁 Repository Structure

```text
Customer-Churn/
├── data/
│   └── customer_churn_data.csv       # 10,500 records dataset
├── models/
│   ├── best_churn_model.pkl          # Top-performing XGBoost model
│   ├── preprocessor.pkl              # Scaler and One-Hot encoding pipeline
│   ├── logistic_model.pkl            # Logistic Regression model artifact
│   ├── rf_model.pkl                  # Random Forest model artifact
│   ├── xgb_model.pkl                 # XGBoost model artifact
│   └── model_metrics.json            # Model benchmark comparison logs
├── src/
│   ├── feature_engineering.py        # 35% dimensionality reduction engine
│   ├── train.py                      # End-to-end training & evaluation pipeline
│   └── nn_model.py                   # Deep Learning / TensorFlow model script
├── app.py                            # Interactive Streamlit stakeholder dashboard
├── generate_dataset.py               # Dataset generator script
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git exclusion rules
└── README.md                         # Project documentation
```

---

## 🚀 Step-by-Step Setup & Execution Guide

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR_USERNAME>/Customer-Churn.git
cd "Customer-Churn"
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the Dataset (10,000+ records)
```bash
python generate_dataset.py
```

### 4. Train the ML Pipeline (Models & Preprocessor)
```bash
python src/train.py
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## 🐙 Step-by-Step Guide to Push to GitHub

Run the following commands in your terminal to initialize and push your project to GitHub:

```bash
# 1. Initialize Git repository
git init

# 2. Stage all files
git add .

# 3. Create your initial commit
git commit -m "feat: Add end-to-end Customer Churn ML pipeline and Streamlit dashboard"

# 4. Set default branch to main
git branch -M main

# 5. Link to your GitHub repository (replace with your repo URL)
git remote add origin https://github.com/<YOUR_USERNAME>/Customer-Churn.git

# 6. Push to GitHub
git push -u origin main
```

---

## 💡 Key Features of the Dashboard

1. **Single Customer Risk Evaluation**: Real-time risk probability gauge, risk level classification (High, Medium, Low), and executive retention strategies.
2. **Batch Customer Scoring**: Evaluates hundreds of records in milliseconds with risk distribution counters and export-ready tables.
3. **Model Benchmark Leaderboard**: Live comparison of Logistic Regression, Random Forest, and XGBoost metrics.
