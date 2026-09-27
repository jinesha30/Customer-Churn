import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure src path is accessible
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from feature_engineering import engineer_features, get_feature_lists

def run_training():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Raw Dataset
    data_path = os.path.join(BASE_DIR, "data", "customer_churn_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_dataset.py first.")
    
    df_raw = pd.read_csv(data_path)
    raw_dim = df_raw.shape[1] - 2  # excluding CustomerID and target Churn
    print(f"[INFO] Loaded raw dataset: {df_raw.shape[0]} rows, {raw_dim} input features.")
    
    # 2. Feature Engineering & 35% Dimensionality Reduction
    df_engineered = engineer_features(df_raw)
    eng_dim = df_engineered.shape[1] - 1  # excluding Churn
    reduction_pct = ((raw_dim - eng_dim) / raw_dim) * 100
    print(f"[INFO] Engineered feature space: {eng_dim} features (Dimensionality reduced by {reduction_pct:.1f}%).")
    
    X = df_engineered.drop("Churn", axis=1)
    y = df_engineered["Churn"]
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 4. Preprocessing Pipeline
    categorical_cols, numeric_cols = get_feature_lists()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
        ]
    )
    
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # 5. Define ML Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            learning_rate=0.07,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42
        )
    }
    
    results = {}
    best_model_name = None
    best_f1 = 0.0
    saved_models = {}
    
    print("\n" + "="*60)
    print("--- MODEL TRAINING & EVALUATION BENCHMARK ---")
    print("="*60)
    
    for name, model in models.items():
        model.fit(X_train_proc, y_train)
        y_pred = model.predict(X_test_proc)
        y_proba = model.predict_proba(X_test_proc)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(auc), 4),
            "Confusion_Matrix": confusion_matrix(y_test, y_pred).tolist()
        }
        
        print(f"\nModel: {name}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f} ({f1*100:.1f}%)")
        print(f"  ROC-AUC:   {auc:.4f}")
        
        saved_models[name] = model
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    print("\n" + "="*60)
    print(f"Best Model: {best_model_name} with F1-Score: {best_f1:.4f} ({best_f1*100:.1f}%)")
    print("="*60)
    
    # 6. Save Preprocessor & Models
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
    joblib.dump(saved_models["Logistic Regression"], os.path.join(models_dir, "logistic_model.pkl"))
    joblib.dump(saved_models["Random Forest"], os.path.join(models_dir, "rf_model.pkl"))
    joblib.dump(saved_models["XGBoost"], os.path.join(models_dir, "xgb_model.pkl"))
    joblib.dump(saved_models[best_model_name], os.path.join(models_dir, "best_churn_model.pkl"))
    
    # Save feature names and metadata
    metadata = {
        "best_model": best_model_name,
        "best_f1_score": round(float(best_f1), 4),
        "raw_features_count": raw_dim,
        "engineered_features_count": eng_dim,
        "dimensionality_reduction_pct": round(float(reduction_pct), 2),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "metrics": results
    }
    
    with open(os.path.join(models_dir, "model_metrics.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\n[SUCCESS] All models, preprocessor, and metrics saved to '{models_dir}'.")

if __name__ == "__main__":
    run_training()
