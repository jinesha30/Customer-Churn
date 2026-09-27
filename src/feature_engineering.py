import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs domain-specific feature engineering and reduces dimensionality by ~35%.
    Original feature space: 23 features -> Reduced & engineered feature space: 15 features.
    (23 - 15) / 23 = 34.8% (~35%) dimensionality reduction.
    """
    data = df.copy()
    
    # 1. Composite Add-on Services Count (combines 6 redundant service columns into 1)
    addon_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    data["TotalAddonServices"] = 0
    for col in addon_cols:
        if col in data.columns:
            data["TotalAddonServices"] += (data[col] == "Yes").astype(int)
            
    # 2. Ratio & Interaction Features
    data["MonthlyChargesToTenureRatio"] = data["MonthlyCharges"] / (data["Tenure"] + 1)
    data["UsageToCostRatio"] = data["AverageMonthlyUsageGB"] / (data["MonthlyCharges"] + 1)
    data["ComplaintRatePerYear"] = data["ComplaintsLastYear"] / ((data["Tenure"] / 12) + 1)
    
    # 3. Selected Top 15 Informative Features after dimensionality reduction
    selected_features = [
        "Tenure",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "TotalAddonServices",
        "SatisfactionScore",
        "ComplaintsLastYear",
        "NumSupportTickets",
        "AverageMonthlyUsageGB",
        "MonthlyChargesToTenureRatio",
        "UsageToCostRatio",
        "ComplaintRatePerYear",
        "SeniorCitizen"
    ]
    
    if "Churn" in data.columns:
        return data[selected_features + ["Churn"]]
    else:
        return data[selected_features]

def get_feature_lists():
    categorical_cols = ["Contract", "InternetService", "PaymentMethod"]
    numeric_cols = [
        "Tenure", "MonthlyCharges", "TotalCharges", "TotalAddonServices",
        "SatisfactionScore", "ComplaintsLastYear", "NumSupportTickets",
        "AverageMonthlyUsageGB", "MonthlyChargesToTenureRatio",
        "UsageToCostRatio", "ComplaintRatePerYear", "SeniorCitizen"
    ]
    return categorical_cols, numeric_cols
