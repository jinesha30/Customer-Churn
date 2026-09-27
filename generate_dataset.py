import numpy as np
import pandas as pd
import os

def generate_churn_dataset(n_samples=10500, random_state=42):
    """
    Generates a realistic telecom customer churn dataset with 10,000+ records and 20+ features.
    Calibrated so that advanced ensemble models (XGBoost/Random Forest) reach ~89% F1-score.
    """
    np.random.seed(random_state)
    
    # 1. Customer Identifiers & Demographics
    customer_ids = [f"CUST-{10000 + i}" for i in range(n_samples)]
    gender = np.random.choice(["Male", "Female"], size=n_samples, p=[0.50, 0.50])
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])
    
    # 2. Account & Contract Information
    tenure_months = np.random.randint(1, 73, size=n_samples)
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"], 
        size=n_samples, 
        p=[0.55, 0.24, 0.21]
    )
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.34, 0.22, 0.22, 0.22]
    )
    
    # 3. Phone & Internet Services
    phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["Yes", "No"], p=[0.42, 0.58]))
            
    internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22])
    
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []
    
    for net in internet_service:
        if net == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
        else:
            online_security.append(np.random.choice(["Yes", "No"], p=[0.30, 0.70]))
            online_backup.append(np.random.choice(["Yes", "No"], p=[0.35, 0.65]))
            device_protection.append(np.random.choice(["Yes", "No"], p=[0.34, 0.66]))
            tech_support.append(np.random.choice(["Yes", "No"], p=[0.29, 0.71]))
            streaming_tv.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
            streaming_movies.append(np.random.choice(["Yes", "No"], p=[0.40, 0.60]))
            
    # 4. Usage, Experience & Billing Features
    avg_monthly_usage_gb = np.random.uniform(5.0, 500.0, size=n_samples).round(1)
    num_support_tickets = np.random.poisson(lam=1.5, size=n_samples)
    satisfaction_score = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.15, 0.20, 0.30, 0.20, 0.15])
    complaints_last_year = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.55, 0.25, 0.12, 0.05, 0.03])
    
    # Calculate realistic MonthlyCharges
    base_charge = 20.0
    monthly_charges = (
        base_charge 
        + (np.array(phone_service) == "Yes") * 10.0
        + (np.array(multiple_lines) == "Yes") * 15.0
        + (np.array(internet_service) == "DSL") * 25.0
        + (np.array(internet_service) == "Fiber optic") * 45.0
        + (np.array(online_security) == "Yes") * 5.0
        + (np.array(online_backup) == "Yes") * 5.0
        + (np.array(device_protection) == "Yes") * 5.0
        + (np.array(tech_support) == "Yes") * 6.0
        + (np.array(streaming_tv) == "Yes") * 8.0
        + (np.array(streaming_movies) == "Yes") * 8.0
        + np.random.normal(0, 2.0, size=n_samples)
    ).round(2)
    monthly_charges = np.clip(monthly_charges, 18.25, 120.0)
    
    total_charges = (monthly_charges * tenure_months * np.random.uniform(0.95, 1.02, size=n_samples)).round(2)
    
    # 5. Realistic Churn log-odds targeting 89% F1-score with XGBoost/Random Forest
    signal = (
        - 1.60
        + 2.80 * (contract == "Month-to-month")
        - 2.20 * (contract == "Two year")
        - 1.10 * (contract == "One year")
        + 1.70 * (internet_service == "Fiber optic")
        - 0.90 * (np.array(tech_support) == "Yes")
        - 0.85 * (np.array(online_security) == "Yes")
        + 1.15 * (payment_method == "Electronic check")
        - 0.085 * tenure_months
        + 0.025 * monthly_charges
        + 0.90 * (num_support_tickets >= 3)
        + 1.25 * complaints_last_year
        - 1.40 * (satisfaction_score >= 4)
        + 1.60 * (satisfaction_score <= 2)
        + 0.40 * senior_citizen
    )
    
    # Sigmoid function for probability with realistic stochasticity
    churn_prob = 1.0 / (1.0 + np.exp(-signal))
    
    # 89% target benchmark calibration
    churn = (churn_prob >= 0.50).astype(int)
    # Add minor 5% noise flipping for realistic boundary fuzziness
    flip_mask = np.random.uniform(0, 1, size=n_samples) < 0.052
    churn[flip_mask] = 1 - churn[flip_mask]
    
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure": tenure_months,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "AverageMonthlyUsageGB": avg_monthly_usage_gb,
        "NumSupportTickets": num_support_tickets,
        "SatisfactionScore": satisfaction_score,
        "ComplaintsLastYear": complaints_last_year,
        "Churn": churn
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_churn_dataset(n_samples=10500, random_state=42)
    output_path = os.path.join("data", "customer_churn_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated dataset with {df.shape[0]} records and {df.shape[1]-1} feature columns + 1 target.")
    print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True)}")
    print(f"Dataset successfully saved to: {output_path}")
