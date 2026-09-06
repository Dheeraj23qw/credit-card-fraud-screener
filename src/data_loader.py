"""Data generator, loader, and initial cleaning for Credit Card Fraud Detection."""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "credit_card_transactions.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_fraud_data.csv")


def generate_fraud_dataset(num_samples: int = 10000, fraud_rate: float = 0.02, random_state: int = 42) -> pd.DataFrame:
    """Generates a realistic credit card transactions dataset with extreme class imbalance."""
    np.random.seed(random_state)
    
    num_fraud = int(num_samples * fraud_rate)
    num_legit = num_samples - num_fraud
    
    # 1. Transaction Amounts
    # Legitimate: log-normal distribution with median ~$45
    legit_amount = np.random.lognormal(mean=3.8, sigma=1.0, size=num_legit)
    legit_amount = np.clip(legit_amount, 1.5, 2500.0)
    
    # Fraudulent: 40% overlap with normal spending patterns, 30% micro-test, 30% high-value
    fraud_normal = np.random.lognormal(mean=3.8, sigma=1.0, size=int(num_fraud * 0.40))  # looks like normal
    fraud_low = np.random.uniform(1.0, 8.0, size=int(num_fraud * 0.30))
    fraud_high = np.random.lognormal(mean=5.8, sigma=0.8, size=num_fraud - len(fraud_normal) - len(fraud_low))
    fraud_high = np.clip(fraud_high, 150.0, 5000.0)
    fraud_amount = np.concatenate([fraud_normal, fraud_low, fraud_high])
    np.random.shuffle(fraud_amount)
    
    # 2. Time (Hour of Day: 0 to 23)
    # Legitimate transactions: peak during 9 AM - 9 PM
    legit_hours = np.random.normal(loc=14.0, scale=4.5, size=num_legit) % 24
    # Fraudulent: mild late-night bias but significant daytime overlap
    f_probs = np.array([0.06, 0.07, 0.08, 0.08, 0.06, 0.04, 0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.05, 0.05])
    f_probs = f_probs / f_probs.sum()
    fraud_hours = np.random.choice(list(range(24)), size=num_fraud, p=f_probs)
    
    # 3. Distance from Cardholder Home (KM)
    # Key fix: 40% of fraud happens nearby (same city), making distance alone insufficient
    legit_dist = np.random.exponential(scale=12.0, size=num_legit)
    fraud_nearby = np.random.exponential(scale=15.0, size=int(num_fraud * 0.40))  # nearby fraud
    fraud_medium = np.random.exponential(scale=80.0, size=int(num_fraud * 0.30)) + 20.0  # moderate distance
    fraud_far = np.random.exponential(scale=400.0, size=num_fraud - len(fraud_nearby) - len(fraud_medium)) + 50.0
    fraud_dist = np.concatenate([fraud_nearby, fraud_medium, fraud_far])
    np.random.shuffle(fraud_dist)
    
    # 4. Merchant Category
    categories = ["Grocery", "Online Retail", "Dining", "Travel & Airlines", "Electronics", "Gas Station", "Luxury Goods", "ATM Withdrawal"]
    legit_cat = np.random.choice(categories, size=num_legit, p=[0.28, 0.22, 0.20, 0.05, 0.06, 0.12, 0.03, 0.04])
    fraud_cat = np.random.choice(categories, size=num_fraud, p=[0.04, 0.38, 0.05, 0.18, 0.16, 0.03, 0.14, 0.02])
    
    # 5. Transaction Channel / Method
    methods = ["In-Person Chip", "Online POS", "Contactless Tap", "ATM Cash"]
    legit_method = np.random.choice(methods, size=num_legit, p=[0.45, 0.30, 0.20, 0.05])
    fraud_method = np.random.choice(methods, size=num_fraud, p=[0.10, 0.75, 0.05, 0.10])
    
    # 6. Card Type
    card_types = ["Visa", "Mastercard", "Amex", "Discover"]
    legit_cards = np.random.choice(card_types, size=num_legit, p=[0.48, 0.34, 0.12, 0.06])
    fraud_cards = np.random.choice(card_types, size=num_fraud, p=[0.45, 0.35, 0.14, 0.06])
    
    # 7. Velocity: Transactions in last 24h by this card
    legit_velocity = np.random.poisson(lam=1.8, size=num_legit)
    fraud_velocity = np.random.poisson(lam=6.5, size=num_fraud)
    
    # 8. Anonymized Latent Risk Factor (PCA components V1, V2) — with significant class overlap
    legit_v1 = np.random.normal(loc=0.0, scale=1.2, size=num_legit)
    fraud_v1 = np.random.normal(loc=-1.2, scale=1.5, size=num_fraud)
    
    legit_v2 = np.random.normal(loc=0.0, scale=1.2, size=num_legit)
    fraud_v2 = np.random.normal(loc=1.5, scale=1.5, size=num_fraud)
    
    # Combine datasets
    df_legit = pd.DataFrame({
        "Amount": np.round(legit_amount, 2),
        "Hour_of_Day": np.round(legit_hours, 1),
        "Distance_from_Home_KM": np.round(legit_dist, 1),
        "Merchant_Category": legit_cat,
        "Transaction_Method": legit_method,
        "Card_Type": legit_cards,
        "Recent_24h_Transactions": legit_velocity,
        "Risk_Score_V1": np.round(legit_v1, 3),
        "Risk_Score_V2": np.round(legit_v2, 3),
        "Is_Fraud": 0
    })
    
    df_fraud = pd.DataFrame({
        "Amount": np.round(fraud_amount, 2),
        "Hour_of_Day": np.round(fraud_hours, 1),
        "Distance_from_Home_KM": np.round(fraud_dist, 1),
        "Merchant_Category": fraud_cat,
        "Transaction_Method": fraud_method,
        "Card_Type": fraud_cards,
        "Recent_24h_Transactions": fraud_velocity,
        "Risk_Score_V1": np.round(fraud_v1, 3),
        "Risk_Score_V2": np.round(fraud_v2, 3),
        "Is_Fraud": 1
    })
    
    df_combined = pd.concat([df_legit, df_fraud], ignore_index=True)
    # Shuffle randomly
    df_shuffled = df_combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df_shuffled


def load_raw_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """Loads raw transaction dataset or generates if missing."""
    file_path = data_path or RAW_DATA_PATH
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        print(f"Generating realistic imbalanced fraud dataset to {file_path}...")
        df = generate_fraud_dataset()
        df.to_csv(file_path, index=False)
        print(f"Generated {len(df)} transactions ({df['Is_Fraud'].mean():.2%} fraud rate).")
    else:
        df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans data and formats types."""
    df_clean = df.copy()
    df_clean["Is_Fraud"] = pd.to_numeric(df_clean["Is_Fraud"]).fillna(0).astype(int)
    return df_clean


def get_feature_target_split(df: pd.DataFrame, target_col: str = "Is_Fraud") -> Tuple[pd.DataFrame, pd.Series]:
    """Splits into feature matrix X and target y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


if __name__ == "__main__":
    df_raw = load_raw_data()
    print(f"Dataset shape: {df_raw.shape}")
    print(f"Fraud distribution:\n{df_raw['Is_Fraud'].value_counts(normalize=True)}")
    df_clean = clean_data(df_raw)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data to {PROCESSED_DATA_PATH}")
