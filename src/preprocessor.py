"""Feature engineering and preprocessing pipeline for Credit Card Fraud Detection."""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-specific feature engineering for credit card transaction risk screening."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # 1. Late-night fraud window (11 PM - 5 AM)
        if "Hour_of_Day" in df.columns:
            df["Is_Night_Transaction"] = ((df["Hour_of_Day"] >= 23) | (df["Hour_of_Day"] <= 5)).astype(int)
        else:
            df["Is_Night_Transaction"] = 0

        # 2. Long-distance geo-anomaly (> 100 KM)
        if "Distance_from_Home_KM" in df.columns:
            df["Is_High_Distance"] = (df["Distance_from_Home_KM"] >= 100.0).astype(int)
        else:
            df["Is_High_Distance"] = 0

        # 3. Log-scaled amount (handling extreme positive skew)
        if "Amount" in df.columns:
            df["Amount_Log"] = np.log1p(np.maximum(0, df["Amount"]))
        else:
            df["Amount_Log"] = 0.0

        # 4. Rapid velocity flag (>= 5 transactions in last 24h)
        if "Recent_24h_Transactions" in df.columns:
            df["High_Velocity_Flag"] = (df["Recent_24h_Transactions"] >= 5).astype(int)
        else:
            df["High_Velocity_Flag"] = 0

        # 5. Elevated risk category indicator
        if "Merchant_Category" in df.columns:
            high_risk_cats = ["Online Retail", "Luxury Goods", "Travel & Airlines", "Electronics"]
            df["High_Risk_Category"] = df["Merchant_Category"].isin(high_risk_cats).astype(int)
        else:
            df["High_Risk_Category"] = 0

        return df


def get_fraud_preprocessor_pipeline() -> Pipeline:
    """Builds a scikit-learn preprocessing ColumnTransformer pipeline using RobustScaler."""
    numeric_features = [
        "Amount", "Amount_Log", "Hour_of_Day", "Distance_from_Home_KM",
        "Recent_24h_Transactions", "Risk_Score_V1", "Risk_Score_V2",
        "Is_Night_Transaction", "Is_High_Distance", "High_Velocity_Flag", "High_Risk_Category"
    ]

    categorical_features = ["Merchant_Category", "Transaction_Method", "Card_Type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline(steps=[
        ("feature_engineering", FraudFeatureEngineer()),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline


def get_feature_names_out(pipeline: Pipeline, input_df: pd.DataFrame) -> list:
    """Extracts output feature names after preprocessing."""
    fe_df = pipeline.named_steps["feature_engineering"].transform(input_df)
    preprocessor = pipeline.named_steps["preprocessor"]
    
    if not hasattr(preprocessor, "transformers_"):
        preprocessor.fit(fe_df)
        
    feature_names = [
        "Amount", "Amount_Log", "Hour_of_Day", "Distance_from_Home_KM",
        "Recent_24h_Transactions", "Risk_Score_V1", "Risk_Score_V2",
        "Is_Night_Transaction", "Is_High_Distance", "High_Velocity_Flag", "High_Risk_Category"
    ]
    
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(["Merchant_Category", "Transaction_Method", "Card_Type"])
    feature_names.extend(cat_names.tolist())
    
    return feature_names
