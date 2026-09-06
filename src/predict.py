"""Inference module for Credit Card Fraud Screening."""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fraud_pipeline.joblib")
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "model_metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(PROJECT_ROOT, "models", "feature_importances.json")

_MODEL = None
_METRICS = None
_FEATURE_IMPORTANCES = None


def load_fraud_model_assets():
    """Loads model and metadata assets."""
    global _MODEL, _METRICS, _FEATURE_IMPORTANCES
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run src/train.py first.")
        _MODEL = joblib.load(MODEL_PATH)
        
    if _METRICS is None and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            _METRICS = json.load(f)
            
    if _FEATURE_IMPORTANCES is None and os.path.exists(FEATURE_IMPORTANCE_PATH):
        with open(FEATURE_IMPORTANCE_PATH, "r") as f:
            _FEATURE_IMPORTANCES = json.load(f)
            
    return _MODEL, _METRICS, _FEATURE_IMPORTANCES


def analyze_transaction_risk_drivers(tx_dict: Dict[str, Any], prob: float) -> List[str]:
    """Generates explainable risk reasons for the transaction."""
    reasons = []
    
    amount = float(tx_dict.get("Amount", 0))
    hour = float(tx_dict.get("Hour_of_Day", 12))
    dist = float(tx_dict.get("Distance_from_Home_KM", 0))
    velocity = int(tx_dict.get("Recent_24h_Transactions", 0))
    cat = tx_dict.get("Merchant_Category", "")
    method = tx_dict.get("Transaction_Method", "")
    
    if (hour >= 23 or hour <= 5):
        reasons.append(f"⚠️ Late-night transaction timestamp ({int(hour):02d}:00 hours)")
    else:
        reasons.append(f"✅ Normal daytime transaction hours ({int(hour):02d}:00 hours)")
        
    if dist >= 200:
        reasons.append(f"⚠️ Geo-location mismatch ({dist:,.1f} KM from billing address)")
    else:
        reasons.append(f"✅ Proximity to home location ({dist:.1f} KM)")
        
    if amount > 1000:
        reasons.append(f"⚠️ Unusually high transaction value (${amount:,.2f})")
    elif amount < 5.0 and method == "Online POS":
        reasons.append(f"⚠️ Potential micro-authorization card test (${amount:.2f})")
        
    if velocity >= 5:
        reasons.append(f"⚠️ Elevated card velocity ({velocity} transactions in 24 hours)")
        
    if cat in ["Online Retail", "Luxury Goods", "Travel & Airlines"]:
        reasons.append(f"⚠️ High-risk merchant category ({cat})")
        
    if method == "Online POS" and dist > 100:
        reasons.append("⚠️ Remote unverified online gateway channel")
        
    return reasons


def screen_transaction(tx_data: Union[Dict[str, Any], pd.DataFrame], threshold: Optional[float] = None) -> Dict[str, Any]:
    """Evaluates transaction risk and returns decision recommendation."""
    model, metrics, _ = load_fraud_model_assets()
    
    opt_thresh = metrics.get("test_metrics", {}).get("optimal_threshold", 0.5) if metrics else 0.5
    active_threshold = threshold if threshold is not None else opt_thresh
    
    if isinstance(tx_data, dict):
        # Ensure PCA features exist if not provided
        if "Risk_Score_V1" not in tx_data:
            tx_data["Risk_Score_V1"] = -2.5 if float(tx_data.get("Amount", 0)) > 1000 and float(tx_data.get("Distance_from_Home_KM", 0)) > 200 else 0.1
        if "Risk_Score_V2" not in tx_data:
            tx_data["Risk_Score_V2"] = 3.0 if float(tx_data.get("Amount", 0)) > 1000 and float(tx_data.get("Distance_from_Home_KM", 0)) > 200 else -0.1
        df = pd.DataFrame([tx_data])
    else:
        df = tx_data.copy()
        if "Risk_Score_V1" not in df.columns:
            df["Risk_Score_V1"] = 0.1
        if "Risk_Score_V2" not in df.columns:
            df["Risk_Score_V2"] = -0.1
        
    prob = float(model.predict_proba(df)[0, 1])
    is_flagged = int(prob >= active_threshold)
    
    if prob >= 0.70:
        verdict = "🚨 DECLINE (High Fraud Risk)"
        action = "BLOCK TRANSACTION & NOTIFY CARDHOLDER"
        badge_color = "#EF4444"
    elif prob >= 0.35:
        verdict = "⚠️ STEP-UP AUTH (Moderate Risk)"
        action = "REQUIRE SMS/BIOMETRIC 2FA CHALLENGE"
        badge_color = "#F59E0B"
    else:
        verdict = "✅ APPROVED (Low Risk)"
        action = "INSTANT CLEARANCE"
        badge_color = "#10B981"
        
    drivers = []
    if isinstance(tx_data, dict):
        drivers = analyze_transaction_risk_drivers(tx_data, prob)
        
    return {
        "fraud_probability": prob,
        "fraud_probability_pct": f"{prob * 100:.1f}%",
        "verdict": verdict,
        "recommended_action": action,
        "badge_color": badge_color,
        "is_flagged": is_flagged,
        "threshold_used": active_threshold,
        "risk_drivers": drivers
    }


if __name__ == "__main__":
    sample_tx = {
        "Amount": 2450.00,
        "Hour_of_Day": 3.5,
        "Distance_from_Home_KM": 850.0,
        "Merchant_Category": "Online Retail",
        "Transaction_Method": "Online POS",
        "Card_Type": "Visa",
        "Recent_24h_Transactions": 7,
        "Risk_Score_V1": -3.2,
        "Risk_Score_V2": 3.8
    }
    
    try:
        res = screen_transaction(sample_tx)
        print("Sample Transaction Screening Result:")
        print(json.dumps(res, indent=2))
    except Exception as e:
        print(f"Screening test note: {e}")
