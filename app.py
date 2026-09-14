"""Streamlit Web Application: Real-Time Credit Card Fraud Screening & Anomaly Intelligence."""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure src in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from predict import screen_transaction, load_fraud_model_assets

try:
    st.set_page_config(
        page_title="Credit Card Fraud Screener & Anomaly Intelligence",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception:
    pass

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


def create_fraud_gauge(prob: float):
    """Creates a gauge chart for fraud risk probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 36, 'color': "#1E293B"}},
        title={'text': "Fraud Anomaly Score", 'font': {'size': 18, 'color': "#475569"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
            'bar': {'color': "#DC2626" if prob >= 0.70 else ("#D97706" if prob >= 0.35 else "#059669"), 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 35], 'color': "#D1FAE5"},
                {'range': [35, 70], 'color': "#FEF3C7"},
                {'range': [70, 100], 'color': "#FEE2E2"}
            ]
        }
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
    return fig


@st.cache_resource
def get_fraud_assets():
    return load_fraud_model_assets()


def main():
    st.markdown('<div class="main-header">🛡️ Real-Time Credit Card Fraud Screener</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cost-Sensitive ML with XGBoost, Extreme Imbalance Handling & Explainable Risk Scoring</div>', unsafe_allow_html=True)

    try:
        model, metrics, feat_importances = get_fraud_assets()
    except Exception as e:
        st.error(f"⚠️ Model artifacts not found. Please run training script first: `python src/train.py`\n\nError: {e}")
        st.stop()

    st.sidebar.header("💳 Transaction Profile Builder")
    
    preset = st.sidebar.selectbox(
        "Load Preset Scenario",
        [
            "Custom Transaction",
            "🚨 Compromised Card ($2,850 Online at 3 AM, 900 KM Away)",
            "☕ Legitimate Coffee Purchase ($4.75 Morning In-Person Chip)",
            "⚠️ Suspicious Micro-Test Charge ($1.50 Online POS, High Velocity)",
            "✈️ Travel Airline Booking ($680 Daytime Online)"
        ]
    )

    defaults = {
        "Amount": 45.00,
        "Hour_of_Day": 14.0,
        "Distance_from_Home_KM": 8.5,
        "Merchant_Category": "Dining",
        "Transaction_Method": "In-Person Chip",
        "Card_Type": "Visa",
        "Recent_24h_Transactions": 2,
        "Risk_Score_V1": 0.1,
        "Risk_Score_V2": -0.2
    }

    if preset == "🚨 Compromised Card ($2,850 Online at 3 AM, 900 KM Away)":
        defaults.update({
            "Amount": 2850.00,
            "Hour_of_Day": 3.0,
            "Distance_from_Home_KM": 900.0,
            "Merchant_Category": "Online Retail",
            "Transaction_Method": "Online POS",
            "Recent_24h_Transactions": 8,
            "Risk_Score_V1": -3.5,
            "Risk_Score_V2": 4.1
        })
    elif preset == "☕ Legitimate Coffee Purchase ($4.75 Morning In-Person Chip)":
        defaults.update({
            "Amount": 4.75,
            "Hour_of_Day": 8.5,
            "Distance_from_Home_KM": 2.0,
            "Merchant_Category": "Dining",
            "Transaction_Method": "In-Person Chip",
            "Recent_24h_Transactions": 1,
            "Risk_Score_V1": 0.3,
            "Risk_Score_V2": -0.4
        })
    elif preset == "⚠️ Suspicious Micro-Test Charge ($1.50 Online POS, High Velocity)":
        defaults.update({
            "Amount": 1.50,
            "Hour_of_Day": 2.5,
            "Distance_from_Home_KM": 450.0,
            "Merchant_Category": "Online Retail",
            "Transaction_Method": "Online POS",
            "Recent_24h_Transactions": 6,
            "Risk_Score_V1": -2.1,
            "Risk_Score_V2": 2.6
        })
    elif preset == "✈️ Travel Airline Booking ($680 Daytime Online)":
        defaults.update({
            "Amount": 680.00,
            "Hour_of_Day": 15.0,
            "Distance_from_Home_KM": 55.0,
            "Merchant_Category": "Travel & Airlines",
            "Transaction_Method": "Online POS",
            "Recent_24h_Transactions": 3,
            "Risk_Score_V1": -0.8,
            "Risk_Score_V2": 1.1
        })

    with st.sidebar.expander("1. Transaction Details", expanded=True):
        amount = st.number_input("Transaction Amount ($)", min_value=0.5, max_value=10000.0, value=float(defaults["Amount"]), step=5.0)
        hour = st.slider("Hour of Day (24h)", min_value=0, max_value=23, value=int(defaults["Hour_of_Day"]))
        distance = st.number_input("Distance from Billing Address (KM)", min_value=0.0, max_value=5000.0, value=float(defaults["Distance_from_Home_KM"]), step=10.0)

    with st.sidebar.expander("2. Merchant & Channel", expanded=True):
        categories = ["Grocery", "Online Retail", "Dining", "Travel & Airlines", "Electronics", "Gas Station", "Luxury Goods", "ATM Withdrawal"]
        category = st.selectbox("Merchant Category", categories, index=categories.index(defaults["Merchant_Category"]))
        
        methods = ["In-Person Chip", "Online POS", "Contactless Tap", "ATM Cash"]
        method = st.selectbox("Payment Channel", methods, index=methods.index(defaults["Transaction_Method"]))
        
        card = st.selectbox("Card Brand", ["Visa", "Mastercard", "Amex", "Discover"], index=["Visa", "Mastercard", "Amex", "Discover"].index(defaults["Card_Type"]))
        velocity = st.slider("Card Velocity (Transactions in last 24h)", min_value=1, max_value=20, value=int(defaults["Recent_24h_Transactions"]))

    with st.sidebar.expander("3. Advanced Risk Parameters", expanded=False):
        risk_v1 = st.slider("Risk Score V1 (PCA Latent)", min_value=-5.0, max_value=5.0, value=float(defaults["Risk_Score_V1"]), step=0.1,
                           help="Anonymized behavioral risk factor. Negative = higher fraud signal.")
        risk_v2 = st.slider("Risk Score V2 (PCA Latent)", min_value=-5.0, max_value=5.0, value=float(defaults["Risk_Score_V2"]), step=0.1,
                           help="Secondary behavioral pattern. Positive = higher fraud signal.")

    tx_payload = {
        "Amount": amount,
        "Hour_of_Day": hour,
        "Distance_from_Home_KM": distance,
        "Merchant_Category": category,
        "Transaction_Method": method,
        "Card_Type": card,
        "Recent_24h_Transactions": velocity,
        "Risk_Score_V1": risk_v1,
        "Risk_Score_V2": risk_v2
    }

    tab1, tab2, tab3 = st.tabs(["⚡ Live Transaction Risk Screening", "📊 Imbalance Model Leaderboard & PR-AUC", "📁 Batch Transaction Screening"])

    with tab1:
        res = screen_transaction(tx_payload)
        prob = res["fraud_probability"]
        verdict = res["verdict"]
        action = res["recommended_action"]
        color = res["badge_color"]
        drivers = res["risk_drivers"]

        col1, col2 = st.columns([1.2, 1.8])

        with col1:
            st.plotly_chart(create_fraud_gauge(prob), use_container_width=True)
            
            st.markdown(f"""
            <div style="background-color: {color}; border-radius: 8px; padding: 12px; text-align: center; color: white; font-weight: 700; font-size: 1.15rem; margin-bottom: 12px;">
                {verdict}
            </div>
            <div style="text-align: center; color: #475569; font-size: 0.95rem;">
                <b>Action:</b> {action}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("📋 Transaction Risk Breakdown")
            tx_summary_df = pd.DataFrame([
                {"Attribute": "Amount", "Value": f"${amount:,.2f}"},
                {"Attribute": "Timestamp", "Value": f"{hour:02d}:00 hours ({'Night Window' if hour >= 23 or hour <= 5 else 'Daytime'})"},
                {"Attribute": "Geo Distance", "Value": f"{distance:,.1f} KM from home address"},
                {"Attribute": "Merchant & Method", "Value": f"{category} via {method}"},
                {"Attribute": "24h Velocity", "Value": f"{velocity} transactions"}
            ])
            st.dataframe(tx_summary_df, hide_index=True, use_container_width=True)

            st.subheader("🔍 Risk Factor Indicators")
            for driver in drivers:
                if "⚠️" in driver:
                    st.warning(driver)
                else:
                    st.success(driver)

    with tab2:
        st.subheader("📈 Model Benchmarks under Extreme Class Imbalance (~2.0% Fraud)")
        if metrics and "cv_results" in metrics:
            cv_df = pd.DataFrame(metrics["cv_results"])
            st.dataframe(
                cv_df.style.highlight_max(subset=["PR-AUC (Avg Precision)", "ROC-AUC", "Recall (Fraud Catch Rate)", "F1-Score"], color="#D1FAE5")
                .format({
                    "PR-AUC (Avg Precision)": "{:.4f}",
                    "ROC-AUC": "{:.4f}",
                    "Recall (Fraud Catch Rate)": "{:.2%}",
                    "Precision": "{:.2%}",
                    "F1-Score": "{:.4f}"
                }),
                use_container_width=True
            )

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if metrics and "test_metrics" in metrics:
                tm = metrics["test_metrics"]
                st.metric("Test PR-AUC (Average Precision)", f"{tm['pr_auc']:.4f}")
                st.metric("Fraud Detection Recall", f"{tm['recall']*100:.1f}%", help="Percentage of all fraudulent attacks caught")
                st.metric("Precision", f"{tm['precision']*100:.1f}%")
                cm = tm.get("confusion_matrix", [[0, 0], [0, 0]])
                st.write(f"**Test Confusion Matrix:** True Negatives: `{cm[0][0]}`, False Positives: `{cm[0][1]}`, False Negatives: `{cm[1][0]}`, True Positives: `{cm[1][1]}`")

        with col_m2:
            st.subheader("Top Predictive Risk Features (XGBoost)")
            if feat_importances:
                top_feats = list(feat_importances.items())[:10]
                feat_df = pd.DataFrame(top_feats, columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
                fig_imp = px.bar(
                    feat_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Predictive Weight in Fraud Decision Engine",
                    color="Importance",
                    color_continuous_scale="Reds"
                )
                fig_imp.update_layout(height=380, showlegend=False)
                st.plotly_chart(fig_imp, use_container_width=True)

        if metrics and "test_metrics" in metrics:
            tm = metrics["test_metrics"]
            st.divider()
            st.subheader("🎯 Precision-Recall Tradeoff & Operational Threshold Tuning")
            st.markdown("""
            In high-stakes fraud detection, **Precision** and **Recall** are inversely coupled:
            * **High Recall (Lower Threshold)**: Catches almost all fraud attacks, but causes verification friction for legitimate cardholders.
            * **High Precision (Higher Threshold)**: Minimizes customer friction, but permits stealthier fraudulent transactions to pass through.
            """)
            
            # Load PR curve from model metrics (computed during training)
            pr_curve = metrics.get("pr_curve", None)
            if pr_curve:
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=pr_curve["recalls"],
                    y=pr_curve["precisions"],
                    mode="lines+markers",
                    name="PR Curve",
                    line=dict(color="#4F46E5", width=2)
                ))
                fig_pr.update_layout(
                    title="Precision-Recall Curve (from Test Evaluation)",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    template="plotly_white"
                )
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                st.info("PR Curve data not available. Retrain the model to generate it.")

    with tab3:
        st.subheader("📁 Batch Transaction Log Screening")
        uploaded_file = st.file_uploader("Upload CSV transaction logs", type=["csv"], key="fraud_batch_csv")
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(batch_df)} transaction records.")
            if st.button("Screen Batch Transactions"):
                with st.spinner("Analyzing risk scores..."):
                    if "Risk_Score_V1" not in batch_df.columns:
                        batch_df["Risk_Score_V1"] = 0.1
                    if "Risk_Score_V2" not in batch_df.columns:
                        batch_df["Risk_Score_V2"] = -0.1
                    try:
                        probs = model.predict_proba(batch_df)[:, 1]
                    except Exception as e:
                        st.error(f"Screening failed: {e}. Ensure CSV has columns: Amount, Hour_of_Day, Distance_from_Home_KM, Merchant_Category, Transaction_Method, Card_Type, Recent_24h_Transactions")
                        st.stop()
                    batch_df["Fraud_Probability"] = np.round(probs, 4)
                    batch_df["Decision"] = ["BLOCK" if p >= 0.7 else ("2FA_CHALLENGE" if p >= 0.35 else "APPROVE") for p in probs]
                    
                    st.session_state["batch_results"] = batch_df
            
            if "batch_results" in st.session_state:
                res_df = st.session_state["batch_results"]
                st.success("Batch Screening Complete!")
                st.dataframe(res_df[["Decision", "Fraud_Probability", "Amount", "Merchant_Category", "Transaction_Method"]].head(20))
                
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Screened Transactions CSV", csv, "screened_transactions_output.csv", "text/csv")


if __name__ == "__main__":
    main()
