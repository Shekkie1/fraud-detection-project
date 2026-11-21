import os
import json
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

DATA_PATH = "data/creditcard.csv"

METRICS_PATH = "reports/metrics.json"
CONFUSION_MATRIX_PATH = "reports/confusion_matrix.png"
ROC_CURVE_PATH = "reports/roc_curve.png"
FEATURE_IMPORTANCE_PATH = "reports/feature_importance.png"

# --------------------------------------------------
# SAFE METRICS LOADING
# --------------------------------------------------

default_metrics = {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0,
    "roc_auc": 0.0
}

metrics = default_metrics.copy()

if os.path.exists(METRICS_PATH):

    try:

        with open(METRICS_PATH, "r") as f:

            content = f.read().strip()

            if content:
                metrics = json.loads(content)
            else:
                st.warning(
                    "metrics.json exists but is empty."
                )

    except Exception as e:

        st.error(
            f"Unable to read metrics.json: {e}"
        )

else:

    st.warning(
        "metrics.json not found. Run train.py first."
    )

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

try:

    df = pd.read_csv(DATA_PATH)

except Exception as e:

    st.error(
        f"Unable to load dataset: {e}"
    )
    st.stop()

# --------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------

total_transactions = len(df)

fraud_transactions = int(
    df["Class"].sum()
)

legitimate_transactions = (
    total_transactions -
    fraud_transactions
)

fraud_rate = (
    fraud_transactions /
    total_transactions
) * 100

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title(
    "💳 Credit Card Fraud Detection Dashboard"
)

st.markdown(
    "### AI-Powered Credit Card Fraud Detection System"
)

st.divider()

# --------------------------------------------------
# EXECUTIVE KPIs
# --------------------------------------------------

st.subheader(
    "Executive KPIs"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

with col2:

    st.metric(
        "Fraud Cases",
        f"{fraud_transactions:,}"
    )

with col3:

    st.metric(
        "Legitimate Transactions",
        f"{legitimate_transactions:,}"
    )

with col4:

    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.3f}%"
    )

st.divider()

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.subheader(
    "Model Performance"
)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Accuracy",
    f"{metrics['accuracy']:.2%}"
)

c2.metric(
    "Precision",
    f"{metrics['precision']:.2%}"
)

c3.metric(
    "Recall",
    f"{metrics['recall']:.2%}"
)

c4.metric(
    "F1 Score",
    f"{metrics['f1_score']:.2%}"
)

c5.metric(
    "ROC AUC",
    f"{metrics['roc_auc']:.2%}"
)

st.divider()

# --------------------------------------------------
# FRAUD DISTRIBUTION
# --------------------------------------------------

st.subheader(
    "Fraud Distribution"
)

fraud_counts = (
    df["Class"]
    .value_counts()
    .reset_index()
)

fraud_counts.columns = [
    "Transaction Type",
    "Count"
]

fraud_counts[
    "Transaction Type"
] = fraud_counts[
    "Transaction Type"
].replace(
    {
        0: "Legitimate",
        1: "Fraud"
    }
)

fig1 = px.bar(
    fraud_counts,
    x="Transaction Type",
    y="Count",
    color="Transaction Type",
    title="Fraud vs Legitimate Transactions"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# --------------------------------------------------
# AMOUNT DISTRIBUTION
# --------------------------------------------------

st.subheader(
    "Transaction Amount Distribution"
)

fig2 = px.histogram(
    df,
    x="Amount",
    nbins=50,
    title="Transaction Amount Distribution"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

st.subheader(
    "Correlation Heatmap"
)

corr = df.corr()

fig3, ax = plt.subplots(
    figsize=(12, 8)
)

sns.heatmap(
    corr,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig3)

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

st.subheader(
    "Confusion Matrix"
)

if os.path.exists(
    CONFUSION_MATRIX_PATH
):

    st.image(
        CONFUSION_MATRIX_PATH
    )

else:

    st.info(
        "Confusion matrix not found."
    )

# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

st.subheader(
    "ROC Curve"
)

if os.path.exists(
    ROC_CURVE_PATH
):

    st.image(
        ROC_CURVE_PATH
    )

else:

    st.info(
        "ROC curve not found."
    )

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.subheader(
    "Feature Importance"
)

if os.path.exists(
    FEATURE_IMPORTANCE_PATH
):

    st.image(
        FEATURE_IMPORTANCE_PATH
    )

else:

    st.info(
        "Feature importance chart not found."
    )

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.subheader(
    "Dataset Overview"
)

st.dataframe(
    df.head(10),
    use_container_width=True
)

# --------------------------------------------------
# MODEL SUMMARY
# --------------------------------------------------

st.subheader(
    "Model Information"
)

st.info(
    f"""
    Model Type: XGBoost

    Total Transactions: {total_transactions:,}

    Fraud Cases: {fraud_transactions:,}

    Fraud Rate: {fraud_rate:.3f}%

    Target Metric: Recall

    Objective:
    Identify fraudulent credit card transactions
    while minimizing missed fraud cases.
    """
)