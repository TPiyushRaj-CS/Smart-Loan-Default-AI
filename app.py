
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Smart Loan Default AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "random_forest.joblib"
RESULTS_PATH = BASE_DIR / "outputs" / "model_comparison.csv"
THRESHOLD_PATH = BASE_DIR / "outputs" / "threshold_comparison.csv"

# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F7FB;
    }
    [data-testid="stSidebar"] {
        background-color: #17243A;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }
    .hero {
        background: linear-gradient(120deg, #17243A, #285B88);
        padding: 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
    }
    .hero h1 {
        color: white;
        margin-bottom: 8px;
    }
    .hero p {
        color: #E5EDF7;
        margin-bottom: 0;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E4E9F2;
        padding: 16px;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Load saved model
# --------------------------------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH}")
    st.info(
        "Run the model-saving cell in your notebook first."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
st.sidebar.title("SMART LOAN AI")
st.sidebar.caption("Machine Learning Risk Analytics")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Default Prediction",
        "Model Performance",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Educational ML prototype")
st.sidebar.caption("Not a credit approval system")

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Smart Loan Default AI</h1>
        <p>
        Explore credit default risk predictions using
        machine learning, model metrics, and interactive inputs.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Overview
# --------------------------------------------------
if page == "Overview":
    st.subheader("Project Overview")
    st.write(
        "This project uses historical credit-card customer "
        "data to estimate the probability of payment default "
        "in the following month."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Training Model", "Random Forest")
    c2.metric("Input Features", "23")
    c3.metric("Saved Model", "Available")

    st.subheader("What you can explore")
    left, middle, right = st.columns(3)

    with left:
        st.markdown("### Risk Prediction")
        st.write(
            "Enter customer financial and repayment "
            "information to generate a model estimate."
        )

    with middle:
        st.markdown("### Model Evaluation")
        st.write(
            "Review accuracy, precision, recall, F1-score, "
            "and ROC-AUC from your saved comparison results."
        )

    with right:
        st.markdown("### Threshold Analysis")
        st.write(
            "Examine how changing a classification threshold "
            "affects precision and recall."
        )

    st.warning(
        "This is an educational prototype. Its predictions "
        "must not be used to approve or reject real loan "
        "applications."
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
elif page == "Default Prediction":
    st.subheader("Customer Default Prediction")
    st.write(
        "Enter values using the same feature definitions "
        "and units as the training dataset."
    )

    # The canonical UCI dataset has these 23 predictors.
    feature_names = [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
        "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]

    # Ensure the form follows the model's actual feature order.
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)

    with st.form("prediction_form"):
        st.markdown("#### Customer Profile")
        a, b, c = st.columns(3)

        with a:
            limit_bal = st.number_input(
                "Credit limit (NT dollars)",
                min_value=0, max_value=2000000,
                value=50000, step=10000,
            )
            age = st.number_input(
                "Age", min_value=18, max_value=100,
                value=30,
            )

        with b:
            sex = st.selectbox(
                "Sex code (dataset encoding)",
                [1, 2],
                format_func=lambda x: {
                    1: "1 — Male (dataset code)",
                    2: "2 — Female (dataset code)",
                }[x],
            )
            education = st.selectbox(
                "Education code",
                [1, 2, 3, 4, 5, 6, 0],
                help="Use the original dataset's numeric codes.",
            )

        with c:
            marriage = st.selectbox(
                "Marriage code",
                [1, 2, 3, 0],
            )
            st.caption(
                "Codes are dataset categories, not a judgment "
                "about an individual."
            )

        st.markdown("#### Repayment Status")
        st.caption(
            "Enter the dataset's PAY values. Negative values "
            "and zero have dataset-specific meanings."
        )

        payment_cols = st.columns(3)
        pay_values = {}
        for i, col in enumerate(
            ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        ):
            with payment_cols[i % 3]:
                pay_values[col] = st.number_input(
                    f"{col}",
                    min_value=-2, max_value=8,
                    value=0, step=1,
                )

        st.markdown("#### Bills and Previous Payments")
        bill_cols = st.columns(3)
        bill_values = {}
        for i in range(1, 7):
            with bill_cols[(i - 1) % 3]:
                bill_values[f"BILL_AMT{i}"] = st.number_input(
                    f"Bill amount {i} (NT dollars)",
                    min_value=-1000000, max_value=10000000,
                    value=0, step=1000,
                )

        payment_amount_cols = st.columns(3)
        amount_values = {}
        for i in range(1, 7):
            with payment_amount_cols[(i - 1) % 3]:
                amount_values[f"PAY_AMT{i}"] = st.number_input(
                    f"Previous payment {i} (NT dollars)",
                    min_value=0, max_value=10000000,
                    value=0, step=1000,
                )

        submitted = st.form_submit_button(
            "Estimate Default Risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        values = {
            "LIMIT_BAL": limit_bal,
            "SEX": sex,
            "EDUCATION": education,
            "MARRIAGE": marriage,
            "AGE": age,
            **pay_values,
            **bill_values,
            **amount_values,
        }

        input_df = pd.DataFrame(
            [[values[name] for name in feature_names]],
            columns=feature_names,
        )

        try:
            probability = float(model.predict_proba(input_df)[0, 1])

            threshold = 0.5
            if THRESHOLD_PATH.exists():
                thresholds = pd.read_csv(THRESHOLD_PATH)
                if (
                    "Threshold" in thresholds.columns
                    and "F1-score" in thresholds.columns
                    and not thresholds.empty
                ):
                    best_idx = thresholds["F1-score"].idxmax()
                    threshold = float(
                        thresholds.loc[best_idx, "Threshold"]
                    )

            st.markdown("### Prediction Result")
            m1, m2 = st.columns(2)
            m1.metric("Estimated Default Probability",
                      f"{probability:.1%}")
            m2.metric("Classification Threshold",
                      f"{threshold:.2f}")

            
            st.progress(
                min(max(probability, 0.0), 1.0),
                text=f"Estimated default probability: {probability:.1%}",
            )

            if probability >= threshold:
                st.error(
                    "Higher predicted default risk",
                    icon="⚠️",
                )
                st.write(
                    "The estimated probability is at or above "
                    "the selected classification threshold."
                )
            else:
                st.success(
                    "Lower predicted default risk",
                    icon="✓",
                )
                st.write(
                    "The estimated probability is below "
                    "the selected classification threshold."
                )

            st.caption(
                "The probability is a model estimate, not a "
                "guarantee or a measure of model confidence. "
                "This educational prototype must not be used "
                "to make real credit or lending decisions."
            )


            st.caption(
                "This is a model estimate, not a guarantee of "
                "future behavior or a real lending decision. "
                "The selected threshold is experimental."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.info(
                "Check that the model was trained with the "
                "23 expected feature columns and compatible "
                "scikit-learn versions."
            )

# --------------------------------------------------
# Model performance
# --------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model Performance")

    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH)
        st.dataframe(
            results.round(4),
            use_container_width=True,
            hide_index=True,
        )

        numeric_cols = [
            col for col in
            ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
            if col in results.columns
        ]

        if numeric_cols:
            metric = st.selectbox(
                "Choose a metric to compare",
                numeric_cols,
            )
            if "Model" in results.columns:
                chart_data = results.set_index("Model")[[metric]]
                st.bar_chart(chart_data)
            else:
                st.info(
                    "Your CSV needs a 'Model' column to label "
                    "the model comparison chart."
                )
    else:
        st.info(
            "model_comparison.csv was not found in the outputs folder."
        )

    st.subheader("Threshold Analysis")
    if THRESHOLD_PATH.exists():
        threshold_df = pd.read_csv(THRESHOLD_PATH)
        st.dataframe(
            threshold_df.round(3),
            use_container_width=True,
            hide_index=True,
        )

        available_metrics = [
            col for col in ["Precision", "Recall", "F1-score"]
            if col in threshold_df.columns
        ]

        if "Threshold" in threshold_df.columns and available_metrics:
            plot_df = threshold_df.set_index("Threshold")[
                available_metrics
            ]
            st.line_chart(plot_df)
    else:
        st.info(
            "threshold_comparison.csv was not found in outputs."
        )

st.divider()
st.caption(
    "Smart Loan Default AI | Educational machine learning project"
)
