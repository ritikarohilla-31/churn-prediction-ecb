"""
Predictive Modeling and Risk Scoring for Bank Customer Churn
The European Central Bank — Interactive Streamlit Dashboard

Run with:  streamlit run app.py
Expects trained_models.pkl, model_results.json, feature_importance.csv,
shap_importance.csv, and European_Bank.csv in the same folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Churn Risk Intelligence", page_icon="📉", layout="wide")

NAVY = "#16263B"
TEAL = "#3FA7A0"
CORAL = "#D97757"
GOLD = "#C9A24B"
RISK_RED = "#B23A32"

@st.cache_resource
def load_artifacts():
    with open("trained_models.pkl", "rb") as f:
        data = pickle.load(f)
    with open("model_results.json") as f:
        results = json.load(f)
    fi = pd.read_csv("feature_importance.csv")
    return data, results, fi

data, results, fi = load_artifacts()
models = data["models"]
scaler = data["scaler"]
feature_names = data["feature_names"]
X_test = data["X_test"]
y_test = data["y_test"]

CHAMPION_NAME = "Gradient Boosting"
champion = models[CHAMPION_NAME]

st.sidebar.title("📉 Churn Risk Intelligence")
st.sidebar.caption("The European Central Bank")
model_choice = st.sidebar.selectbox("Model", list(models.keys()), index=list(models.keys()).index(CHAMPION_NAME))
active_model = models[model_choice]

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Performance (Test Set)**")
r = results[model_choice]
st.sidebar.metric("ROC-AUC", r["roc_auc"])
st.sidebar.metric("F1-Score", r["f1"])
st.sidebar.metric("Recall", r["recall"])
st.sidebar.metric("Precision", r["precision"])

st.title("Predictive Modeling & Risk Scoring for Bank Customer Churn")
st.caption("The European Central Bank — assigns churn probability before a customer leaves")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Churn Risk Calculator", "📊 Probability Distribution", "🔍 Feature Importance", "🧪 What-If Simulator"
])

# ----------------------------------------------------------------------------
# TAB 1: Churn Risk Calculator
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Score an Individual Customer")
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Credit Score", 350, 850, 650)
        age = st.slider("Age", 18, 92, 40)
        tenure = st.slider("Tenure (years)", 0, 10, 5)
    with c2:
        balance = st.number_input("Balance (€)", 0.0, 260000.0, 75000.0, step=1000.0)
        num_products = st.slider("Number of Products", 1, 4, 2)
        salary = st.number_input("Estimated Salary (€)", 10.0, 200000.0, 100000.0, step=1000.0)
    with c3:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        has_card = st.checkbox("Has Credit Card", value=True)
        is_active = st.checkbox("Is Active Member", value=True)

    def build_feature_row(credit_score, age, tenure, balance, num_products, has_card,
                           is_active, salary, geography, gender):
        row = {
            "CreditScore": credit_score, "Age": age, "Tenure": tenure, "Balance": balance,
            "NumOfProducts": num_products, "HasCrCard": int(has_card), "IsActiveMember": int(is_active),
            "EstimatedSalary": salary,
            "BalanceToSalaryRatio": balance / (salary if salary else 1),
            "ProductDensity": num_products / (tenure if tenure else 1),
            "EngagementProductInteraction": int(is_active) * num_products,
            "AgeTenureInteraction": age * tenure,
            "Geography_Germany": 1 if geography == "Germany" else 0,
            "Geography_Spain": 1 if geography == "Spain" else 0,
            "Gender_Male": 1 if gender == "Male" else 0,
        }
        return pd.DataFrame([row])[feature_names]

    input_row = build_feature_row(credit_score, age, tenure, balance, num_products, has_card,
                                    is_active, salary, geography, gender)

    if model_choice == "Logistic Regression":
        input_for_model = pd.DataFrame(scaler.transform(input_row), columns=feature_names)
    else:
        input_for_model = input_row

    proba = active_model.predict_proba(input_for_model)[0, 1]

    st.markdown("---")
    colA, colB = st.columns([1, 2])
    with colA:
        risk_label = "🔴 High Risk" if proba >= 0.5 else ("🟡 Medium Risk" if proba >= 0.25 else "🟢 Low Risk")
        st.metric("Churn Probability", f"{proba*100:.1f}%")
        st.markdown(f"### {risk_label}")
    with colB:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=proba*100,
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': NAVY},
                   'steps': [{'range': [0, 25], 'color': '#DCEFEC'},
                             {'range': [25, 50], 'color': '#F5EBD6'},
                             {'range': [50, 100], 'color': '#F8DAD6'}]},
            title={'text': "Churn Risk %"}))
        fig.update_layout(height=250, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 2: Probability Distribution
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Predicted Churn Probability Distribution (Test Set)")
    if model_choice == "Logistic Regression":
        Xte = data["X_test_scaled"]
    else:
        Xte = X_test
    test_proba = active_model.predict_proba(Xte)[:, 1]
    dist_df = pd.DataFrame({"probability": test_proba, "actual": y_test.map({0: "Retained", 1: "Churned"}).values})

    fig = px.histogram(dist_df, x="probability", color="actual", nbins=40, barmode="overlay",
                        color_discrete_map={"Retained": TEAL, "Churned": RISK_RED},
                        labels={"probability": "Predicted Churn Probability"})
    fig.update_layout(height=420, title=f"{model_choice} — Probability Distribution by Actual Outcome")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", results[model_choice]["accuracy"])
    c2.metric("ROC-AUC", results[model_choice]["roc_auc"])
    c3.metric("F1-Score", results[model_choice]["f1"])

    st.subheader("Model Comparison")
    comp_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    st.dataframe(comp_df.style.highlight_max(axis=0, color="#DCEFEC"), use_container_width=True)

    cm = np.array(results[model_choice]["confusion_matrix"])
    fig2 = px.imshow(cm, text_auto=True, x=["Predicted: Stay", "Predicted: Churn"],
                      y=["Actual: Stay", "Actual: Churn"], color_continuous_scale="Blues",
                      title=f"{model_choice} — Confusion Matrix")
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 3: Feature Importance Dashboard
# ----------------------------------------------------------------------------
with tab3:
    st.subheader(f"What Drives Churn Predictions? ({CHAMPION_NAME})")
    fig = px.bar(fi.head(12).sort_values("importance"), x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale=["#3FA7A0", "#B23A32"],
                 labels={"importance": "Feature Importance", "feature": ""})
    fig.update_layout(height=450, coloraxis_showscale=False, title="Top 12 Churn Drivers")
    st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **NumOfProducts** and **Age** are by far the strongest churn drivers — consistent with "
            "SHAP analysis. The relationship with NumOfProducts is non-linear: 2 products is the "
            "stickiest point, but 3-4 products is a strong churn-risk signal.")

    try:
        st.image("charts/09_shap_summary.png", caption="SHAP summary — feature impact and direction across all customers", use_container_width=True)
    except Exception:
        pass

# ----------------------------------------------------------------------------
# TAB 4: What-If Scenario Simulator
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("What-If Scenario Simulator")
    st.caption("Start from a baseline customer, then adjust engagement/product values to see churn probability shift.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Baseline customer**")
        base_active = st.checkbox("Active member (baseline)", value=False, key="base_active")
        base_products = st.slider("Products (baseline)", 1, 4, 1, key="base_products")
        base_age = st.slider("Age (baseline)", 18, 92, 45, key="base_age")
    with c2:
        st.markdown("**What-if scenario**")
        sim_active = st.checkbox("Active member (scenario)", value=True, key="sim_active")
        sim_products = st.slider("Products (scenario)", 1, 4, 2, key="sim_products")
        sim_age = st.slider("Age (scenario)", 18, 92, 45, key="sim_age")

    fixed = dict(credit_score=650, tenure=5, balance=75000.0, salary=100000.0,
                 geography="France", gender="Female", has_card=True)

    base_row = build_feature_row(fixed["credit_score"], base_age, fixed["tenure"], fixed["balance"],
                                   base_products, fixed["has_card"], base_active, fixed["salary"],
                                   fixed["geography"], fixed["gender"])
    sim_row = build_feature_row(fixed["credit_score"], sim_age, fixed["tenure"], fixed["balance"],
                                  sim_products, fixed["has_card"], sim_active, fixed["salary"],
                                  fixed["geography"], fixed["gender"])

    if model_choice == "Logistic Regression":
        base_input = pd.DataFrame(scaler.transform(base_row), columns=feature_names)
        sim_input = pd.DataFrame(scaler.transform(sim_row), columns=feature_names)
    else:
        base_input, sim_input = base_row, sim_row

    base_proba = active_model.predict_proba(base_input)[0, 1]
    sim_proba = active_model.predict_proba(sim_input)[0, 1]
    delta = sim_proba - base_proba

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline Churn Risk", f"{base_proba*100:.1f}%")
    c2.metric("Scenario Churn Risk", f"{sim_proba*100:.1f}%", delta=f"{delta*100:+.1f} pts", delta_color="inverse")
    c3.metric("Risk Change", f"{'Reduced' if delta<0 else 'Increased'}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Baseline", "Scenario"], y=[base_proba*100, sim_proba*100],
                          marker_color=[GOLD, TEAL if delta < 0 else RISK_RED],
                          text=[f"{base_proba*100:.1f}%", f"{sim_proba*100:.1f}%"], textposition="outside"))
    fig.update_layout(height=350, yaxis_title="Churn Probability (%)", title="Baseline vs Scenario")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Predictive Modeling & Risk Scoring for Bank Customer Churn · The European Central Bank · Unified Mentor Project")
