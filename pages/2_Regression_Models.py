import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go
from utils import load_neo_data

st.set_page_config(page_title="Regression Models", layout="wide")

st.title("Regression Models for NEO Prediction")

# Load data
@st.cache_data(show_spinner=False)
def load_data():
    return load_neo_data()

with st.spinner("Loading dataset..."):
    X, y_hazardous, y_distance, y_velocity, df = load_data()

st.markdown("""
## Objective: Predict Asteroid Properties

**Why is this important?**
- Predicting miss distance helps assess collision risk
- Velocity prediction aids in impact energy estimation
- Early prediction enables timely response measures
- Supports mission planning for asteroid deflection or mining

**Societal Impact:**
- Protects Earth from potential catastrophic impacts
- Saves billions in damage prevention
- Enables strategic planning for planetary defense
- Supports scientific research and space exploration
""")

# Model Selection
st.sidebar.subheader("Model Configuration")
target = st.sidebar.selectbox("Prediction Target", 
                              ["Miss Distance (km)", "Relative Velocity (km/s)"])
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20) / 100
random_state = st.sidebar.number_input("Random State", 0, 100, 42)

# Select target
y = y_distance if target == "Miss Distance (km)" else y_velocity

# Train-test split
@st.cache_data(show_spinner=False)
def get_train_test_split(_X, _y, test_size, random_state):
    return train_test_split(_X, _y, test_size=test_size, random_state=random_state)

X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size, random_state)

st.subheader("Data Split Information")
col1, col2 = st.columns(2)
col1.metric("Training Samples", X_train.shape[0])
col2.metric("Testing Samples", X_test.shape[0])

# Linear Regression
st.markdown("---")
st.subheader("1. Linear Regression")

st.markdown("""
**Algorithm Explanation:**
- Assumes a linear relationship between features and target
- Finds the best-fitting line that minimizes prediction errors
- Fast and interpretable
- Works well when relationships are approximately linear

**Formula:** y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

**Use Case:** Quick baseline for asteroid property prediction
""")

lr_exists = os.path.exists('linear_regression_neo.pkl')
if lr_exists:
    st.info("Pre-trained model found! Will load instantly.")

if st.button("Train/Load Linear Regression", key="lr"):
    with st.spinner("Training Linear Regression..."):
        if lr_exists:
            with open('linear_regression_neo.pkl', 'rb') as f:
                lr = pickle.load(f)
            st.success("⚡ Loaded from disk!")
        else:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            pickle.dump(lr, open('linear_regression_neo.pkl', 'wb'))
        
        y_pred_lr = lr.predict(X_test)
        
        # Metrics
        mse_lr = mean_squared_error(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mse_lr)
        r2_lr = r2_score(y_test, y_pred_lr)
        mae_lr = mean_absolute_error(y_test, y_pred_lr)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{rmse_lr:.2e}")
        col2.metric("R² Score", f"{r2_lr:.4f}")
        col3.metric("MAE", f"{mae_lr:.2e}")
        col4.metric("MSE", f"{mse_lr:.2e}")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred_lr,
                                mode='markers', name='Predictions',
                                marker=dict(size=5, opacity=0.5, color='blue')))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                y=[y_test.min(), y_test.max()],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        fig.update_layout(title=f"Linear Regression: Actual vs Predicted {target}",
                         xaxis_title=f"Actual {target}",
                         yaxis_title=f"Predicted {target}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Coefficients
        st.subheader("Feature Coefficients")
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': lr.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

# Polynomial Regression
st.markdown("---")
st.subheader("2. Polynomial Regression")

st.markdown("""
**Algorithm Explanation:**
- Captures non-linear relationships by creating polynomial features
- More flexible than linear regression
- Can model curved relationships
- Degree 2 creates quadratic terms (x², x₁×x₂, etc.)

**Formula:** y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
""")

poly_degree = st.slider("Polynomial Degree", 2, 4, 2)

poly_exists = os.path.exists('poly_regression_neo.pkl') and poly_degree == 2
if poly_exists:
    st.info("Pre-trained model (degree=2) found! Will load instantly.")

if st.button("Train/Load Polynomial Regression", key="poly"):
    with st.spinner(f"Processing degree={poly_degree}..."):
        if poly_exists:
            # Load pre-trained model
            with open('poly_regression_neo.pkl', 'rb') as f:
                lr_poly = pickle.load(f)
            with open('poly_features_neo.pkl', 'rb') as f:
                poly = pickle.load(f)
            X_train_poly = poly.transform(X_train)
            st.success("⚡ Loaded from disk!")
        else:
            # Train new model
            poly = PolynomialFeatures(degree=poly_degree)
            X_train_poly = poly.fit_transform(X_train)
            lr_poly = LinearRegression()
            lr_poly.fit(X_train_poly, y_train)
            
            if poly_degree == 2:
                pickle.dump(lr_poly, open('poly_regression_neo.pkl', 'wb'))
                pickle.dump(poly, open('poly_features_neo.pkl', 'wb'))
        
        X_test_poly = poly.transform(X_test)
        y_pred_poly = lr_poly.predict(X_test_poly)
        
        # Metrics
        mse_poly = mean_squared_error(y_test, y_pred_poly)
        rmse_poly = np.sqrt(mse_poly)
        r2_poly = r2_score(y_test, y_pred_poly)
        mae_poly = mean_absolute_error(y_test, y_pred_poly)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{rmse_poly:.2e}", f"{((rmse_poly-rmse_lr)/rmse_lr*100):.1f}%" if 'rmse_lr' in locals() else None)
        col2.metric("R² Score", f"{r2_poly:.4f}", f"{((r2_poly-r2_lr)/r2_lr*100):.1f}%" if 'r2_lr' in locals() else None)
        col3.metric("MAE", f"{mae_poly:.2e}")
        col4.metric("MSE", f"{mse_poly:.2e}")
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred_poly,
                                mode='markers', name='Predictions',
                                marker=dict(size=5, opacity=0.5, color='green')))
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                y=[y_test.min(), y_test.max()],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        fig.update_layout(title=f"Polynomial Regression (degree={poly_degree}): Actual vs Predicted {target}",
                         xaxis_title=f"Actual {target}",
                         yaxis_title=f"Predicted {target}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual Plot
        residuals = y_test - y_pred_poly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_pred_poly, y=residuals,
                                mode='markers',
                                marker=dict(size=5, opacity=0.5, color='purple')))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(title="Residual Plot",
                         xaxis_title="Predicted Values",
                         yaxis_title="Residuals")
        st.plotly_chart(fig, use_container_width=True)

# Conclusions
st.markdown("---")
st.subheader("Regression Conclusions")
st.markdown("""
**Model Comparison:**
- **Linear Regression:** Fast, interpretable, good baseline
- **Polynomial Regression:** Better accuracy, captures non-linear patterns

**Key Findings:**
- Features have varying importance for prediction
- Non-linear relationships exist in asteroid data
- Higher polynomial degrees may overfit

**Impact:**
- Accurate predictions enable early warning systems
- Helps prioritize which asteroids to monitor closely
- Supports mission planning for deflection or study
    - Critical for planetary defense strategies
    """)

st.info("**Next Steps:** Try Classification models to predict hazardous asteroids!")