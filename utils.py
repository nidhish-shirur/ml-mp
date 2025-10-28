import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Increase cache TTL for better performance
@st.cache_data(ttl=7200, show_spinner=False)  # Cache for 2 hours, no spinner
def load_neo_data():
    """Load and preprocess NEO dataset (cached for 2 hours)"""
    df = pd.read_csv('neo_data.csv')
    
    # Select relevant numerical features (removing redundant units)
    feature_cols = [
        'Absolute Magnitude (H)',
        'Min Diameter (km)',
        'Max Diameter (km)',
        'Relative Velocity (km/s)',
        'Miss Distance (astronomical)',
        'Epoch Date Close Approach'
    ]
    
    X = df[feature_cols]
    y_hazardous = df['Is Potentially Hazardous']
    y_distance = df['Miss Distance (km)']
    y_velocity = df['Relative Velocity (km/s)']
    
    return X, y_hazardous, y_distance, y_velocity, df

@st.cache_resource(show_spinner=False)
def get_scaler(X):
    """Get fitted scaler (cached permanently)"""
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

@st.cache_data(show_spinner=False)
def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """Get train-test split (cached)"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

@st.cache_data(show_spinner=False)
def compute_metrics(_y_true, _y_pred, task='regression'):
    """Compute evaluation metrics (cached)"""
    if task == 'regression':
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        mse = mean_squared_error(_y_true, _y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(_y_true, _y_pred)
        mae = mean_absolute_error(_y_true, _y_pred)
        return {'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAE': mae}
    else:  # classification
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(_y_true, _y_pred)
        precision = precision_score(_y_true, _y_pred, average='weighted', zero_division=0)
        recall = recall_score(_y_true, _y_pred, average='weighted', zero_division=0)
        f1 = f1_score(_y_true, _y_pred, average='weighted', zero_division=0)
        return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
