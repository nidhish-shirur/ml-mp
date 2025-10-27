import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_neo_data():
    """Load and preprocess NEO dataset (cached)"""
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

@st.cache_resource
def get_scaler(X):
    """Get fitted scaler (cached)"""
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

@st.cache_resource
def load_or_train_model(model_name, X_train, y_train, model_type='classification'):
    """Load existing model or train new one (cached)"""
    model_path = f"{model_name}.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, True  # True means loaded from disk
    
    # Train model if not found
    if model_name == 'linear_regression':
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif model_name == 'decision_tree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == 'svm':
        from sklearn.svm import SVC
        model = SVC(random_state=42)
    elif model_name == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42, n_estimators=100)
    elif model_name == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(random_state=42, n_estimators=100)
    elif model_name == 'kmeans':
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=3, random_state=42)
    elif model_name == 'pca':
        from sklearn.decomposition import PCA
        model = PCA(n_components=3)
    else:
        return None, False
    
    model.fit(X_train, y_train)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    return model, False  # False means newly trained

@st.cache_data
def get_train_test_split(X, y, test_size=0.2, random_state=42):
    """Get train-test split (cached)"""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

@st.cache_data
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
