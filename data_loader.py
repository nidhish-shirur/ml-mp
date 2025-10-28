"""
Centralized data loading module for NEO ML project
Handles data loading with caching and session state
"""

import streamlit as st
import pandas as pd
from config import SESSION_DATA_KEY, DATA_FILE, FEATURE_COLS, TARGET_HAZARDOUS, TARGET_DISTANCE, TARGET_VELOCITY

@st.cache_data(ttl=7200, show_spinner=False)  # Cache for 2 hours, no spinner
def load_neo_data():
    """Load NEO dataset from CSV (cached for 2 hours, instant after first load)"""
    try:
        df = pd.read_csv(DATA_FILE)
        return df
    except FileNotFoundError:
        st.error(f"Error: {DATA_FILE} not found!")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

def load_data_once():
    """Load data once and store in session state"""
    if SESSION_DATA_KEY not in st.session_state:
        df = load_neo_data()
        
        # Extract features and targets
        X = df[FEATURE_COLS]
        y_hazardous = df[TARGET_HAZARDOUS]
        y_distance = df[TARGET_DISTANCE]
        y_velocity = df[TARGET_VELOCITY]
        
        # Store in session state
        st.session_state[SESSION_DATA_KEY] = {
            'df': df,
            'X': X,
            'y_hazardous': y_hazardous,
            'y_distance': y_distance,
            'y_velocity': y_velocity,
            'metadata': {
                'n_samples': len(df),
                'n_features': len(FEATURE_COLS),
                'hazardous_count': y_hazardous.sum(),
                'features': FEATURE_COLS
            }
        }
    
    return st.session_state[SESSION_DATA_KEY]

def get_X():
    """Get feature matrix from session state"""
    data = load_data_once()
    return data['X']

def get_y_hazardous():
    """Get hazardous classification target"""
    data = load_data_once()
    return data['y_hazardous']

def get_y_distance():
    """Get miss distance target"""
    data = load_data_once()
    return data['y_distance']

def get_y_velocity():
    """Get velocity target"""
    data = load_data_once()
    return data['y_velocity']

def get_full_dataframe():
    """Get complete dataframe"""
    data = load_data_once()
    return data['df']

def get_metadata():
    """Get dataset metadata"""
    data = load_data_once()
    return data['metadata']

def clear_data_cache():
    """Clear data cache - useful for debugging"""
    if SESSION_DATA_KEY in st.session_state:
        del st.session_state[SESSION_DATA_KEY]
    load_neo_data.clear()

def get_data_source():
    """Return data source description"""
    return f"⚡ Data loaded from {DATA_FILE}"

def get_train_test_split(test_size=0.2, random_state=42):
    """Get cached train-test split for consistency across pages"""
    from sklearn.model_selection import train_test_split
    X = get_X()
    y_hazardous = get_y_hazardous()
    
    return train_test_split(X, y_hazardous, test_size=test_size, random_state=random_state)

@st.cache_data(show_spinner=False)
def get_scaled_data():
    """Get scaled feature matrix (cached)"""
    from sklearn.preprocessing import StandardScaler
    X = get_X()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
