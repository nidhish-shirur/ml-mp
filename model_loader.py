"""
Model loading utilities for NEO ML project
Handles loading pre-trained models with caching
"""

import streamlit as st
import pickle
import os
from config import MODEL_SUFFIX

@st.cache_resource
def load_model(model_name):
    """
    Load pre-trained model from disk (cached)
    
    Args:
        model_name: Base name of the model (e.g., 'linear_regression', 'decision_tree')
    
    Returns:
        Loaded model object or None if not found
    """
    # Add suffix if not already present
    if not model_name.endswith(MODEL_SUFFIX):
        filename = f"{model_name}{MODEL_SUFFIX}.pkl"
    else:
        filename = f"{model_name}.pkl"
    
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            st.warning(f"Model file not found: {filename}")
            return None
    except Exception as e:
        st.error(f"Error loading model {filename}: {str(e)}")
        return None

@st.cache_resource
def load_train_test_splits():
    """Load pre-computed train/test splits if available"""
    split_file = "train_test_splits_neo.pkl"
    
    if os.path.exists(split_file):
        try:
            with open(split_file, 'rb') as f:
                splits = pickle.load(f)
            return splits
        except Exception as e:
            st.warning(f"Could not load train/test splits: {str(e)}")
            return None
    else:
        return None

def check_model_exists(model_name):
    """Check if a model file exists"""
    if not model_name.endswith(MODEL_SUFFIX):
        filename = f"{model_name}{MODEL_SUFFIX}.pkl"
    else:
        filename = f"{model_name}.pkl"
    
    return os.path.exists(filename)

def list_available_models():
    """List all available pre-trained models"""
    models = []
    for file in os.listdir('.'):
        if file.endswith(f"{MODEL_SUFFIX}.pkl"):
            model_name = file.replace(f"{MODEL_SUFFIX}.pkl", "")
            models.append(model_name)
    return models

def get_model_status():
    """Get status of all expected models"""
    expected_models = [
        'linear_regression',
        'poly_regression',
        'poly_features',
        'decision_tree',
        'svm',
        'random_forest',
        'gradient_boosting',
        'kmeans',
        'dbscan',
        'pca',
        'svd'
    ]
    
    status = {}
    for model in expected_models:
        status[model] = check_model_exists(model)
    
    return status

def clear_model_cache():
    """Clear model cache - useful for debugging"""
    load_model.clear()
    load_train_test_splits.clear()
