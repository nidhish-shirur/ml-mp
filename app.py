"""
Main Streamlit application for NEO Detection ML Project
"""

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))
from data_loader import load_data_once, get_data_source, get_metadata
from config import PAGE_TITLE, PAGE_ICON

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Preload data on app start (cached, instant after first load)
data = load_data_once()
metadata = get_metadata()

# Show data source in sidebar
data_source = get_data_source()
st.sidebar.success(f"{data_source}")
st.sidebar.info(f"{metadata['n_samples']} asteroids loaded")

# Main page
st.title("Near-Earth Object Detection - ML Project")
st.markdown("---")

st.markdown("""
## Project Overview

### Domain: **Space Science & Planetary Defense**

Near-Earth Objects (NEOs) are asteroids and comets whose orbits bring them into proximity with Earth. 
This project uses machine learning to predict whether asteroids are potentially hazardous to Earth, 
which has significant societal impact:

### Real-World Applications & Societal Impact

1. **Planetary Defense**
   - Early detection of potentially hazardous asteroids
   - Can save millions of lives by enabling evacuation or deflection
   - Protects critical infrastructure from asteroid impacts
   - Enables timely mission planning for asteroid deflection

2. **Space Mission Planning**
   - Identifies suitable asteroids for mining operations
   - Helps plan flyby and sample return missions
   - Optimizes resources for space exploration
   - Supports commercial space ventures

3. **Scientific Research**
   - Understanding solar system formation
   - Studying asteroid composition and properties
   - Tracking near-Earth object populations
   - Contributing to planetary science knowledge

4. **Economic Benefits**
   - Asteroid mining could provide rare materials worth trillions
   - Reduces costs of space exploration
   - Creates new space industry opportunities
   - Protects global economy from catastrophic impacts

### Dataset Information

- **Source:** NASA Near-Earth Object Dataset
- **Samples:** 905 asteroid close approaches
- **Features:** 27 properties including size, velocity, and distance
- **Target:** Potentially Hazardous Classification (True/False)

### Project Objectives

This project demonstrates various machine learning techniques:
- Data Visualization & Exploration
- Regression Models (Linear, Polynomial)
- Classification Models (Decision Trees, SVM, Ensemble)
- Clustering Algorithms (K-Means, DBSCAN)
- Dimensionality Reduction (PCA, SVD)
- Comprehensive Model Comparison

### Navigation

Use the sidebar to navigate through different sections:
- **Home:** Project overview (current page)
- **Data Exploration:** Dataset analysis and visualizations
- **Regression Models:** Predict miss distance and velocity
- **Classification Models:** Classify hazardous asteroids
- **Clustering Analysis:** Group similar asteroids
- **Dimensionality Reduction:** Feature analysis
- **Model Comparison:** Performance metrics
- **Conclusion:** Key findings and recommendations
""")

st.markdown("---")
st.info("Select a page from the sidebar to begin exploring!")

# Performance optimizations
st.markdown("---")
st.subheader("Performance Optimizations Applied")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    **Lightning-Fast Loading:**
    - Data loaded once (instant!)
    - Shared across all pages
    - Cached automatically
    - Pages load in milliseconds
    """)

with col2:
    st.success("""
    **Model Optimization:**
    - Pre-trained models cached
    - Instant model loading
    - Modular architecture
    - Session state management
    """)

# Quick Stats
st.markdown("---")
st.subheader("Current Session Data")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Asteroids", f"{metadata['n_samples']:,}")
col2.metric("Features", metadata['n_features'])
col3.metric("Hazardous", f"{metadata['hazardous_count']}")
col4.metric("Load Time", "< 1 second")

# Show first-time setup tip
if not os.path.exists('linear_regression_neo.pkl'):
    st.warning("""
    **First Time Setup Recommended:**
    ```bash
    python train_all_models.py
    ```
    This pre-trains all models (2-5 minutes) for instant loading in all pages!
    """)
else:
    st.info("**Pro Tip:** All models are pre-trained and cached. Navigate freely - pages load instantly!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built using Streamlit, Scikit-learn, and NASA data</p>
    <p>Protecting Earth through Data Science</p>
</div>
""", unsafe_allow_html=True)
