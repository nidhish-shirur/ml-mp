import streamlit as st
import time

# Page configuration
st.set_page_config(
    page_title="NEO Detection ML Project",
    page_icon="🌠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page
st.title("🌠 Near-Earth Object Detection - ML Project")
st.markdown("---")

# Loading indicator
with st.spinner("Loading project overview..."):
    time.sleep(2)  # Simulate loading time

st.markdown("""
## Project Overview

### Domain: **Space Science & Planetary Defense**

Near-Earth Objects (NEOs) are asteroids and comets whose orbits bring them into proximity with Earth. 
This project uses machine learning to predict whether asteroids are potentially hazardous to Earth, 
which has significant societal impact:

### 🌍 Real-World Applications & Societal Impact

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

### 📊 Dataset Information

- **Source:** NASA Near-Earth Object Dataset
- **Samples:** 905 asteroid close approaches
- **Features:** 27 properties including size, velocity, and distance
- **Target:** Potentially Hazardous Classification (True/False)

### 🎯 Project Objectives

This project demonstrates various machine learning techniques:
- Data Visualization & Exploration
- Regression Models (Linear, Polynomial)
- Classification Models (Decision Trees, SVM, Ensemble)
- Clustering Algorithms (K-Means, DBSCAN)
- Dimensionality Reduction (PCA, SVD)
- Comprehensive Model Comparison

### 📱 Navigation

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
st.info("👈 Select a page from the sidebar to begin exploring!")

# Performance tips
st.markdown("---")
st.subheader("⚡ Performance Tips")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **First Time Setup:**
    ```bash
    # Pre-train all models (run once)
    python train_all_models.py
    ```
    This takes 2-5 minutes but makes the app load instantly afterward!
    """)

with col2:
    st.markdown("""
    **Why pages load slowly:**
    - First visit loads 905 NEO samples
    - After that, data is cached automatically
    - Pre-trained models load in milliseconds
    - Subsequent visits are much faster ⚡
    """)

# Quick Stats
st.markdown("---")
st.subheader("📈 Quick Statistics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Asteroids", "905")
col2.metric("Features", "27")
col3.metric("Time Range", "1900-2187")
col4.metric("Hazardous %", "~100%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit, Scikit-learn, and NASA data</p>
    <p>Protecting Earth through Data Science 🛡️</p>
</div>
""", unsafe_allow_html=True)
