import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils import load_neo_data

st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")

st.title("📊 Near-Earth Object Data Exploration")

# Load data
@st.cache_data
def load_data():
    X, y_hazardous, y_distance, y_velocity, df = load_neo_data()
    return X, y_hazardous, y_distance, y_velocity, df

X, y_hazardous, y_distance, y_velocity, df = load_data()

st.markdown("""
## Dataset Overview

This dataset contains information about Near-Earth Objects (NEOs) - asteroids and comets 
that pass close to Earth's orbit. Understanding these objects is crucial for planetary defense.

**Key Questions:**
- Which asteroids are potentially hazardous?
- What characteristics make an asteroid dangerous?
- How can we predict asteroid behavior?
""")

# Basic Statistics
st.markdown("---")
st.subheader("📈 Dataset Statistics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Observations", len(df))
col2.metric("Features", len(X.columns))
col3.metric("Hazardous Count", y_hazardous.sum())
col4.metric("Non-Hazardous", len(y_hazardous) - y_hazardous.sum())

# Display sample data
st.markdown("---")
st.subheader("📋 Sample Data")
st.dataframe(df.head(20), use_container_width=True)

# Feature Information
st.markdown("---")
st.subheader("🔍 Feature Descriptions")

feature_info = pd.DataFrame({
    'Feature': X.columns,
    'Description': [
        'Brightness measure (lower = brighter)',
        'Minimum estimated diameter in kilometers',
        'Maximum estimated diameter in kilometers',
        'Speed relative to Earth in km/s',
        'Closest distance to Earth in AU',
        'Date of close approach (epoch time)'
    ],
    'Type': ['Numerical'] * len(X.columns)
})
st.dataframe(feature_info, use_container_width=True, hide_index=True)

# Statistical Summary
st.markdown("---")
st.subheader("📊 Statistical Summary")
st.dataframe(X.describe(), use_container_width=True)

# Distribution Plots
st.markdown("---")
st.subheader("📉 Feature Distributions")

col1, col2 = st.columns(2)

with col1:
    fig = px.histogram(df, x='Absolute Magnitude (H)', 
                      title='Absolute Magnitude Distribution',
                      nbins=30, color_discrete_sequence=['#1f77b4'])
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.histogram(df, x='Relative Velocity (km/s)', 
                      title='Relative Velocity Distribution',
                      nbins=30, color_discrete_sequence=['#ff7f0e'])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(df, x='Min Diameter (km)', 
                      title='Asteroid Diameter Distribution',
                      nbins=30, color_discrete_sequence=['#2ca02c'])
    st.plotly_chart(fig, use_container_width=True)
    
    fig = px.histogram(df, x='Miss Distance (astronomical)', 
                      title='Miss Distance Distribution',
                      nbins=30, color_discrete_sequence=['#d62728'])
    st.plotly_chart(fig, use_container_width=True)

# Correlation Heatmap
st.markdown("---")
st.subheader("🔥 Feature Correlation Matrix")

corr_matrix = X.corr()
fig = px.imshow(corr_matrix, 
                text_auto=True,
                title='Feature Correlation Heatmap',
                color_continuous_scale='RdBu_r',
                aspect='auto')
st.plotly_chart(fig, use_container_width=True)

# Scatter Plots
st.markdown("---")
st.subheader("🎯 Relationship Analysis")

col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(df, x='Min Diameter (km)', y='Relative Velocity (km/s)',
                    color='Is Potentially Hazardous',
                    title='Diameter vs Velocity',
                    color_discrete_map={True: 'red', False: 'blue'},
                    labels={'Is Potentially Hazardous': 'Hazardous'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(df, x='Miss Distance (astronomical)', y='Absolute Magnitude (H)',
                    color='Is Potentially Hazardous',
                    title='Miss Distance vs Magnitude',
                    color_discrete_map={True: 'red', False: 'blue'},
                    labels={'Is Potentially Hazardous': 'Hazardous'})
    st.plotly_chart(fig, use_container_width=True)

# Box Plots
st.markdown("---")
st.subheader("📦 Feature Comparison by Hazard Status")

col1, col2 = st.columns(2)

with col1:
    fig = px.box(df, x='Is Potentially Hazardous', y='Min Diameter (km)',
                title='Diameter by Hazard Status',
                color='Is Potentially Hazardous',
                color_discrete_map={True: 'red', False: 'blue'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.box(df, x='Is Potentially Hazardous', y='Relative Velocity (km/s)',
                title='Velocity by Hazard Status',
                color='Is Potentially Hazardous',
                color_discrete_map={True: 'red', False: 'blue'})
    st.plotly_chart(fig, use_container_width=True)

# Time Series
st.markdown("---")
st.subheader("📅 Temporal Analysis")

df_sorted = df.sort_values('Close Approach Date')
fig = px.scatter(df_sorted, x='Close Approach Date', y='Miss Distance (km)',
                color='Is Potentially Hazardous',
                title='Close Approaches Over Time',
                color_discrete_map={True: 'red', False: 'blue'},
                labels={'Is Potentially Hazardous': 'Hazardous'})
st.plotly_chart(fig, use_container_width=True)

# Key Insights
st.markdown("---")
st.subheader("💡 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Distribution Insights:**
    - Most asteroids are small (< 1 km diameter)
    - Velocity ranges from ~2-15 km/s
    - Miss distances vary greatly
    - Magnitude correlates with size
    """)

with col2:
    st.markdown("""
    **Hazard Patterns:**
    - All samples marked as potentially hazardous
    - Size and velocity are key factors
    - Closer approaches pose higher risk
    - Need to analyze orbital characteristics
    """)

st.success("✅ Data exploration complete! Proceed to modeling pages.")
