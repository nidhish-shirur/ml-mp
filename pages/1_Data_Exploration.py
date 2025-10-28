import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils import load_neo_data

st.set_page_config(page_title="Data Exploration", layout="wide")

st.title("Near-Earth Object Data Exploration")

# Load data - cached, no spinner for instant feel
@st.cache_data(show_spinner=False)
def load_data():
    X, y_hazardous, y_distance, y_velocity, df = load_neo_data()
    return X, y_hazardous, y_distance, y_velocity, df

X, y_hazardous, y_distance, y_velocity, df = load_data()

st.markdown("""
## Dataset Overview

Before building predictive models, we need to understand:
- The distribution of asteroid sizes, velocities, and distances
- Which features correlate with hazardous classification
- Whether the data has any quality issues
- What preprocessing steps might be needed

**Key Questions We'll Answer:**
- Which asteroids are potentially hazardous?
- What characteristics make an asteroid dangerous?
- How can we predict asteroid behavior?
- What patterns exist in the orbital data?
""")

# Basic Statistics
st.markdown("---")
st.subheader("Dataset Statistics")

st.markdown("""
**Understanding the Dataset Size:**
- **Total Observations**: Number of asteroid close approach events recorded
- **Features**: The characteristics we measure for each asteroid (size, speed, distance, etc.)
- **Hazardous Count**: Asteroids classified as potentially dangerous to Earth
- **Non-Hazardous**: Asteroids that pose no significant threat
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Observations", len(df))
col2.metric("Features", len(X.columns))
col3.metric("Hazardous Count", y_hazardous.sum())
col4.metric("Non-Hazardous", len(y_hazardous) - y_hazardous.sum())

# Display sample data
st.markdown("---")
st.subheader("Sample Data")
st.dataframe(df.head(20), use_container_width=True)

# Feature Information
st.markdown("---")
st.subheader("Feature Descriptions")

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
st.subheader("Statistical Summary")

st.markdown("""
**Reading the Statistics Table:**
- **count**: Number of valid (non-missing) values
- **mean**: Average value across all observations
- **std**: Standard deviation (how spread out the values are)
- **min/max**: Smallest and largest values observed
- **25%/50%/75%**: Quartiles (25% of data is below 25th percentile, etc.)

This helps us understand the typical range and variation in each feature.
""")

st.dataframe(X.describe(), use_container_width=True)

# Distribution Plots
st.markdown("---")
st.subheader("Feature Distributions")

st.markdown("""
**What are Distribution Plots?**

Histograms show how values are spread across a range. They help us understand:
- **Shape**: Is the data normally distributed (bell curve) or skewed?
- **Central tendency**: Where do most values cluster?
- **Spread**: How wide is the range of values?
- **Outliers**: Are there unusual extreme values?

**Why This Matters:**
- Skewed distributions may need transformation for some ML algorithms
- Outliers might indicate special cases requiring attention
- Understanding typical values helps with feature engineering
""")

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
st.subheader("Feature Correlation Matrix")

st.markdown("""
**Understanding Correlation:**

Correlation measures how two variables change together, ranging from -1 to +1:
- **+1**: Perfect positive correlation (both increase together)
- **0**: No correlation (independent variables)
- **-1**: Perfect negative correlation (one increases, other decreases)

**How to Read the Heatmap:**
- **Red colors**: Strong positive correlation
- **Blue colors**: Strong negative correlation
- **White/Light colors**: Weak or no correlation

**Why It's Important:**
- Highly correlated features may be redundant (multicollinearity)
- Identifies which features might be good predictors
- Helps in feature selection and dimensionality reduction
- Reveals hidden relationships in the data
""")

corr_matrix = X.corr()
fig = px.imshow(corr_matrix, 
                text_auto=True,
                title='Feature Correlation Heatmap',
                color_continuous_scale='RdBu_r',
                aspect='auto')
st.plotly_chart(fig, use_container_width=True)

# Scatter Plots
st.markdown("---")
st.subheader("Relationship Analysis")

st.markdown("""
**What are Scatter Plots?**

Scatter plots show the relationship between two variables, with each point representing one asteroid:
- **Red dots**: Potentially hazardous asteroids
- **Blue dots**: Non-hazardous asteroids

**What to Look For:**
- **Clustering**: Do hazardous asteroids group together?
- **Separation**: Can we draw a line to separate hazardous from safe?
- **Trends**: Do larger/faster asteroids tend to be more dangerous?
- **Outliers**: Are there unusual cases that don't fit the pattern?

This visual analysis helps us understand if these features can predict hazard status.
""")

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
st.subheader("Feature Comparison by Hazard Status")

st.markdown("""
**Understanding Box Plots:**

Box plots show the distribution of values for each group:
- **Box**: Contains the middle 50% of data (25th to 75th percentile)
- **Line in box**: Median (middle value)
- **Whiskers**: Extend to min/max within 1.5x the box height
- **Dots**: Outliers (extreme values)

**What We're Comparing:**
Do hazardous and non-hazardous asteroids differ significantly in:
- Size (diameter)
- Speed (velocity)
- Other characteristics

If the boxes don't overlap much, that feature is a good discriminator for classification.
""")

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
st.subheader("Temporal Analysis")

st.markdown("""
**Analyzing Asteroid Approaches Over Time:**

This timeline shows when asteroids made their closest approaches to Earth:
- **X-axis**: Date of close approach
- **Y-axis**: How close they came (in kilometers)
- **Color**: Red = hazardous, Blue = safe

**What This Reveals:**
- Frequency of asteroid encounters
- Whether close approaches are becoming more common
- Seasonal or temporal patterns
- Historical context for risk assessment

Understanding temporal patterns helps with long-term monitoring and resource allocation.
""")

df_sorted = df.sort_values('Close Approach Date')
fig = px.scatter(df_sorted, x='Close Approach Date', y='Miss Distance (km)',
                color='Is Potentially Hazardous',
                title='Close Approaches Over Time',
                color_discrete_map={True: 'red', False: 'blue'},
                labels={'Is Potentially Hazardous': 'Hazardous'})
st.plotly_chart(fig, use_container_width=True)

# Key Insights
st.markdown("---")
st.subheader("Key Insights & Next Steps")

st.markdown("""
**Summary of Exploration:**

From our analysis, we've learned that:

1. **Data Quality**: We have 905 asteroid observations with 6 key features
2. **Feature Relationships**: Size and velocity show correlations with hazard status
3. **Data Distribution**: Most asteroids are small, with varying velocities and distances
4. **Temporal Patterns**: Observations span a wide time range with varying approach distances

**What This Means for Machine Learning:**

Based on this exploration, we can proceed with:
- **Regression Models**: Predict miss distance and velocity
- **Classification Models**: Identify potentially hazardous asteroids
- **Clustering**: Group similar asteroids for efficient monitoring
- **Dimensionality Reduction**: Simplify the feature space while retaining information
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Distribution Insights:**
    - Most asteroids are small (< 1 km diameter)
    - Velocity ranges from ~2-15 km/s
    - Miss distances vary greatly (10K - 60M+ km)
    - Magnitude correlates strongly with size
    - Some features show skewed distributions
    """)

with col2:
    st.markdown("""
    **Hazard Patterns:**
    - All samples marked as potentially hazardous in this dataset
    - Size and velocity are key discriminating factors
    - Closer approaches generally pose higher risk
    - Orbital characteristics play important roles
    - Need robust models for accurate classification
    """)


