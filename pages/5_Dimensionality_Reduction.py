import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from utils import load_neo_data

st.set_page_config(page_title="Dimensionality Reduction", layout="wide")

st.title("Dimensionality Reduction for NEO Analysis")

# Load data
@st.cache_data(show_spinner=False)
def load_data():
    X, y_hazardous, y_distance, y_velocity, df = load_neo_data()
    return X, y_hazardous

X, y = load_data()

st.markdown("""
## Objective: Reduce Feature Complexity

**Why Dimensionality Reduction?**
- Simplify complex high-dimensional data
- Visualize relationships in 2D/3D
- Remove redundant features
- Speed up model training
- Reduce storage requirements

**Techniques:**
- **PCA:** Finds directions of maximum variance
- **SVD:** Matrix factorization approach, similar to PCA
""")

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.subheader("Original Data Dimensions")
col1, col2 = st.columns(2)
col1.metric("Number of Features", X.shape[1])
col2.metric("Number of Samples", X.shape[0])

# PCA
st.markdown("---")
st.subheader("1. Principal Component Analysis (PCA)")

st.markdown("""
**Algorithm Explanation:**
- Finds directions (principal components) of maximum variance
- Projects data onto these directions
- First component captures most variance, second captures second-most, etc.
- Components are orthogonal (uncorrelated)

**Mathematical Concept:**
PCA performs eigenvalue decomposition on the covariance matrix:
- Eigenvectors = Principal Components (directions)
- Eigenvalues = Amount of variance explained

**Analogy:**
Imagine viewing asteroid data from different angles. PCA finds the "best angles" 
that show the most variation in asteroid properties.
""")

n_components_pca = st.slider("Number of PCA Components", 2, min(X.shape[0], X.shape[1]), 3)

if st.button("Run PCA", key="pca"):
    with st.spinner("Running PCA..."):
        pca = PCA(n_components=n_components_pca)
        X_pca = pca.fit_transform(X_scaled)
        
        # Explained Variance
        st.subheader("Variance Explained by Components")
        
        variance_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(n_components_pca)],
            'Variance Explained (%)': pca.explained_variance_ratio_ * 100,
            'Cumulative Variance (%)': np.cumsum(pca.explained_variance_ratio_) * 100
        })
        
        st.dataframe(variance_df, use_container_width=True, hide_index=True)
        
        # Scree Plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=variance_df['Component'], 
                            y=variance_df['Variance Explained (%)'],
                            name='Individual'))
        fig.add_trace(go.Scatter(x=variance_df['Component'], 
                                y=variance_df['Cumulative Variance (%)'],
                                name='Cumulative', mode='lines+markers',
                                line=dict(color='red', width=2)))
        fig.update_layout(title='PCA Variance Explained (Scree Plot)',
                         xaxis_title='Principal Component',
                         yaxis_title='Variance Explained (%)')
        st.plotly_chart(fig, use_container_width=True)
        
        # 2D Visualization
        if n_components_pca >= 2:
            st.subheader("2D PCA Projection")
            df_pca = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Hazardous': y.map({True: 'Yes', False: 'No'})
            })
            
            fig = px.scatter(df_pca, x='PC1', y='PC2', color='Hazardous',
                            title='Asteroids in 2D PCA Space',
                            color_discrete_map={'Yes': 'red', 'No': 'blue'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Component Loadings
        st.subheader("Feature Loadings (Component Composition)")
        loadings_df = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components_pca)],
            index=X.columns
        )
        st.dataframe(loadings_df.style.background_gradient(cmap='RdBu', axis=0), 
                    use_container_width=True)
        
        st.info("""
        **Interpreting Loadings:**
        - Each row shows how much each feature contributes to each component
        - Positive values (red): feature increases with component
        - Negative values (blue): feature decreases with component
        - Large absolute values indicate strong influence
        """)
        
        pickle.dump(pca, open('pca_neo.pkl', 'wb'))

# Conclusions
st.markdown("---")
st.subheader("Dimensionality Reduction Conclusions")

st.markdown("""
**Key Findings:**
- First few components capture most variance
- 2-3 components often retain 70-90% of information
- Visualization in 2D reveals patterns
- Reduced dimensions speed up downstream models

**Applications:**
- **Visualization:** Understand data structure in 2D
- **Feature Engineering:** Use components as new features
- **Anomaly Detection:** Outliers in reduced space
- **Model Speed:** Train faster on reduced features
- **Data Compression:** Store less, preserve information
- **Combine with clustering:** Efficient monitoring of asteroid groups

**Recommendations:**
- Use PCA for dimensionality reduction (highly interpretable)
- Keep components that explain 80-90% variance
- Visualize in reduced space to understand patterns
""")
