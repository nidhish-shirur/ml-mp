import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
import plotly.express as px
import plotly.graph_objects as go
from utils import load_neo_data

st.set_page_config(page_title="Dimensionality Reduction", page_icon="�", layout="wide")

st.title("Dimensionality Reduction for NEO Analysis")

# Load data
@st.cache_data
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

st.subheader("📊 Original Data Dimensions")
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
        
        # 3D Visualization
        if n_components_pca >= 3:
            st.subheader("3D PCA Projection")
            
            fig = go.Figure(data=[go.Scatter3d(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                z=X_pca[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=y.astype(int),
                    colorscale='RdBu',
                    showscale=True,
                    colorbar=dict(title="Hazardous")
                ),
                text=y.map({True: 'Hazardous', False: 'Non-Hazardous'})
            )])
            fig.update_layout(title='Asteroids in 3D PCA Space',
                             scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
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
        st.success("✅ PCA model saved!")

# SVD
st.markdown("---")
st.subheader("2. Singular Value Decomposition (SVD)")

st.markdown("""
**Algorithm Explanation:**
- Matrix factorization: X = U * Σ * V^T
- Similar to PCA but computationally different
- Works well with sparse data
- Often used in recommender systems

**Difference from PCA:**
- PCA: Based on covariance matrix
- SVD: Direct matrix factorization
- Results are very similar for centered data

**Use Case:**
Alternative dimensionality reduction, especially for sparse matrices
""")

n_components_svd = st.slider("Number of SVD Components", 2, min(X.shape[0], X.shape[1]), 3)

if st.button("Run SVD", key="svd"):
    with st.spinner("Running SVD..."):
        svd = TruncatedSVD(n_components=n_components_svd, random_state=42)
        X_svd = svd.fit_transform(X_scaled)
        
        # Explained Variance
        st.subheader("Variance Explained by Components")
        
        variance_df = pd.DataFrame({
            'Component': [f'SV{i+1}' for i in range(n_components_svd)],
            'Variance Explained (%)': svd.explained_variance_ratio_ * 100,
            'Cumulative Variance (%)': np.cumsum(svd.explained_variance_ratio_) * 100
        })
        
        st.dataframe(variance_df, use_container_width=True, hide_index=True)
        
        # Scree Plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=variance_df['Component'], 
                            y=variance_df['Variance Explained (%)'],
                            name='Individual', marker_color='purple'))
        fig.add_trace(go.Scatter(x=variance_df['Component'], 
                                y=variance_df['Cumulative Variance (%)'],
                                name='Cumulative', mode='lines+markers',
                                line=dict(color='orange', width=2)))
        fig.update_layout(title='SVD Variance Explained',
                         xaxis_title='Singular Vector',
                         yaxis_title='Variance Explained (%)')
        st.plotly_chart(fig, use_container_width=True)
        
        # 2D Visualization
        if n_components_svd >= 2:
            st.subheader("2D SVD Projection")
            df_svd = pd.DataFrame({
                'SV1': X_svd[:, 0],
                'SV2': X_svd[:, 1],
                'Hazardous': y.map({True: 'Yes', False: 'No'})
            })
            
            fig = px.scatter(df_svd, x='SV1', y='SV2', color='Hazardous',
                            title='Asteroids in 2D SVD Space',
                            color_discrete_map={'Yes': 'red', 'No': 'green'})
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D Visualization
        if n_components_svd >= 3:
            st.subheader("3D SVD Projection")
            
            fig = go.Figure(data=[go.Scatter3d(
                x=X_svd[:, 0],
                y=X_svd[:, 1],
                z=X_svd[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=y.astype(int),
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title="Hazardous")
                ),
                text=y.map({True: 'Hazardous', False: 'Non-Hazardous'})
            )])
            fig.update_layout(title='Asteroids in 3D SVD Space',
                             scene=dict(xaxis_title='SV1', yaxis_title='SV2', zaxis_title='SV3'))
            st.plotly_chart(fig, use_container_width=True)
        
        pickle.dump(svd, open('svd_neo.pkl', 'wb'))
        st.success("✅ SVD model saved!")

# Comparison
st.markdown("---")
st.subheader("⚖️ PCA vs SVD Comparison")

comparison_df = pd.DataFrame({
    'Aspect': ['Computation', 'Interpretability', 'Speed', 'Memory', 'Use Case'],
    'PCA': [
        'Covariance-based',
        'High (variance explained)',
        'Fast for small data',
        'Requires full data in memory',
        'General dimensionality reduction'
    ],
    'SVD': [
        'Matrix factorization',
        'Medium',
        'Fast for sparse data',
        'Can work with partial data',
        'Sparse data, recommender systems'
    ]
})

st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Conclusions
st.markdown("---")
st.subheader("Dimensionality Reduction Conclusions")

st.markdown("""
**Key Findings:**
- First few components capture most variance
- 2-3 components often retain 70-90% of information
- Visualization in 2D/3D reveals patterns
- Reduced dimensions speed up downstream models

**Applications:**
- **Visualization:** Understand data structure in 2D/3D
- **Feature Engineering:** Use components as new features
- **Anomaly Detection:** Outliers in reduced space
- **Model Speed:** Train faster on reduced features
- **Data Compression:** Store less, preserve information
- **Combine with clustering:** Efficient monitoring of asteroid groups

**Recommendations:**
- Use PCA for most cases (more interpretable)
- Use SVD for sparse data or when memory is limited
- Keep components that explain 80-90% variance
- Visualize in reduced space to understand patterns
""")

st.success("Dimensionality reduction reveals hidden structure in asteroid data!")
