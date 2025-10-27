import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go
from utils import load_neo_data

st.set_page_config(page_title="Clustering Analysis", page_icon="🔍", layout="wide")

st.title("🔍 Clustering Analysis of Near-Earth Objects")

# Load data
@st.cache_data
def load_data():
    X, y_hazardous, y_distance, y_velocity, df = load_neo_data()
    return X, y_hazardous

X, y = load_data()

st.markdown("""
## Objective: Discover Asteroid Groups

**Problem Statement:**
Group asteroids based on their properties to identify:
- Families of similar asteroids
- Patterns in orbital characteristics
- Natural groupings for tracking purposes
- Outliers that need special attention

**Societal Impact:**
- Identifies asteroid families worth investigating together
- Reveals hidden patterns in NEO populations
- Guides strategic monitoring efforts
- Helps discover unusual or particularly dangerous objects
""")

# Prepare data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
st.markdown("---")
st.subheader("1️⃣ K-Means Clustering")

st.markdown("""
**Algorithm Explanation:**
- Partitions data into K clusters
- Each point belongs to cluster with nearest centroid
- Iteratively updates centroids
- Fast and scalable

**How it works:**
1. Randomly initialize K cluster centers
2. Assign each point to nearest center
3. Update centers to mean of assigned points
4. Repeat until convergence

**Use Case:**
Discover K distinct families of asteroids with similar properties.
""")

n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)

if st.button("Run K-Means Clustering", key="kmeans"):
    with st.spinner(f"Running K-Means with {n_clusters} clusters..."):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters_km = kmeans.fit_predict(X_scaled)
        
        # Metrics
        silhouette_km = silhouette_score(X_scaled, clusters_km)
        davies_bouldin_km = davies_bouldin_score(X_scaled, clusters_km)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Silhouette Score", f"{silhouette_km:.3f}")
        col2.metric("Davies-Bouldin Index", f"{davies_bouldin_km:.3f}")
        col3.metric("Clusters Found", n_clusters)
        
        st.info("""
        **Interpretation:**
        - **Silhouette Score:** -1 to 1 (higher is better, >0.5 is good)
        - **Davies-Bouldin Index:** Lower is better
        """)
        
        # 2D Visualization with PCA
        st.subheader("Cluster Visualization (2D PCA Projection)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        df_plot = pd.DataFrame({
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'Cluster': clusters_km.astype(str),
            'Hazardous': y
        })
        
        fig = px.scatter(df_plot, x='PCA1', y='PCA2', color='Cluster',
                        hover_data=['Hazardous'], title='K-Means Clusters (2D PCA Projection)',
                        color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D Visualization
        st.subheader("Cluster Visualization (3D PCA Projection)")
        pca_3d = PCA(n_components=3)
        X_pca_3d = pca_3d.fit_transform(X_scaled)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=X_pca_3d[:, 0],
            y=X_pca_3d[:, 1],
            z=X_pca_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=clusters_km,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cluster")
            )
        )])
        fig.update_layout(title='K-Means Clusters (3D PCA Projection)',
                         scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='PC3'))
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster Statistics
        st.subheader("Cluster Statistics")
        for i in range(n_clusters):
            cluster_mask = clusters_km == i
            st.write(f"**Cluster {i}:** {cluster_mask.sum()} asteroids")
            cluster_data = X[cluster_mask].describe()
            st.dataframe(cluster_data.T, use_container_width=True)
        
        pickle.dump(kmeans, open('kmeans_neo.pkl', 'wb'))
        st.success("✅ K-Means model saved!")

# DBSCAN Clustering
st.markdown("---")
st.subheader("2️⃣ DBSCAN Clustering")

st.markdown("""
**Algorithm Explanation:**
- Density-Based Spatial Clustering
- Finds arbitrarily shaped clusters
- Automatically determines number of clusters
- Identifies outliers as noise (-1 label)

**Parameters:**
- **eps:** Maximum distance between points in same cluster
- **min_samples:** Minimum points to form dense region

**Use Case:**
Discover natural groupings without specifying cluster count, identify unusual asteroids.
""")

col1, col2 = st.columns(2)
with col1:
    eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
with col2:
    min_samples = st.slider("Minimum Samples", 3, 20, 5)

if st.button("Run DBSCAN Clustering", key="dbscan"):
    with st.spinner("Running DBSCAN..."):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters_db = dbscan.fit_predict(X_scaled)
        
        n_clusters_db = len(set(clusters_db)) - (1 if -1 in clusters_db else 0)
        n_noise = list(clusters_db).count(-1)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Clusters Found", n_clusters_db)
        col2.metric("Noise Points", n_noise)
        col3.metric("Clustered Points", len(clusters_db) - n_noise)
        
        if n_clusters_db > 1:
            # Only calculate if we have valid clusters
            valid_mask = clusters_db != -1
            if valid_mask.sum() > 0:
                silhouette_db = silhouette_score(X_scaled[valid_mask], clusters_db[valid_mask])
                st.metric("Silhouette Score", f"{silhouette_db:.3f}")
        
        # 2D Visualization
        st.subheader("DBSCAN Clusters (2D PCA Projection)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        df_plot = pd.DataFrame({
            'PCA1': X_pca[:, 0],
            'PCA2': X_pca[:, 1],
            'Cluster': clusters_db.astype(str),
            'Hazardous': y
        })
        
        fig = px.scatter(df_plot, x='PCA1', y='PCA2', color='Cluster',
                        hover_data=['Hazardous'], title='DBSCAN Clusters (2D PCA Projection)',
                        color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig, use_container_width=True)
        
        # Noise points analysis
        if n_noise > 0:
            st.subheader("⚠️ Noise Points (Outliers)")
            st.write(f"Found {n_noise} unusual asteroids that don't fit any cluster!")
            st.write("These could be particularly interesting objects requiring special attention.")
            
            noise_mask = clusters_db == -1
            noise_data = X[noise_mask]
            st.dataframe(noise_data.describe().T, use_container_width=True)
        
        pickle.dump(dbscan, open('dbscan_neo.pkl', 'wb'))
        st.success("✅ DBSCAN model saved!")

# Conclusions
st.markdown("---")
st.subheader("📊 Clustering Conclusions")
st.markdown("""
**Algorithm Comparison:**
- **K-Means:** Fast, works well with spherical clusters, requires specifying K
- **DBSCAN:** Finds arbitrary shapes, handles noise, determines clusters automatically

**Insights:**
- Different asteroid families emerge based on size, velocity, and trajectory
- Outliers identified by DBSCAN may be particularly dangerous or unique
- Clustering helps organize monitoring and tracking efforts
- Natural groupings can inform mission planning strategies

**Applications:**
- **Resource Allocation:** Monitor similar asteroids as groups
- **Anomaly Detection:** DBSCAN outliers need special attention
- **Scientific Discovery:** Understand asteroid population structure
- **Mission Planning:** Target similar asteroids for exploration
""")

st.success("✅ Clustering analysis reveals natural groupings in asteroid population!")
