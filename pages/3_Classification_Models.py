import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.figure_factory as ff
from utils import load_neo_data

st.set_page_config(page_title="Classification Models", layout="wide")

st.title("Classification Models for Hazardous Asteroid Detection")

# Load data
@st.cache_data(show_spinner=False)
def load_data():
    X, y_hazardous, y_distance, y_velocity, df = load_neo_data()
    return X, y_hazardous

X, y = load_data()

st.markdown("""
## Objective: Classify Potentially Hazardous Asteroids

**Problem Statement:**
Identify which Near-Earth Objects pose a potential hazard to Earth based on their physical 
and orbital characteristics.

**Societal Impact:**
- **Early Warning:** Detect dangerous asteroids years in advance
- **Resource Allocation:** Focus monitoring on high-risk objects
- **Mission Planning:** Enable timely deflection missions
- **Public Safety:** Potential to save millions of lives
- **Economic Protection:** Prevent trillions in potential damage

**Classification Target:** Is Potentially Hazardous (True/False)
""")

# Data split
test_size = st.sidebar.slider("Test Set Size (%)", 10, 40, 20, key="class_split") / 100
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

st.subheader("Class Distribution")
col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", len(y))
col2.metric("Hazardous (True)", int(y.sum()))
col3.metric("Non-Hazardous (False)", len(y) - int(y.sum()))

# Decision Tree
st.markdown("---")
st.subheader("1. Decision Tree Classifier")

st.markdown("""
**Algorithm Explanation:**
- Creates a tree of decisions based on feature values
- Each node represents a decision (e.g., "Is diameter > threshold?")
- Easy to interpret and visualize
- Handles non-linear relationships naturally

**How it works:**
1. Selects feature that best splits data
2. Creates branches for different values
3. Repeats recursively
4. Stops when pure nodes or max depth reached

**Use Case:** Quickly identify key factors that determine asteroid hazard level
""")

if st.button("Train Decision Tree", key="dt"):
    with st.spinner("Training Decision Tree..."):
        dt = DecisionTreeClassifier(random_state=42, max_depth=5)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        
        acc_dt = accuracy_score(y_test, y_pred_dt)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{acc_dt:.4f}")
            st.text("Classification Report:")
            report = classification_report(y_test, y_pred_dt, target_names=['Non-Hazardous', 'Hazardous'])
            st.text(report)
        
        with col2:
            cm = confusion_matrix(y_test, y_pred_dt)
            fig = ff.create_annotated_heatmap(cm, x=['Non-Hazardous', 'Hazardous'], 
                                             y=['Non-Hazardous', 'Hazardous'],
                                             colorscale='Blues')
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)
        
        # Tree visualization
        st.subheader("Decision Tree Structure")
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(dt, filled=True, feature_names=X.columns, class_names=['Non-Hazardous', 'Hazardous'],
                 max_depth=3, ax=ax, fontsize=10)
        plt.title("Decision Tree Structure (Max Depth 3 shown)")
        st.pyplot(fig)
        
        # Feature Importance
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': dt.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = go.Figure(go.Bar(x=importance_df['Importance'], y=importance_df['Feature'],
                              orientation='h'))
        fig.update_layout(title="Feature Importance", xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)
        
        pickle.dump(dt, open('decision_tree_neo.pkl', 'wb'))

# SVM
st.markdown("---")
st.subheader("2. Support Vector Machine (SVM)")

st.markdown("""
**Algorithm Explanation:**
- Finds optimal hyperplane that separates classes
- Maximizes margin between classes
- Works well in high-dimensional spaces
- Uses kernel trick for non-linear boundaries

**Key Concept:**
Imagine a line (or hyperplane in higher dimensions) that best separates hazardous from 
non-hazardous asteroids, with maximum distance to nearest points of each class.

**Use Case:** Robust classification when classes have complex boundaries
""")

if st.button("Train SVM", key="svm"):
    with st.spinner("Training SVM (this may take a moment)..."):
        svm = SVC(random_state=42, kernel='rbf')
        svm.fit(X_train, y_train)
        y_pred_svm = svm.predict(X_test)
        
        acc_svm = accuracy_score(y_test, y_pred_svm)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{acc_svm:.4f}")
            st.text("Classification Report:")
            report = classification_report(y_test, y_pred_svm, target_names=['Non-Hazardous', 'Hazardous'])
            st.text(report)
        
        with col2:
            cm = confusion_matrix(y_test, y_pred_svm)
            fig = ff.create_annotated_heatmap(cm, x=['Non-Hazardous', 'Hazardous'], 
                                             y=['Non-Hazardous', 'Hazardous'],
                                             colorscale='Viridis')
            fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)
        
        # SVM Decision Boundary on 2D PCA
        st.subheader("Decision Boundary Visualization (2D PCA)")
        
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        svm_pca = SVC(random_state=42, kernel='rbf')
        svm_pca.fit(X_train_pca, y_train)
        
        # Create mesh grid
        x_min, x_max = X_train_pca[:, 0].min() - 0.5, X_train_pca[:, 0].max() + 0.5
        y_min, y_max = X_train_pca[:, 1].min() - 0.5, X_train_pca[:, 1].max() + 0.5
        
        x_grid = np.linspace(x_min, x_max, 200)
        y_grid = np.linspace(y_min, y_max, 200)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        
        # Add clear hyperplane (decision boundary) as thick black line
        ax.contour(xx, yy, Z, colors='black', linewidths=3, levels=[0.5], linestyles='solid')
        
        scatter = ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, 
                            cmap='RdYlBu', edgecolors='black', s=50)
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_title('SVM Decision Boundary (2D PCA Projection)')
        plt.colorbar(scatter, ax=ax, label='Hazardous')
        st.pyplot(fig)
        
        pickle.dump(svm, open('svm_neo.pkl', 'wb'))

# Ensemble Methods
st.markdown("---")
st.subheader("3. Ensemble Learning")

st.markdown("""
**Random Forest (Bagging):**
- Builds multiple decision trees on random subsets
- Combines predictions by voting
- Reduces overfitting, more robust

**Gradient Boosting:**
- Builds trees sequentially
- Each tree corrects errors of previous ones
- Often achieves highest accuracy

**Use Case:** State-of-the-art performance for critical asteroid classification
""")

if st.button("Train Ensemble Models", key="ensemble"):
    with st.spinner("Training Random Forest and Gradient Boosting..."):
        # Random Forest
        rf = RandomForestClassifier(random_state=42, n_estimators=100)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(random_state=42, n_estimators=100)
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)
        acc_gb = accuracy_score(y_test, y_pred_gb)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Random Forest")
            st.metric("Accuracy", f"{acc_rf:.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred_rf, target_names=['Non-Hazardous', 'Hazardous']))
        
        with col2:
            st.subheader("Gradient Boosting")
            st.metric("Accuracy", f"{acc_gb:.4f}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred_gb, target_names=['Non-Hazardous', 'Hazardous']))
        
        # Feature Importance (Random Forest)
        st.subheader("Feature Importance (Random Forest)")
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=X.columns[indices], y=importances[indices]))
        fig.update_layout(title="Feature Importances",
                         xaxis_title="Features", yaxis_title="Importance",
                         xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrices
        col1, col2 = st.columns(2)
        with col1:
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            fig = ff.create_annotated_heatmap(cm_rf, x=['Non-Hazardous', 'Hazardous'], 
                                             y=['Non-Hazardous', 'Hazardous'],
                                             colorscale='Greens')
            fig.update_layout(title="Random Forest Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            cm_gb = confusion_matrix(y_test, y_pred_gb)
            fig = ff.create_annotated_heatmap(cm_gb, x=['Non-Hazardous', 'Hazardous'], 
                                             y=['Non-Hazardous', 'Hazardous'],
                                             colorscale='Oranges')
            fig.update_layout(title="Gradient Boosting Confusion Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        pickle.dump(rf, open('random_forest_neo.pkl', 'wb'))
        pickle.dump(gb, open('gradient_boosting_neo.pkl', 'wb'))

# Conclusions
st.markdown("---")
st.subheader("Classification Conclusions")
st.markdown("""
**Model Comparison:**
- **Decision Tree:** Fast, interpretable, good for understanding decision logic
- **SVM:** Powerful for complex boundaries, works well in high dimensions
- **Random Forest:** Robust, reduces overfitting, provides feature importance
- **Gradient Boosting:** Often highest accuracy, industry standard

**Business Impact:**
- Accurate classification enables early warning systems
- Helps prioritize asteroid tracking resources
- Supports mission planning for deflection
- Critical for protecting Earth from impacts
- Can save months or years of warning time

**Key Features for Hazard Detection:**
- Asteroid diameter (size matters!)
- Relative velocity (speed of approach)
- Miss distance (how close it gets)
- Absolute magnitude (brightness/size indicator)
""")
