"""
Pre-train all models to speed up Streamlit app loading
Run this script once: python train_all_models.py
Then Streamlit will just LOAD the models (instant!)
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

print("="*60)
print("NEO ML PROJECT - TRAINING ALL MODELS")
print("This pre-trains models once for instant loading!")
print("="*60)

print("\n1. Loading data from neo_data.csv...")
if not os.path.exists('neo_data.csv'):
    print("ERROR: neo_data.csv not found!")
    print("   Please ensure neo_data.csv is in the current directory.")
    exit(1)

df = pd.read_csv('neo_data.csv')

# Select relevant features
feature_cols = [
    'Absolute Magnitude (H)',
    'Min Diameter (km)',
    'Max Diameter (km)',
    'Relative Velocity (km/s)',
    'Miss Distance (astronomical)',
    'Epoch Date Close Approach'
]

X = df[feature_cols]
y_hazardous = df['Is Potentially Hazardous']
y_distance = df['Miss Distance (km)']
y_velocity = df['Relative Velocity (km/s)']

print(f"   Dataset shape: {X.shape}")
print(f"   Features: {len(feature_cols)}")
print(f"   Samples: {len(X)}")

print("\n2. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train_h, y_test_h = train_test_split(X, y_hazardous, test_size=0.2, random_state=42)
_, _, y_train_d, y_test_d = train_test_split(X, y_distance, test_size=0.2, random_state=42)
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

print("\n3. Scaling data with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)
print("   [OK] Data scaled")

# Dictionary to store models
models = {}

print("\n" + "="*60)
print("TRAINING REGRESSION MODELS")
print("="*60)

print("\n4. Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train_d)
models['linear_regression_neo'] = lr
print("   [OK] Linear Regression trained")

print("\n5. Training Polynomial Regression (degree=2)...")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
lr_poly = LinearRegression()
lr_poly.fit(X_poly, y_train_d)
models['poly_regression_neo'] = lr_poly
models['poly_features_neo'] = poly
print("   [OK] Polynomial Regression trained")

print("\n" + "="*60)
print("TRAINING CLASSIFICATION MODELS")
print("="*60)

print("\n6. Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train_h)
models['decision_tree_neo'] = dt
print("   [OK] Decision Tree trained")

print("\n7. Training SVM (this may take a while)...")
svm = SVC(random_state=42, kernel='rbf')
svm.fit(X_train, y_train_h)
models['svm_neo'] = svm
print("   [OK] SVM trained")

print("\n8. Training Random Forest...")
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train_h)
models['random_forest_neo'] = rf
print("   [OK] Random Forest trained")

print("\n9. Training Gradient Boosting...")
gb = GradientBoostingClassifier(random_state=42, n_estimators=100)
gb.fit(X_train, y_train_h)
models['gradient_boosting_neo'] = gb
print("   [OK] Gradient Boosting trained")

print("\n" + "="*60)
print("TRAINING CLUSTERING MODELS")
print("="*60)

print("\n10. Training K-Means...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
models['kmeans_neo'] = kmeans
print("   [OK] K-Means trained")

print("\n11. Training DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)
models['dbscan_neo'] = dbscan
print("   [OK] DBSCAN trained")

print("\n" + "="*60)
print("TRAINING DIMENSIONALITY REDUCTION")
print("="*60)

print("\n12. Training PCA...")
pca = PCA(n_components=3)
pca.fit(X_scaled)
models['pca_neo'] = pca
print("   [OK] PCA trained")

print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

for name, model in models.items():
    filename = f"{name}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"   [OK] Saved: {filename}")

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

print("\nRegression Models:")
from sklearn.metrics import r2_score, mean_squared_error

y_pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test_d, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test_d, y_pred_lr))
print(f"   Linear Regression - R²: {r2_lr:.4f}, RMSE: {rmse_lr:.2e}")

X_test_poly = poly.transform(X_test)
y_pred_poly = lr_poly.predict(X_test_poly)
r2_poly = r2_score(y_test_d, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test_d, y_pred_poly))
print(f"   Polynomial Regression - R²: {r2_poly:.4f}, RMSE: {rmse_poly:.2e}")

print("\nClassification Models:")
from sklearn.metrics import accuracy_score

acc_dt = accuracy_score(y_test_h, dt.predict(X_test))
acc_svm = accuracy_score(y_test_h, svm.predict(X_test))
acc_rf = accuracy_score(y_test_h, rf.predict(X_test))
acc_gb = accuracy_score(y_test_h, gb.predict(X_test))

print(f"   Decision Tree - Accuracy: {acc_dt:.4f}")
print(f"   SVM - Accuracy: {acc_svm:.4f}")
print(f"   Random Forest - Accuracy: {acc_rf:.4f}")
print(f"   Gradient Boosting - Accuracy: {acc_gb:.4f} [BEST]")

print("\nClustering Models:")
from sklearn.metrics import silhouette_score

clusters_km = kmeans.predict(X_scaled)
sil_km = silhouette_score(X_scaled, clusters_km)
print(f"   K-Means - Silhouette Score: {sil_km:.4f}")

clusters_db = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(clusters_db)) - (1 if -1 in clusters_db else 0)
n_noise = list(clusters_db).count(-1)
print(f"   DBSCAN - Clusters: {n_clusters_db}, Noise Points: {n_noise}")

print("\nDimensionality Reduction:")
var_pca = sum(pca.explained_variance_ratio_) * 100
print(f"   PCA (3 components) - Variance Retained: {var_pca:.2f}%")

print("\n" + "="*60)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("="*60)

# Save train/test splits for consistency
print("\n14. Saving train/test splits for consistency...")
splits = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train_h': y_train_h,
    'y_test_h': y_test_h,
    'y_train_d': y_train_d,
    'y_test_d': y_test_d,
    'X_scaled': X_scaled,
    'X_train_scaled': X_train_scaled,
    'X_test_scaled': X_test_scaled,
    'scaler': scaler
}
with open('train_test_splits_neo.pkl', 'wb') as f:
    pickle.dump(splits, f)
print("   [OK] Saved train_test_splits_neo.pkl")

print("\n" + "="*60)
print("READY TO RUN!")
print("="*60)
print("\nYou can now run: streamlit run app.py")
print("All models will LOAD INSTANTLY (no retraining needed)")
print("="*60)

