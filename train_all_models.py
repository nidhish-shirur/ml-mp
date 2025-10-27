"""
Pre-train all models to speed up Streamlit app loading
Run this script once: python train_all_models.py
"""

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
from sklearn.decomposition import PCA, TruncatedSVD

print("="*60)
print("NEO ML Project - Model Training Script")
print("="*60)

print("\n1. Loading data...")
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

print("\n2. Splitting data...")
X_train, X_test, y_train_h, y_test_h = train_test_split(X, y_hazardous, test_size=0.2, random_state=42)
_, _, y_train_d, y_test_d = train_test_split(X, y_distance, test_size=0.2, random_state=42)

print("\n3. Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dictionary to store models
models = {}

print("\n" + "="*60)
print("TRAINING REGRESSION MODELS")
print("="*60)

print("\n4. Training Linear Regression...")
lr = LinearRegression()
lr.fit(X_train, y_train_d)
models['linear_regression_neo'] = lr
print("   ✓ Linear Regression trained")

print("\n5. Training Polynomial Regression (degree=2)...")
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
lr_poly = LinearRegression()
lr_poly.fit(X_poly, y_train_d)
models['poly_regression_neo'] = lr_poly
models['poly_features_neo'] = poly
print("   ✓ Polynomial Regression trained")

print("\n" + "="*60)
print("TRAINING CLASSIFICATION MODELS")
print("="*60)

print("\n6. Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train_h)
models['decision_tree_neo'] = dt
print("   ✓ Decision Tree trained")

print("\n7. Training SVM (this may take a while)...")
svm = SVC(random_state=42, kernel='rbf')
svm.fit(X_train, y_train_h)
models['svm_neo'] = svm
print("   ✓ SVM trained")

print("\n8. Training Random Forest...")
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train_h)
models['random_forest_neo'] = rf
print("   ✓ Random Forest trained")

print("\n9. Training Gradient Boosting...")
gb = GradientBoostingClassifier(random_state=42, n_estimators=100)
gb.fit(X_train, y_train_h)
models['gradient_boosting_neo'] = gb
print("   ✓ Gradient Boosting trained")

print("\n" + "="*60)
print("TRAINING CLUSTERING MODELS")
print("="*60)

print("\n10. Training K-Means...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
models['kmeans_neo'] = kmeans
print("   ✓ K-Means trained")

print("\n11. Training DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)
models['dbscan_neo'] = dbscan
print("   ✓ DBSCAN trained")

print("\n" + "="*60)
print("TRAINING DIMENSIONALITY REDUCTION")
print("="*60)

print("\n12. Training PCA...")
pca = PCA(n_components=3)
pca.fit(X_scaled)
models['pca_neo'] = pca
print("   ✓ PCA trained")

print("\n13. Training SVD...")
svd = TruncatedSVD(n_components=3, random_state=42)
svd.fit(X_scaled)
models['svd_neo'] = svd
print("   ✓ SVD trained")

print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)

for name, model in models.items():
    filename = f"{name}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✓ Saved: {filename}")

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
print(f"   Gradient Boosting - Accuracy: {acc_gb:.4f} 🏆")

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
var_svd = sum(svd.explained_variance_ratio_) * 100
print(f"   PCA (3 components) - Variance Retained: {var_pca:.2f}%")
print(f"   SVD (3 components) - Variance Retained: {var_svd:.2f}%")

print("\n" + "="*60)
print("✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("="*60)
print("\nYou can now run: streamlit run app.py")
print("Models will load instantly from disk!\n")
