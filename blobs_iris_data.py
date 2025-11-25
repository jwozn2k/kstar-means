import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.decomposition import PCA
from kstar_means import KStarMeans

# Test 1: make_blobs
print("="*60)
print("TEST 1: make_blobs (5 klastrów)")
print("="*60)

X_blobs, y_blobs = make_blobs(n_samples=500, centers=10, n_features=2,
                               random_state=42, cluster_std=0.6)

kstar_blobs = KStarMeans(random_state=42)
kstar_blobs.fit(X_blobs)

print(f"Prawdziwa liczba klastrów: 5")
print(f"Znaleziona k*: {kstar_blobs.n_clusters_}")
print(f"Iteracje: {len(kstar_blobs.cost_history_)}")

# Test 2: Iris
print(f"\n{'='*60}")
print("TEST 2: Iris (3 gatunki)")
print("="*60)

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# PCA do 2D dla wizualizacji
pca = PCA(n_components=2)
X_iris_2d = pca.fit_transform(X_iris)

kstar_iris = KStarMeans(random_state=42)
kstar_iris.fit(X_iris)

print(f"Prawdziwa liczba klastrów: 3")
print(f"Znaleziona k*: {kstar_iris.n_clusters_}")
print(f"Iteracje: {len(kstar_iris.cost_history_)}")

# Wizualizacja
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Blobs - prawdziwe
axes[0, 0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap='tab10', alpha=0.6)
axes[0, 0].set_title('make_blobs - prawdziwe (k=5)')
axes[0, 0].grid(True, alpha=0.3)

# Blobs - K*-Means
pred_blobs = kstar_blobs.predict(X_blobs)
axes[0, 1].scatter(X_blobs[:, 0], X_blobs[:, 1], c=pred_blobs, cmap='tab10', alpha=0.6)
axes[0, 1].scatter(kstar_blobs.centroids_[:, 0], kstar_blobs.centroids_[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidth=2)
axes[0, 1].set_title(f'K*-Means (k*={kstar_blobs.n_clusters_})')
axes[0, 1].grid(True, alpha=0.3)

# Blobs - koszt MDL
axes[0, 2].plot(kstar_blobs.cost_history_, 'b-o')
axes[0, 2].set_title('Koszt MDL')
axes[0, 2].set_xlabel('Iteracja')
axes[0, 2].set_yscale('log')
axes[0, 2].grid(True, alpha=0.3)

# Iris - prawdziwe (2D PCA)
axes[1, 0].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=y_iris, cmap='tab10', alpha=0.6)
axes[1, 0].set_title('Iris - prawdziwe (k=3, PCA 2D)')
axes[1, 0].grid(True, alpha=0.3)

# Iris - K*-Means (2D PCA)
pred_iris = kstar_iris.predict(X_iris)
centroids_2d = pca.transform(kstar_iris.centroids_)
axes[1, 1].scatter(X_iris_2d[:, 0], X_iris_2d[:, 1], c=pred_iris, cmap='tab10', alpha=0.6)
axes[1, 1].scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidth=2)
axes[1, 1].set_title(f'K*-Means (k*={kstar_iris.n_clusters_})')
axes[1, 1].grid(True, alpha=0.3)

# Iris - koszt MDL
axes[1, 2].plot(kstar_iris.cost_history_, 'b-o')
axes[1, 2].set_title('Koszt MDL')
axes[1, 2].set_xlabel('Iteracja')
axes[1, 2].set_yscale('log')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig('kstar_test_blobs_iris.png', dpi=150, bbox_inches='tight')
print(f"\nWykres zapisany: {'kstar_test_blobs_iris.png'}")
plt.show()