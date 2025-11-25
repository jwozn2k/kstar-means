import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from kstar_means import KStarMeans

# Te same dane
X_blobs, y_blobs = make_blobs(n_samples=500, centers=5, n_features=2,
                               random_state=42, cluster_std=0.6)

# Sprawdźmy prawdziwe centra
from sklearn.cluster import KMeans
kmeans_oracle = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_oracle.fit(X_blobs)

print("Prawdziwe centra klastrów (z make_blobs):")
for i in range(5):
    mask = y_blobs == i
    center = X_blobs[mask].mean(axis=0)
    size = mask.sum()
    print(f"  Klaster {i}: centrum={center}, rozmiar={size}")

# Odległości między centrami
from scipy.spatial.distance import pdist, squareform
true_centers = np.array([X_blobs[y_blobs == i].mean(axis=0) for i in range(5)])
distances = squareform(pdist(true_centers))

print("\nMacierz odległości między centrami:")
print(distances.round(2))
min_dist_idx = np.unravel_index(distances.argmin() + np.argmax(distances.shape), distances.shape)
print(f"\nNajbliższa para: klastry {min_dist_idx}, odległość: {distances[min_dist_idx]:.2f}")

# K*-Means
kstar = KStarMeans(random_state=42)
kstar.fit(X_blobs)
pred = kstar.predict(X_blobs)

print(f"\nK*-Means znalazł k*={kstar.n_clusters_}")
print("\nCentroidy K*-Means:")
for i in range(kstar.n_clusters_):
    print(f"  Klaster {i}: {kstar.centroids_[i]}")

# Który klaster został połączony?
print("\nRozmiary klastrów K*-Means:")
for i in range(kstar.n_clusters_):
    size = (pred == i).sum()
    print(f"  Klaster {i}: {size} punktów")

# Wizualizacja problemu
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Prawdziwe
axes[0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap='tab10', alpha=0.6, s=30)
axes[0].scatter(true_centers[:, 0], true_centers[:, 1],
                c='red', marker='X', s=300, edgecolors='black', linewidth=2)
axes[0].set_title('Prawdziwe klastry (k=5)')
axes[0].grid(True, alpha=0.3)

# Dodaj linie między najbliższymi centrami
closest_pairs = np.argsort(distances.flatten())
for idx in closest_pairs[:5]:  # 5 najbliższych par
    i, j = np.unravel_index(idx, distances.shape)
    if i < j:
        axes[0].plot([true_centers[i, 0], true_centers[j, 0]],
                     [true_centers[i, 1], true_centers[j, 1]],
                     'r--', alpha=0.3, linewidth=2)
        axes[0].text((true_centers[i, 0] + true_centers[j, 0])/2,
                     (true_centers[i, 1] + true_centers[j, 1])/2,
                     f'{distances[i, j]:.2f}', fontsize=8)

# K*-Means
axes[1].scatter(X_blobs[:, 0], X_blobs[:, 1], c=pred, cmap='tab10', alpha=0.6, s=30)
axes[1].scatter(kstar.centroids_[:, 0], kstar.centroids_[:, 1],
                c='red', marker='X', s=300, edgecolors='black', linewidth=2)
axes[1].set_title(f'K*-Means (k*={kstar.n_clusters_})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig('kstar_blobs_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nWykres zapisany: kstar_blobs_analysis.png")
plt.show()

# Test: co by było gdyby zwiększyć separację?
print("\n" + "="*60)
print("TEST: większa separacja (cluster_std=0.4)")
print("="*60)

X_blobs2, y_blobs2 = make_blobs(n_samples=500, centers=5, n_features=2,
                                 random_state=42, cluster_std=0.4)
kstar2 = KStarMeans(random_state=42)
kstar2.fit(X_blobs2)
print(f"Znaleziona k*: {kstar2.n_clusters_}")