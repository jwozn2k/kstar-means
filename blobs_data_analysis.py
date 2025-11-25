import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from kstar_means import KStarMeans

np.random.seed(42)

# Generowanie danych
n_samples = 500
n_clusters_true = 5

X_blobs, y_blobs = make_blobs(
    n_samples=n_samples,
    centers=n_clusters_true,
    n_features=2,
    cluster_std=0.6,
    random_state=42
)

print("BLOBS DATASET")
print(f"Liczba punktów: {n_samples}")
print(f"Prawdziwa liczba klastrów: {n_clusters_true}")
print(f"Rozkład klas: {[np.sum(y_blobs == i) for i in range(n_clusters_true)]}")

# K*-Means
print("\nK*-Means klasteryzacja...")
kstar_blobs = KStarMeans(patience=5, convergence_threshold=2.0, max_iterations=1000, random_state=42)
kstar_blobs.fit(X_blobs)

k_found = kstar_blobs.n_clusters_
labels_pred = kstar_blobs.labels_

print(f"Znaleziona k*: {k_found}")
print(f"Iteracje: {len(kstar_blobs.cost_history_)}")
print(f"Rozmiary klastrów: {[np.sum(labels_pred == i) for i in range(k_found)]}")

# Metryki
ari = adjusted_rand_score(y_blobs, labels_pred)
nmi = normalized_mutual_info_score(y_blobs, labels_pred)
print(f"\nARI: {ari:.3f}")
print(f"NMI: {nmi:.3f}")

if k_found == n_clusters_true:
    cm = confusion_matrix(y_blobs, labels_pred)
    print(f"\nMacierz pomyłek:\n{cm}")

# Wykres 1: Koszt MDL
fig1 = plt.figure(figsize=(10, 6))
plt.plot(kstar_blobs.cost_history_, 'b-o', linewidth=2, markersize=6)
plt.xlabel('Iteracja', fontsize=12)
plt.ylabel('Koszt MDL', fontsize=12)
plt.title('Ewolucja kosztu MDL (Blobs)', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./img/blobs_mdl_cost.png', dpi=150, bbox_inches='tight')
print("\nZapisano: blobs_mdl_cost.png")

# Wykres 2: Klastry
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))

# Prawdziwe etykiety
axes[0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap='tab10',
                alpha=0.6, s=30, edgecolors='k', linewidth=0.3)
axes[0].set_title(f'Prawdziwe klastry (k={n_clusters_true})', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('Feature 1', fontsize=11)
axes[0].set_ylabel('Feature 2', fontsize=11)

# K*-Means
axes[1].scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels_pred, cmap='tab10',
                alpha=0.6, s=30, edgecolors='k', linewidth=0.3)
axes[1].scatter(kstar_blobs.centroids_[:, 0], kstar_blobs.centroids_[:, 1],
                c='red', marker='X', s=300, edgecolors='black', linewidth=2.5,
                label='Centroidy', zorder=10)
axes[1].set_title(f'K*-Means (k*={k_found})', fontsize=14, fontweight='bold',
                  color='green' if k_found == n_clusters_true else 'orange')
axes[1].legend(loc='best', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('Feature 1', fontsize=11)
axes[1].set_ylabel('Feature 2', fontsize=11)

plt.tight_layout()
plt.savefig('./img/blobs_clusters.png', dpi=150, bbox_inches='tight')
print("Zapisano: blobs_clusters.png")

# Test różnych k
print("\nTest różnych k dla K-Means:")
k_values = [3, 4, 5, 6, 7]
results = []

for k in k_values:
    kmeans_test = KMeans(n_clusters=k, random_state=42)
    kmeans_test.fit(X_blobs)
    labels_test = kmeans_test.labels_
    ari_test = adjusted_rand_score(y_blobs, labels_test)
    nmi_test = normalized_mutual_info_score(y_blobs, labels_test)
    results.append({'k': k, 'ARI': ari_test, 'NMI': nmi_test})

print(f"{'k':<5} {'ARI':<8} {'NMI':<8}")
for r in results:
    marker = ""
    if r['k'] == n_clusters_true:
        marker = " ← prawdziwe"
    if r['k'] == k_found:
        marker = f" ← K*-Means"
    print(f"{r['k']:<5} {r['ARI']:.3f}  {r['NMI']:.3f}{marker}")

# Wykres 3: Porównanie k
fig3 = plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
k_vals = [r['k'] for r in results]
ari_vals = [r['ARI'] for r in results]
plt.plot(k_vals, ari_vals, 'bo-', linewidth=2, markersize=10)
plt.axvline(n_clusters_true, color='blue', linestyle='--', linewidth=2, alpha=0.5,
            label=f'Prawdziwe k={n_clusters_true}')
plt.axvline(k_found, color='green', linestyle='--', linewidth=2,
            label=f'K*-Means (k*={k_found})')
plt.xlabel('Liczba klastrów k', fontsize=12)
plt.ylabel('Adjusted Rand Index', fontsize=12)
plt.title('Wpływ k na ARI', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xticks(k_vals)

plt.subplot(1, 2, 2)
nmi_vals = [r['NMI'] for r in results]
plt.plot(k_vals, nmi_vals, 'ro-', linewidth=2, markersize=10)
plt.axvline(n_clusters_true, color='blue', linestyle='--', linewidth=2, alpha=0.5,
            label=f'Prawdziwe k={n_clusters_true}')
plt.axvline(k_found, color='green', linestyle='--', linewidth=2,
            label=f'K*-Means (k*={k_found})')
plt.xlabel('Liczba klastrów k', fontsize=12)
plt.ylabel('Normalized Mutual Information', fontsize=12)
plt.title('Wpływ k na NMI', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xticks(k_vals)

plt.tight_layout()
plt.savefig('./img/blobs_k_comparison.png', dpi=150, bbox_inches='tight')
print("Zapisano: blobs_k_comparison.png")

print("\nGotowe!")
plt.show()