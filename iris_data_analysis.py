"""
Szczegółowa analiza algorytmu K*-Means na zbiorze Iris.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from kstar_means import KStarMeans

np.random.seed(42)

# Wczytanie i przygotowanie danych
iris = load_iris()
X = iris.data
y_true = iris.target
feature_names = iris.feature_names
species_names = iris.target_names

print("IRIS DATASET")
print(f"Kształt danych: {X.shape}")
print(f"Prawdziwa liczba klastrów: 3")

# Normalizacja
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K*-Means
print("\nK*-Means klasteryzacja...")
kstar = KStarMeans(patience=5, convergence_threshold=2.0, max_iterations=1000, random_state=42)
kstar.fit(X_scaled)

k_star = kstar.n_clusters_
labels_pred = kstar.labels_
centroids = kstar.centroids_

print(f"Znaleziona k*: {k_star}")
print(f"Iteracje: {len(kstar.cost_history_)}")
print(f"Rozmiary klastrów: {[np.sum(labels_pred == i) for i in range(k_star)]}")

# Metryki
ari = adjusted_rand_score(y_true, labels_pred)
nmi = normalized_mutual_info_score(y_true, labels_pred)
print(f"\nARI: {ari:.3f}")
print(f"NMI: {nmi:.3f}")

# Macierz pomyłek
cm = confusion_matrix(y_true, labels_pred)
print(f"\nMacierz pomyłek:\n{cm}")

# Analiza po gatunkach
print("\nDokładność per gatunek:")
for i, name in enumerate(species_names):
    correct = cm[i, :].max()
    total = cm[i, :].sum()
    print(f"  {name}: {correct}/{total} ({correct/total*100:.1f}%)")

# Wykres 1: Koszt MDL
fig1 = plt.figure(figsize=(10, 6))
plt.plot(kstar.cost_history_, 'b-o', linewidth=2, markersize=6)
plt.xlabel('Iteracja', fontsize=12)
plt.ylabel('Koszt MDL', fontsize=12)
plt.title('Ewolucja kosztu MDL (Iris)', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./img/iris_mdl_cost.png', dpi=150, bbox_inches='tight')
print("\nZapisano: iris_mdl_cost.png")

# Wykres 2: Klastry (PCA 2D)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
centroids_2d = pca.transform(centroids)
variance_explained = pca.explained_variance_ratio_

fig2, axes = plt.subplots(1, 2, figsize=(16, 6))

# Prawdziwe etykiety
axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='tab10',
                alpha=0.7, s=60, edgecolors='k', linewidth=0.5)
axes[0].set_title('Prawdziwe gatunki (k=3)', fontsize=14, fontweight='bold')
axes[0].set_xlabel(f'PC1 ({variance_explained[0]:.1%})', fontsize=11)
axes[0].set_ylabel(f'PC2 ({variance_explained[1]:.1%})', fontsize=11)
axes[0].grid(True, alpha=0.3)
for i, name in enumerate(species_names):
    axes[0].scatter([], [], c=f'C{i}', label=name, s=60, edgecolors='k', linewidth=0.5)
axes[0].legend(loc='best', fontsize=10)

# K*-Means predykcje
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=labels_pred, cmap='tab10',
                alpha=0.7, s=60, edgecolors='k', linewidth=0.5)
axes[1].scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                c='red', marker='X', s=300, edgecolors='black',
                linewidth=2.5, label='Centroidy', zorder=10)
axes[1].set_title(f'K*-Means (k*={k_star})', fontsize=14, fontweight='bold',
                  color='green' if k_star == 3 else 'orange')
axes[1].set_xlabel(f'PC1 ({variance_explained[0]:.1%})', fontsize=11)
axes[1].set_ylabel(f'PC2 ({variance_explained[1]:.1%})', fontsize=11)
axes[1].legend(loc='best', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./img/iris_clusters.png', dpi=150, bbox_inches='tight')
print("Zapisano: iris_clusters.png")

# Porównanie z K-Means
print("\nPorównanie z K-Means:")
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels_kmeans = kmeans.labels_

ari_kmeans = adjusted_rand_score(y_true, labels_kmeans)
nmi_kmeans = normalized_mutual_info_score(y_true, labels_kmeans)

print(f"{'Metoda':<15} {'k':<5} {'ARI':<8} {'NMI':<8}")
print(f"{'K*-Means':<15} {k_star:<5} {ari:.3f}  {nmi:.3f}")
print(f"{'K-Means (k=3)':<15} {'3':<5} {ari_kmeans:.3f}  {nmi_kmeans:.3f}")

# Test różnych k
print("\nTest różnych k dla K-Means:")
k_values = [2, 3, 4, 5]
results = []

for k in k_values:
    kmeans_test = KMeans(n_clusters=k, random_state=42)
    kmeans_test.fit(X_scaled)
    labels_test = kmeans_test.labels_
    ari_test = adjusted_rand_score(y_true, labels_test)
    nmi_test = normalized_mutual_info_score(y_true, labels_test)
    results.append({'k': k, 'ARI': ari_test, 'NMI': nmi_test})

print(f"{'k':<5} {'ARI':<8} {'NMI':<8}")
for r in results:
    marker = " ← optymalne" if r['k'] == 3 else ""
    print(f"{r['k']:<5} {r['ARI']:.3f}  {r['NMI']:.3f}{marker}")

# Wykres 3: Porównanie k
fig3 = plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
k_vals = [r['k'] for r in results]
ari_vals = [r['ARI'] for r in results]
plt.plot(k_vals, ari_vals, 'bo-', linewidth=2, markersize=10)
plt.axvline(k_star, color='green', linestyle='--', linewidth=2, label=f'K*-Means (k*={k_star})')
plt.xlabel('Liczba klastrów k', fontsize=12)
plt.ylabel('Adjusted Rand Index', fontsize=12)
plt.title('Wpływ k na ARI', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xticks(k_vals)

plt.subplot(1, 2, 2)
nmi_vals = [r['NMI'] for r in results]
plt.plot(k_vals, nmi_vals, 'ro-', linewidth=2, markersize=10)
plt.axvline(k_star, color='green', linestyle='--', linewidth=2, label=f'K*-Means (k*={k_star})')
plt.xlabel('Liczba klastrów k', fontsize=12)
plt.ylabel('Normalized Mutual Information', fontsize=12)
plt.title('Wpływ k na NMI', fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xticks(k_vals)

plt.tight_layout()
plt.savefig('./img/iris_k_comparison.png', dpi=150, bbox_inches='tight')
print("Zapisano: iris_k_comparison.png")

print("\nGotowe!")
plt.show()