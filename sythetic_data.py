import numpy as np
import matplotlib.pyplot as plt
from kstar_means import KStarMeans

# Ustawienie seed dla powtarzalności
np.random.seed(42)

# Wygenerowanie prostych danych - 3 klastry w 2D
# Klaster 1: centrum w (0, 0)
cluster1 = np.random.randn(100, 2) * 0.5 + np.array([0, 0])

# Klaster 2: centrum w (5, 0)
cluster2 = np.random.randn(100, 2) * 0.5 + np.array([5, 0])

# Klaster 3: centrum w (2.5, 4)
cluster3 = np.random.randn(100, 2) * 0.5 + np.array([2.5, 4])

# Połączenie wszystkich danych
X = np.vstack([cluster1, cluster2, cluster3])
true_labels = np.array([0]*100 + [1]*100 + [2]*100)

print("="*60)
print("Test K*-Means na prostych danych syntetycznych")
print("="*60)
print(f"\nRozmiar danych: {X.shape}")
print(f"Prawdziwa liczba klastrów: 3")
print(f"Punkty w każdym klastrze: 100")

# Utworzenie i dopasowanie modelu
print("\nTrening K*-Means...")
kstar = KStarMeans(patience=5, random_state=42)
kstar.fit(X)

print(f"\n{'='*60}")
print("WYNIKI")
print(f"{'='*60}")
print(f"Znaleziona liczba klastrów k*: {kstar.n_clusters_}")
print(f"Liczba iteracji: {len(kstar.cost_history_)}")
print(f"Końcowy koszt MDL: {kstar.cost_history_[-1]:.2f}")

# Predykcja
predicted_labels = kstar.predict(X)

print(f"\nCentroidy klastrów:")
for i, centroid in enumerate(kstar.centroids_):
    print(f"  Klaster {i}: [{centroid[0]:.2f}, {centroid[1]:.2f}]")

print(f"\nRozmiary znalezionych klastrów:")
for i in range(kstar.n_clusters_):
    count = np.sum(predicted_labels == i)
    print(f"  Klaster {i}: {count} punktów")

# Wizualizacja
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Prawdziwe etykiety
ax = axes[0]
scatter = ax.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='viridis',
                     alpha=0.6, edgecolors='k', linewidth=0.5)
ax.set_title('Prawdziwe klastry (3 klastry)', fontsize=14, fontweight='bold')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Klaster')

# 2. Predykcje K*-Means
ax = axes[1]
scatter = ax.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis',
                     alpha=0.6, edgecolors='k', linewidth=0.5)
# Zaznaczenie centroidów
ax.scatter(kstar.centroids_[:, 0], kstar.centroids_[:, 1],
          c='red', marker='X', s=200, edgecolors='black', linewidth=2,
          label='Centroidy')
ax.set_title(f'K*-Means (k* = {kstar.n_clusters_})', fontsize=14, fontweight='bold')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.grid(True, alpha=0.3)
ax.legend()
plt.colorbar(scatter, ax=ax, label='Klaster')

# 3. Historia kosztu MDL
ax = axes[2]
ax.plot(kstar.cost_history_, linewidth=2, color='blue', marker='o', markersize=4)
ax.set_title('Historia kosztu MDL', fontsize=14, fontweight='bold')
ax.set_xlabel('Iteracja')
ax.set_ylabel('Koszt MDL')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('kstar_simple_test.png', dpi=150, bbox_inches='tight')
print(f"\nWykres zapisany jako: kstar_simple_test.png")

# Dodatkowa analiza - porównanie z prawdziwymi klastrami
print(f"\n{'='*60}")
print("ANALIZA ZGODNOŚCI")
print(f"{'='*60}")

# Sprawdzenie czystości klastrów (purity)
def purity_score(true_labels, pred_labels):
    """Oblicza czystość klastrowania"""
    contingency_matrix = np.zeros((len(np.unique(true_labels)),
                                   len(np.unique(pred_labels))))
    for i, true_label in enumerate(np.unique(true_labels)):
        for j, pred_label in enumerate(np.unique(pred_labels)):
            contingency_matrix[i, j] = np.sum((true_labels == true_label) &
                                              (pred_labels == pred_label))
    return np.sum(np.max(contingency_matrix, axis=0)) / len(true_labels)

purity = purity_score(true_labels, predicted_labels)
print(f"Czystość klastrowania (purity): {purity:.3f}")
print(f"  (1.0 = perfekcyjne dopasowanie, 0.33 = losowe dla 3 klastrów)")

# Macierz pomyłek
print(f"\nMacierz przypisań (prawdziwe vs predykcja):")
print("Wiersze = prawdziwe klastry, Kolumny = predykcje K*-Means")
contingency = np.zeros((3, kstar.n_clusters_), dtype=int)
for true_cluster in range(3):
    for pred_cluster in range(kstar.n_clusters_):
        count = np.sum((true_labels == true_cluster) &
                       (predicted_labels == pred_cluster))
        contingency[true_cluster, pred_cluster] = count

for i in range(3):
    print(f"  Prawdziwy klaster {i}: {contingency[i]}")

print(f"\n{'='*60}")