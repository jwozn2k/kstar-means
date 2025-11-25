import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from kstar_means import KStarMeans

np.random.seed(0)

# Przygotowanie klasycznych zbiorów benchmarkowych
n_samples = 500

# 1. Noisy circles
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)

# 2. Noisy moons
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)

# 3. Blobs - varied (różne odległości)
blobs_centers = [(-5, -5), (0, 0), (5, 5)]
varied = datasets.make_blobs(n_samples=n_samples, centers=blobs_centers,
                             cluster_std=0.5, random_state=0)

# 4. Anisotropic (wydłużone)
X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# 5. Regular blobs
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

# 6. No structure (uniform random)
no_structure = (np.random.rand(n_samples, 2), np.zeros(n_samples))

# Lista wszystkich zbiorów
datasets_list = [
    ('Noisy Circles', noisy_circles, 2),
    ('Noisy Moons', noisy_moons, 2),
    ('Varied Density', varied, 3),
    ('Anisotropic', aniso, 3),
    ('Blobs', blobs, 3),
    ('No Structure', no_structure, 1),
]

# Testowanie
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

print("=" * 70)
print("TEST K*-MEANS NA KLASYCZNYCH BENCHMARKACH")
print("=" * 70)

results = []

for idx, (name, (X_raw, y_true), k_true) in enumerate(datasets_list):
    # Normalizacja
    X = StandardScaler().fit_transform(X_raw)

    print(f"\n{idx + 1}. {name}")
    print("-" * 70)

    # K*-Means
    kstar = KStarMeans(random_state=0, patience=5)
    kstar.fit(X)
    pred = kstar.predict(X)

    k_found = kstar.n_clusters_

    print(f"Prawdziwe k: {k_true}")
    print(f"Znalezione k*: {k_found}")
    print(f"Iteracje: {len(kstar.cost_history_)}")

    # Rozmiary klastrów
    sizes = [np.sum(pred == i) for i in range(k_found)]
    print(f"Rozmiary klastrów: {sizes}")

    results.append({
        'name': name,
        'k_true': k_true,
        'k_found': k_found,
        'match': k_found == k_true
    })

    # Wizualizacja - prawdziwe etykiety
    ax = axes[idx * 2]
    if name == 'No Structure':
        ax.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, s=20)
    else:
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab10',
                             alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
    ax.set_title(f'{name}\nPrawdziwe (k={k_true})', fontsize=11)
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Wizualizacja - K*-Means
    ax = axes[idx * 2 + 1]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=pred, cmap='tab10',
                         alpha=0.6, s=20, edgecolors='k', linewidth=0.3)

    # Centroidy
    if k_found > 0:
        ax.scatter(kstar.centroids_[:, 0], kstar.centroids_[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidth=2,
                   label='Centroidy', zorder=10)

    # Kolor tytułu w zależności od wyniku
    if k_found == k_true:
        color = 'green'
        status = '✓'
    else:
        color = 'orange'
        status = f'✗ ({k_found - k_true:+d})'

    ax.set_title(f'K*-Means (k*={k_found}) {status}',
                 color=color, fontweight='bold', fontsize=11)
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

output_path = 'kstar_benchmarks.png'

plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n{'=' * 70}")
print(f"Wykres zapisany: {output_path}")

# Podsumowanie
print("\n" + "=" * 70)
print("PODSUMOWANIE")
print("=" * 70)
print(f"{'Dataset':<20} {'k prawdziwe':<15} {'k* znalezione':<15} {'Status':<15}")
print("-" * 70)

correct = 0
for r in results:
    status = '✓ DOBRZE' if r['match'] else f"✗ BŁĄD ({r['k_found'] - r['k_true']:+d})"
    if r['match']:
        correct += 1
    print(f"{r['name']:<20} {r['k_true']:<15} {r['k_found']:<15} {status:<15}")

accuracy = correct / len(results) * 100
print("-" * 70)
print(f"Dokładność: {correct}/{len(results)} ({accuracy:.1f}%)")

print("\n" + "=" * 70)
print("ANALIZA")
print("=" * 70)
print("K*-Means (jak standardowy K-Means) najlepiej działa na:")
print("  ✓ Sferycznych/okrągłych klastrach")
print("  ✓ Podobnej gęstości/wielkości klastrów")
print("  ✓ Dobrze rozdzielonych danych")
print("\nMa trudności z:")
print("  ✗ Nieregularnymi kształtami (moons, circles)")
print("  ✗ Różnymi gęstościami")
print("  ✗ Brakiem struktury")
print("\nTo ograniczenie założeń K-Means, nie specyficzne dla K*-Means!")

plt.show()