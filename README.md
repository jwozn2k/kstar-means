# K*-Means - Klasteryzacja bez parametru k

Implementacja algorytmu K*-Means, który automatycznie znajduje optymalną liczbę klastrów używając zasady MDL (Minimum Description Length).

## Instalacja

```bash
git clone https://github.com/jwozn2k/kstar-means.git
cd kstar-means
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Podstawowe użycie

```python
from kstar_means import KStarMeans
import numpy as np

X = np.array([[1, 2], [1, 4], [10, 2], [10, 4]])

kstar = KStarMeans(random_state=42)
kstar.fit(X)

print(f"Znaleziona liczba klastrów: {kstar.n_clusters_}")
print(f"Etykiety: {kstar.labels_}")
```

## Testy i przykłady

### Zbiór Iris
```bash
python iris_data_analysis.py
```
Analiza na zbiorze Iris (3 gatunki kwiatów). Sprawdza czy algorytm automatycznie znajdzie k=3.

Generuje:
- `img/iris_mdl_cost.png` - ewolucja kosztu MDL
- `img/iris_clusters.png` - wizualizacja klastrów (PCA 2D)
- `img/iris_k_comparison.png` - porównanie z różnymi k w K-Means

### Dane syntetyczne (blobs)
```bash
python blobs_data_analysis.py
```
Test na 5 sztucznych klastrach. Sprawdza czy K*-Means prawidłowo wykryje wszystkie grupy.

Generuje:
- `img/blobs_mdl_cost.png` - ewolucja kosztu MDL
- `img/blobs_clusters.png` - wizualizacja klastrów
- `img/blobs_k_comparison.png` - porównanie z różnymi k

### Benchmarki
```bash
python test_benchmarks.py
```
Test na 6 klasycznych zbiorach (circles, moons, blobs z różną gęstością, etc.).

Generuje:
- `img/kstar_benchmarks.png` - porównanie na wszystkich zbiorach

## Parametry

```python
KStarMeans(
    patience=5,                 # ile iteracji bez poprawy przed stop
    convergence_threshold=2.0,  # minimalna poprawa kosztu
    max_iterations=1000,        # max iteracji
    random_state=None           # seed dla powtarzalności
)
```

## Atrybuty po fit()

- `n_clusters_` - znaleziona liczba klastrów
- `labels_` - przypisania punktów do klastrów
- `centroids_` - współrzędne centroidów
- `cost_history_` - historia kosztu MDL

## Wyniki

Po uruchomieniu testów:
- **Iris**: K*-Means znalazł k*=3 (prawidłowe), ARI≈0.64
- **Blobs (5 klastrów)**: K*-Means znalazł k*=4-5 (zależy od random_state)
- **Benchmarki**: Najlepsze wyniki na regularnych, sferycznych klastrach

## Kiedy używać?

**Działa dobrze:**
- Nie wiesz ile jest klastrów
- Klastry są sferyczne i dobrze rozdzielone
- Podobna wielkość i gęstość klastrów

**Może nie działać:**
- Nieregularne kształty (moons, circles)
- Bardzo różne gęstości
- Silnie nakładające się grupy

## Struktura projektu

```
kstar-means/
├── kstar_means.py            # główna implementacja
├── iris_data_analysis.py     # test na Iris
├── blobs_data_analysis.py    # test na danych syntetycznych
├── test_benchmarks.py        # testy benchmarkowe
├── img/                      # wygenerowane wykresy
├── requirements.txt          # zależności
├── README.md                 # ten plik
├── DESC.md                   # opis teoretyczny
└── EXAMPLE.md                # szczegółowy przykład
```

## Więcej informacji

- `DESC.md` - jak działa algorytm, porównanie z K-Means
- `EXAMPLE.md` - szczegółowy przykład analizy na Iris krok po kroku

## Licencja

MIT License
