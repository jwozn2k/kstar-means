# K*-Means - Opis algorytmu

## Problem

W klasteryzacji często nie wiemy z góry ile jest grup w danych. Klasyczny K-Means wymaga podania parametru k, co prowadzi do:
- Męczącego testowania różnych wartości k
- Subiektywnej interpretacji metody łokcia czy silhouette score
- Ryzyka złego wyboru (za mało klastrów spowoduje nadmierne uogólnienie, a za dużo klastrów overfitting)

**K\*-Means rozwiązuje ten problem** - automatycznie znajduje optymalną liczbę klastrów.

## K-Means (klasyczny)

1. Wybieramy k (często problematyczne)
2. Losowo inicjalizujemy k centroidów
3. Powtarzamy:
   - **ASSIGN**: przypisanie punktów do najbliższych centroidów
   - **UPDATE**: przesunięcie centroidów do środka przypisanych punktów
4. Stop gdy centroidy się nie zmieniają

**Zalety**: prosty, szybki, działa dobrze na sferycznych klastrach

**Wady**: musimy z góry wiedzieć ile jest klastrów

## K*-Means

K*-Means dodaje do K-Means dwie operacje:
- **SPLIT**: dzieli klastry, jeśli to opłacalne
- **MERGE**: łączy klastry, jeśli to opłacalne

Decyzja o "opłacalności" opiera się na **MDL (Minimum Description Length)** - zasadzie z teorii informacji mówiącej, że najlepszy model to taki, który minimalizuje całkowity koszt zakodowania danych.

### Koszt MDL

```
MDL = model_cost + index_cost + residual_cost
```

1. **Model cost**: koszt przechowania k centroidów
   - `k × d × floatcost`, gdzie floatcost zależy od zakresu danych i precyzji
   - Więcej klastrów --> większy model --> wyższy koszt

2. **Index cost**: koszt przypisania każdego punktu do klastra
   - `n × log₂(k)` bitów
   - Więcej klastrów --> wyższy koszt indeksowania

3. **Residual cost**: koszt zakodowania odchyleń od centroidów
   - `n × d × log(2π) + Σ(odległości²) / 2`
   - Mniej klastrów --> gorsze dopasowanie → wyższy koszt

**Balans**: Za mało klastrów → wysoki residual_cost. Za dużo klastrów → wysoki model_cost + index_cost.

Algorytm szuka k, które minimalizuje całkowity koszt MDL.

### Przebieg algorytmu

```
1. Start z k=1 (wszystkie punkty w jednym klastrze)
   Dla każdego klastra trzymamy 2 "pod-centroidy"

2. Pętla:
   a) Krok K-Means (ASSIGN + UPDATE)

   b) SPLIT: Sprawdź każdy klaster
      - Czy podział na 2 pod-centroidy zmniejszy koszt MDL?
      - Jeśli tak: podziel (k := k+1)

   c) Jeśli nie było podziału:
      - Wykonaj krok K-Means
      - MERGE: Znajdź najbliższą parę klastrów
      - Czy scalenie zmniejszy koszt MDL?
      - Jeśli tak: scal (k := k-1)

   d) Oblicz całkowity koszt MDL

   e) Jeśli brak poprawy przez 'patience' iteracji to stop

3. Zwróć znalezione k* i przypisania
```

### Kluczowe różnice vs K-Means

| Cecha | K-Means | K*-Means |
|-------|---------|----------|
| Liczba klastrów | **Musisz podać k** | **Automatycznie znajduje k*** |
| Inicjalizacja | k centroidów losowo | 1 klaster → stopniowy podział |
| Operacje | ASSIGN + UPDATE | ASSIGN + UPDATE + SPLIT + MERGE |
| Kryterium | Minimalizuj wariancję | Minimalizuj MDL |
| Gwarancja zbieżności | Tak (do lokalnego minimum) | Heurystyka (patience) |

## Kiedy to działa

**Dobrze:**
- Sferyczne, zwarte klastry
- Podobna wielkość i gęstość klastrów
- Wyraźna separacja między grupami
- Przykład: 5 oddzielnych chmur punktów

**Słabo:**
- Nieregularne kształty (półksiężyce, pierścienie)
- Bardzo różne gęstości (duży rzadki klaster + mały gęsty)
- Silne nakładanie się grup
- Brak struktury (czysty szum)

To są **te same ograniczenia co K-Means** - K*-Means buduje na K-Means, więc dziedziczy jego założenia geometryczne.

## Wyniki z testów

Po uruchomieniu `iris_data_analysis.py`, `blobs_data_analysis.py` i `test_benchmarks.py`:

### Iris (3 gatunki)
- Prawdziwe k: 3
- Znalezione k*: 3 ✓
- ARI: 0.645 (dobre dopasowanie)
- Setosa doskonale rozdzielona, Versicolor/Virginica trochę się mieszają

### Blobs (5 sztucznych klastrów)
- Prawdziwe k: 5
- Znalezione k*: 4-5 (zależy od random_state)
- Czasem scala dwa blisko siebie klastry

### Benchmarki (6 różnych zbiorów)
- **Varied density**: ✓ k* = 3 = k_true
- **Moons**: ✓ k* = 2 = k_true
- **Circles**: ✗ k* = 3, k_true = 2 (nieregularny kształt)
- **Blobs**: ✗ k* = 2, k_true = 3 (trudny przypadek)
- **Anisotropic**: ✗ k* = 2, k_true = 3
- **No Structure**: ✗ k* = 3, k_true = 1 (szum)

**Wniosek**: K*-Means działa świetnie na danych które pasują do założeń K-Means (sferyczne klastry). Dla dziwnych kształtów lepiej użyć DBSCAN czy spectral clustering.

## Praktyczne zastosowania

- **Segmentacja klientów**: nie wiemy na ile grup podzielić klientów
- **Analiza obrazów**: grupowanie podobnych kolorów/tekstur
- **Kompresja danych**: reprezentacja danych przez centroidy

## Podsumowanie

K*-Means to rozszerzenie K-Means które:
- ✅ Automatycznie znajduje k (nie musisz zgadywać!)
- ✅ Używa solidnej teorii (MDL)
- ✅ Daje wyniki porównywalne z K-Means gdy k jest znane
- ⚠️ Ma te same ograniczenia co K-Means (sferyczne klastry)

K*-Means to dobry wybór, jesli mamy sferyczne i dobrze rozdzielone klastry, ale nie wiemy ile ich jest.

## Źródło

Algorytm opisany w: Mahon, L., & Lapata, M. (2025). "K*-Means: A Parameter-free Clustering Algorithm"