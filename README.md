# MACD

Repozytorium zawiera algorytm do handlu oparty na wskaźniku MACD (Moving Average Convergence Divergence) napisany w języku Python.

## Opis

Projekt umożliwia analizę danych giełdowych z wykorzystaniem wskaźnika MACD, generowanie sygnałów kupna/sprzedaży oraz symulację strategii inwestycyjnych.

## Funkcje

- Obliczanie wskaźnika MACD dla wybranych danych
- Generowanie sygnałów kupna/sprzedaży na podstawie przecięć MACD i sygnału
- Możliwość testowania strategii na historycznych danych

## Wymagania

- Python 
- Pandas
- Numpy
- Matplotlib (opcjonalnie, do wizualizacji)

Instalacja zależności:
```bash
pip install -r requirements.txt
```

## Użycie

Przykład użycia znajduje się w pliku `main.py`:

```bash
python main.py
```

## Struktura repozytorium

- `main.py` – główny plik uruchamiający strategię
- `data/` – przykładowe dane giełdowe
- `README.md` – ten plik
- `MACD.pdf` – sprawozdanie

## Licencja

Projekt dostępny na licencji MIT.

