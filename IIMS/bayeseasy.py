import os
os.chdir("C:/Users/jakub/Visual Studio Code/IIMS")             # Zmiana sciezki dostepu

import numpy as np                                             # Import bibliotek
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder                 # Uzyj tylko jesli dane zawieraja zmienne kategorialne
import matplotlib.pyplot as plt

class NaiveBayes:                               # Klasa NaiveBayes
    def fit(self, X, y):                        # Dopasowanie modelu do danych
        n_samples, n_features = X.shape         # Liczba probek i zmiennych
        self._classes = np.unique(y)            # Unikalne klasy
        n_classes = len(self._classes)          # Liczba klas

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)    # Srednie
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)     # Wariancje
        self._priors = np.zeros(n_classes, dtype=np.float64)                # Prawdopodobienstwa a priori

        for idx, c in enumerate(self._classes):                 # Iteracja po klasach y = unikalne klasy
            X_c = X[y == c]                                     # Wybor probek dla danej klasy
            self._mean[idx, :] = X_c.mean(axis=0)               # Obliczenie srednich
            self._var[idx, :] = X_c.var(axis=0)                 # Obliczenie wariancji
            self._priors[idx] = X_c.shape[0] / float(n_samples) # Obliczenie prawdopodobienstw a priori

    def predict_proba(self, X):                                 # Obliczenie prawdopodobienstw przynaleznosci do klas
        return np.array([self._predict_proba(x) for x in X])    # Zwrocenie wynikow w postaci tablicy x = l. probek

    def _predict_proba(self, x):                      # Obliczenie prawdopodobienstw przynaleznosci do klas dla pojedynczej probki
        posteriors = []                               # Lista prawdopodobienstw a posteriori

        for idx in range(len(self._classes)):                       # Iteracja po klasach y = unikalne klasy
            prior = np.log(self._priors[idx])                       # Logarytm prawdopodobienstwa a priori
            class_conditional = np.sum(np.log(self._pdf(idx, x)))   # Logarytm warunkowego prawdopodobienstwa klasy
            posterior = prior + class_conditional                   # Logarytm prawdopodobienstwa a posteriori
            posteriors.append(posterior)                            # Dodanie do listy

        posteriors = np.exp(posteriors)                             # Wyliczenie prawdopodobienstw a posteriori
        return posteriors / sum(posteriors)                         # Zwrocenie prawdopodobienstw przynaleznosci do klas

    def _pdf(self, class_idx, x):                           # Obliczenie funkcji gestosci prawdopodobienstwa
        mean = self._mean[class_idx]                        # Srednia
        var = self._var[class_idx]                          # Wariancja
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))  # Licznik
        denominator = np.sqrt(2 * np.pi * var)              # Mianownik
        return numerator / denominator                      # Wynik
    
if __name__ == "__main__":                                      # Głowna funkcja programu

    df = pd.read_csv('heartdisease.csv')                        # Odczyt danych z pliku CSV

    """le = LabelEncoder()                                      # Zamiana wartosci kategorialnych na liczbowe np. 'yes' -> 1, 'no' -> 0          
    df['num'] = le.fit_transform(df['num'])                     # Zamiana wartosci kategorialnych na liczbowe w tabeli 'num'"""

    correlations = df.drop('num', axis=1).apply(lambda x: x.corr(df['num'])).abs()  # Obliczenie korelacji miedzy zmiennymi a zmienna 'num'

    threshold = correlations.median()                                               # Obliczenie mediany korelacji

    to_drop = correlations[correlations < threshold].index.tolist() # Wybranie zmiennych o korelacji mniejszej niz mediana

    X = df.drop(['num'] + to_drop, axis=1).values                   # Wybranie zmiennych niezaleznych odrzucajac te o korelacji mniejszej niż mediana
    y = df['num'].values                                            # Wybranie zmiennej zaleznej 'num'

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)  # Podzial danych na zbior treningowy i testowy

    nb = NaiveBayes()                                   # Utworzenie obiektu klasy NaiveBayes
    nb.fit(X_train, y_train)                            # Dopasowanie modelu do danych treningowych
    probabilities = nb.predict_proba(X_test)            # Obliczenie prawdopodobienstw przynaleznosci do klas dla danych testowych

    prob_positive = probabilities[:, 1]                 # Prawdopodobienstwa przynależnosci do klasy pozytywnej

    plt.figure(figsize=(15,5))                              # Wykresy

    plt.subplot(1,3,1)
    plt.hist(prob_positive, bins=10, edgecolor='k')         # Histogram prawdopodobienstw przynaleznosci do klasy pozytywnej
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of heart disease')
    plt.ylabel('Frequency')

    plt.subplot(1,3,2)
    plt.hist(prob_positive, bins=10, edgecolor='k', cumulative=True)    # Dystrybuanta prawdopodobienstw przynaleznosci do klasy pozytywnej
    plt.title('Cumulative distribution of predicted probabilities')
    plt.xlabel('Predicted probability of heart disease')
    plt.ylabel('Cumulative frequency')

    plt.subplot(1,3,3)
    plt.plot(np.sort(prob_positive))                                    # Wykres iteracji i prawdopodobienstw przynaleznosci do klasy pozytywnej
    plt.title('Probability by iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Predicted probability of heart disease')

    plt.tight_layout()
    plt.show()