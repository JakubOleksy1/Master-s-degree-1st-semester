import os
os.chdir("C:/Users/jakub/Visual Studio Code/IIMS")             # Zmiana sciezki dostepu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVC

def discretize_prediction(prediction):
    possible_values = [0, 25, 50, 75, 100]
    return min(possible_values, key=lambda x: abs(x - prediction))

def calculate_squared_errors(model, X_train, y_train, X_test, y_test, errVar):
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    prob_positive = probabilities[:, 1]
    squared_errors = []

    for i in range(errVar, len(X_train)):
        model.fit(X_train[:i], y_train[:i])
        probabilities = model.predict_proba(X_test)

        if probabilities.shape[1] == 1:
            continue

        prob_positive = probabilities[:, 1]
        squared_error = np.mean((prob_positive - y_test / 100) ** 2)
        squared_errors.append(squared_error)

    return prob_positive, squared_errors

def plot_probabilities(prob_positive):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(prob_positive, bins=10, edgecolor='k', density=True)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of heart disease')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(prob_positive, bins=10, edgecolor='k', cumulative=True, density=True)
    plt.title('Cumulative distribution of predicted probabilities')
    plt.xlabel('Predicted probability of heart disease')
    plt.ylabel('Cumulative frequency')

    plt.subplot(1, 3, 3)
    plt.plot(np.sort(prob_positive))
    plt.title('Probability by iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Predicted probability of heart disease')

    plt.tight_layout()
    plt.show()

def calculate_mae(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae

def calculate_r2_errors(model, X_train, y_train, X_test, y_test, errVar=5):
    r2_errors = []
    for i in range(errVar, len(X_train)):
        model.fit(X_train[:i], y_train[:i])
        predictions = model.predict(X_test)
        r2_error = r2_score(y_test, predictions)
        r2_errors.append(r2_error)
    return r2_errors

class SupportVectorMachine:
    def __init__(self, kernel='rbf', probability=True):
        self._model = SVC(kernel=kernel, probability=probability)

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict_proba(self, X):
        return np.array([self._predict_proba(x) for x in X])

    def _predict_proba(self, x):
        posteriors = []

        for idx in range(len(self._classes)):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        posteriors = np.exp(posteriors)
        return posteriors / sum(posteriors)

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx] + 1e-9
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return (numerator / denominator) + 1e-9

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

if __name__ == "__main__":
    df = pd.read_csv('heartdisease.csv')
    df['num'] = (df['num'] / 4) * 100  # 1-4 -> 0-100

    errVar = 5

    correlations = df.drop('num', axis=1).apply(lambda x: x.corr(df['num'])).abs()
    threshold = correlations.median()
    to_drop = correlations[correlations < threshold].index.tolist()

    X = df.drop(['num'] + to_drop, axis=1).values
    y = df['num'].values

    max_value_df = df['num'].max()
    print("Max value of probability", max_value_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    probabilities = nb.predict_proba(X_test)
    prob_positive_nb, squared_errors_nb = calculate_squared_errors(nb, X_train, y_train, X_test, y_test, errVar)
    plot_probabilities(prob_positive_nb)

    svm = SupportVectorMachine()
    svm.fit(X_train, y_train)
    probabilities = svm.predict_proba(X_test)
    prob_positive_svm, squared_errors_svm = calculate_squared_errors(svm, X_train, y_train, X_test, y_test, errVar)
    plot_probabilities(prob_positive_svm)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    probabilities = rf.predict_proba(X_test)
    prob_positive_rf, squared_errors_rf = calculate_squared_errors(rf, X_train, y_train, X_test, y_test, errVar)
    plot_probabilities(prob_positive_rf)

    plt.figure(figsize=(10, 6))
    plt.plot(range(errVar, len(X_train)), squared_errors_nb, label='Naive Bayes')
    plt.plot(range(errVar, len(X_train)), squared_errors_svm, label='SVM')
    plt.plot(range(errVar, len(X_train)), squared_errors_rf, label='Random Forest')
    plt.title('Squared error by number of training samples')
    plt.xlabel('Iteration')
    plt.ylabel('Squared error')
    plt.legend()
    plt.show()

    r2_errors_nb = calculate_r2_errors(nb, X_train, y_train, X_test, y_test)
    r2_errors_svm = calculate_r2_errors(svm, X_train, y_train, X_test, y_test)
    r2_errors_rf = calculate_r2_errors(rf, X_train, y_train, X_test, y_test)

    plt.figure(figsize=(10, 6))
    plt.plot(range(errVar, len(X_train)), r2_errors_nb, label='Naive Bayes')
    plt.plot(range(errVar, len(X_train)), r2_errors_svm, label='SVM')
    plt.plot(range(errVar, len(X_train)), r2_errors_rf, label='Random Forest')
    plt.xlabel('Iteration')
    plt.ylabel('R^2 Error')
    plt.title('R^2 Error by Number of Training Samples')
    plt.legend()
    plt.show()

    svm_mae = calculate_mae(svm, X_train, y_train, X_test, y_test)
    naive_bayes_mae = calculate_mae(nb, X_train, y_train, X_test, y_test)
    random_forest_mae = calculate_mae(rf, X_train, y_train, X_test, y_test)

    plt.figure(figsize=(10, 5))
    plt.bar(['SVM', 'Naive Bayes', 'Random Forest'], [svm_mae, naive_bayes_mae, random_forest_mae])
    plt.title('Mean Absolute Error of Different Models')
    plt.ylabel('Mean Absolute Error')
    plt.show()

    # Discretize predictions
    discretized_nb_predictions = np.array([discretize_prediction(pred) for pred in nb.predict(X_test)])
    discretized_svm_predictions = np.array([discretize_prediction(pred) for pred in svm.predict(X_test)])
    discretized_rf_predictions = np.array([discretize_prediction(pred) for pred in rf.predict(X_test)])

    discretized_nb_squared = []
    discretized_svm_squared =  []
    discretized_rf_squared =  []

    discretized_nb_r2 =  []
    discretized_svm_r2 =  []
    discretized_rf_r2 =  []

    for i in range(errVar, len(X_train)):

        discretized_nb_squared_errors = np.mean((discretized_nb_predictions - y_test) ** 2)
        discretized_svm_squared_errors = np.mean((discretized_svm_predictions - y_test) ** 2)
        discretized_rf_squared_errors = np.mean((discretized_rf_predictions - y_test) ** 2)

        discretized_nb_r2_errors = r2_score(y_test, discretized_nb_predictions)
        discretized_svm_r2_errors = r2_score(y_test, discretized_svm_predictions)
        discretized_rf_r2_errors = r2_score(y_test, discretized_rf_predictions)

        discretized_nb_squared.append(discretized_nb_squared_errors)
        discretized_svm_squared.append(discretized_svm_squared_errors)
        discretized_rf_squared.append(discretized_rf_squared_errors)

        discretized_nb_r2.append(discretized_nb_r2_errors)
        discretized_svm_r2.append(discretized_svm_r2_errors)
        discretized_rf_r2.append(discretized_rf_r2_errors)

    plt.figure(figsize=(10, 6))
    plt.plot(range(errVar, len(X_train)), discretized_nb_squared, label='Naive Bayes')
    plt.plot(range(errVar, len(X_train)), discretized_svm_squared, label='SVM')
    plt.plot(range(errVar, len(X_train)), discretized_rf_squared, label='Random Forest')
    plt.title('Squared error by number of training samples')
    plt.xlabel('Iteration')
    plt.ylabel('Squared error')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(errVar, len(X_train)), discretized_nb_r2, label='Naive Bayes')
    plt.plot(range(errVar, len(X_train)), discretized_svm_r2, label='SVM')
    plt.plot(range(errVar, len(X_train)), discretized_rf_r2, label='Random Forest')
    plt.xlabel('Iteration')
    plt.ylabel('R^2 Error')
    plt.title('R^2 Error by Number of Training Samples')
    plt.legend()
    plt.show()
    
    discretized_nb_mae = mean_absolute_error(y_test, discretized_nb_predictions)
    discretized_svm_mae = mean_absolute_error(y_test, discretized_svm_predictions)
    discretized_rf_mae = mean_absolute_error(y_test, discretized_rf_predictions)

    plt.figure(figsize=(10, 5))
    plt.bar(['Naive Bayes', 'SVM', 'Random Forest'], [discretized_nb_mae, discretized_svm_mae, discretized_rf_mae])
    plt.title('Mean Absolute Error of Different Models')
    plt.ylabel('Mean Absolute Error')
    plt.show()

    # Wykresy dla modeli z dyskretyzacja 2 sa stale R^2 i squared error MAE nie dziala bo sa iteracje wiec albo zmien na 1 wynik albo na iteracje ten poprzedni
"""
import numpy as np                                             # Import bibliotek
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder                 # Uzyj tylko jesli dane zawieraja zmienne kategorialne
from sklearn.svm import SVC

def discretize_prediction(prediction):
    possible_values = [0, 25, 50, 75, 100]
    return min(possible_values, key=lambda x:abs(x-prediction))

def calculate_squared_errors(model, X_train, y_train, X_test, y_test, errVar):
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)
    prob_positive = probabilities[:, 1]
    squared_errors = []
    
    for i in range(errVar, len(X_train)):
        model.fit(X_train[:i], y_train[:i])
        probabilities = model.predict_proba(X_test)
        
        if probabilities.shape[1] == 1:
            continue

        prob_positive = probabilities[:, 1]
        squared_error = np.mean((prob_positive - y_test/100) ** 2)
        squared_errors.append(squared_error)
    
    return prob_positive, squared_errors

def plot_probabilities(prob_positive):
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.hist(prob_positive, bins=10, edgecolor='k', density=True)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability of heart disease')
    plt.ylabel('Frequency')

    plt.subplot(1,3,2)
    plt.hist(prob_positive, bins=10, edgecolor='k', cumulative=True, density=True)
    plt.title('Cumulative distribution of predicted probabilities')
    plt.xlabel('Predicted probability of heart disease')
    plt.ylabel('Cumulative frequency')

    plt.subplot(1,3,3)
    plt.plot(np.sort(prob_positive))
    plt.title('Probability by iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Predicted probability of heart disease')

    plt.tight_layout()
    plt.show()

def calculate_mae(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae

def calculate_r2_errors(model, X_train, y_train, X_test, y_test, errVar=5):
    r2_errors = []
    for i in range(errVar, len(X_train)):
        model.fit(X_train[:i], y_train[:i])
        predictions = model.predict(X_test)
        r2_error = r2_score(y_test, predictions)
        r2_errors.append(r2_error)
    return r2_errors

class SupportVectorMachine:
    def __init__(self, kernel='rbf', probability=True):
        self._model = SVC(kernel=kernel, probability=probability)

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

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
        var = self._var[class_idx] + 1e-9                   # Wariancja 
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))  # Licznik
        denominator = np.sqrt(2 * np.pi * var)              # Mianownik
        return (numerator / denominator) + 1e-9             # Wynik 
    
    def predict(self, X):                                   # Przewidywanie klas
        proba = self.predict_proba(X)                       # Obliczenie prawdopodobienstw przynaleznosci do klas
        return np.argmax(proba, axis=1)                     # Zwrocenie indeksu najwiekszego prawdopodobienstwa

if __name__ == "__main__":                                      # Głowna funkcja programu

    df = pd.read_csv('heartdisease.csv')                        # Odczyt danych z pliku CSV
    
    df['num'] = (df['num'] / 4) * 100   # 1-4 -> 0-100          # Zamiana wartosci zmiennej 'num' na skale 0-100
    
    errVar = 5                                                  # Zmienna okreslajaca liczbe probek do obliczenia bledu
    
    #le = LabelEncoder()                                      # Zamiana wartosci kategorialnych na liczbowe np. 'yes' -> 1, 'no' -> 0          
    #df['num'] = le.fit_transform(df['num'])                     # Zamiana wartosci kategorialnych na liczbowe w tabeli 'num'

    correlations = df.drop('num', axis=1).apply(lambda x: x.corr(df['num'])).abs()  # Obliczenie korelacji miedzy zmiennymi a zmienna 'num'

    threshold = correlations.median()                                               # Obliczenie mediany korelacji

    to_drop = correlations[correlations < threshold].index.tolist() # Wybranie zmiennych o korelacji mniejszej niz mediana

    X = df.drop(['num'] + to_drop, axis=1).values                   # Wybranie zmiennych niezaleznych odrzucajac te o korelacji mniejszej niż mediana
    y = df['num'].values                                            # Wybranie zmiennej zaleznej 'num'

    max_value_df = df['num'].max()                                  # Obliczenie maksymalnej wartosci zmiennej 'num'
    print("Max value of probability", max_value_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)  # Podzial danych na zbior treningowy i testowy
    
    nb = NaiveBayes()                                   # Utworzenie obiektu klasy NaiveBayes
    nb.fit(X_train, y_train)                            # Dopasowanie modelu do danych treningowych
    probabilities = nb.predict_proba(X_test) 
    prob_positive_nb, squared_errors_nb = calculate_squared_errors(nb, X_train, y_train, X_test, y_test, errVar)    # Obliczenie bledu sredniokwadratowego
    plot_probabilities(prob_positive_nb)                                                                            # Wykres prawdopodobienstw

    svm = SupportVectorMachine()                                                                                    # Utworzenie obiektu klasy SupportVectorMachine
    svm.fit(X_train, y_train)                                                                                       # Dopasowanie modelu do danych treningowych
    probabilities = svm.predict_proba(X_test)                                                                       # Obliczenie prawdopodobienstw przynaleznosci do klas
    prob_positive_svm, squared_errors_svm = calculate_squared_errors(svm, X_train, y_train, X_test, y_test, errVar) # Obliczenie bledu sredniokwadratowego
    plot_probabilities(prob_positive_svm)                                                                           # Wykres prawdopodobienstw

    rf = RandomForestClassifier()                                                                                # Utworzenie obiektu klasy RandomForestClassifier
    rf.fit(X_train, y_train)                                                                                     # Dopasowanie modelu do danych treningowych
    probabilities = rf.predict_proba(X_test)                                                                     # Obliczenie prawdopodobienstw przynaleznosci do klas
    prob_positive_rf, squared_errors_rf = calculate_squared_errors(rf, X_train, y_train, X_test, y_test, errVar) # Obliczenie bledu sredniokwadratowego
    plot_probabilities(prob_positive_rf)                                                                         # Wykres prawdopodobienstw
    
    plt.plot(squared_errors_nb)                                 # Wykres bledu sredniokwadratowego
    plt.plot(squared_errors_svm)                                # Wykres bledu sredniokwadratowego
    plt.plot(squared_errors_rf)                                 # Wykres bledu sredniokwadratowego
    plt.title('Squared error by number of training samples')    # Tytul wykresu
    plt.xlabel('Number of training samples')                    # Etykieta osi x
    plt.ylabel('Squared error')                                 # Etykieta osi y
    plt.plot(squared_errors_nb, label='Naive Bayes')            # Legenda
    plt.plot(squared_errors_svm, label='SVM')                   # Legenda
    plt.plot(squared_errors_rf, label='Random Forest')          # Legenda
    plt.legend()                                                # Legenda
    plt.show()                                                  # Wyswietlenie wykresu

    r2_errors_nb = calculate_r2_errors(nb, X_train, y_train, X_test, y_test)    # Obliczenie bledu R^2
    r2_errors_svm = calculate_r2_errors(svm, X_train, y_train, X_test, y_test)  # Obliczenie bledu R^2
    r2_errors_rf = calculate_r2_errors(rf, X_train, y_train, X_test, y_test)    # Obliczenie bledu R^2

    plt.figure(figsize=(10, 6))                                 # Utworzenie wykresu
    plt.plot(r2_errors_nb, label='Naive Bayes')                 # Wykres bledu R^2
    plt.plot(r2_errors_svm, label='SVM')                        # Wykres bledu R^2
    plt.plot(r2_errors_rf, label='Random Forest')               # Wykres bledu R^2
    plt.xlabel('Number of training samples')                    # Etykieta osi x
    plt.ylabel('R^2 Error')                                     # Etykieta osi y
    plt.title('R^2 Error by Number of Training Samples')        # Tytul wykresu
    plt.legend()                                                # Legenda
    plt.show()                                                  # Wyswietlenie wykresu

    svm_mae = calculate_mae(svm, X_train, y_train, X_test, y_test)          # Obliczenie bledu sredniego bezwzglednego
    naive_bayes_mae = calculate_mae(nb, X_train, y_train, X_test, y_test)   # Obliczenie bledu sredniego bezwzglednego
    random_forest_mae = calculate_mae(rf, X_train, y_train, X_test, y_test) # Obliczenie bledu sredniego bezwzglednego

    plt.figure(figsize=(10, 5))                                                                     # Utworzenie wykresu
    plt.bar(['SVM', 'Naive Bayes', 'Random Forest'], [svm_mae, naive_bayes_mae, random_forest_mae]) # Wykres bledu sredniego bezwzglednego
    plt.title('Mean Absolute Error of Different Models')                                            # Tytul wykresu
    plt.ylabel('Mean Absolute Error')                                                               # Etykieta osi y
    plt.show()                                                                                    # Wyswietlenie wykresu
    """