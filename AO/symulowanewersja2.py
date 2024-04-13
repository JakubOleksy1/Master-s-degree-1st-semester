import random
import math
import matplotlib.pyplot as plt
import os

os.chdir("C:/Users/jakub/Visual Studio Code/AO")

def read_rpq_data(filepath):
    tasks = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                r, p, q = map(int, parts)
                tasks.append((r, p, q))
    return tasks

def calculate_makespan(sequence, tasks):
    time = 0
    end_time = 0
    for task in sequence:
        r, p, q = tasks[task]
        time = max(time, r) + p 
        end_time = max(end_time, time + q) 
    return end_time

def generujsasiada(sequence, T, N):
    i = random.randint(0, N-1)
    j = random.randint(0, N-1)
    while i == j:
        j = random.randint(0, N-1)
    sequence[i], sequence[j] = sequence[j], sequence[i]
    return sequence

def symulowane_wyzarzanie_rpq(tasks, temperatura_poczatkowa, wspolczynnik_chlodzenia, liczba_prob, max_epoki):
    N = len(tasks)
    # Wylosuj rozwiązanie początkowe (sekwencję zadań)
    X = list(range(N))
    random.shuffle(X)
    T = temperatura_poczatkowa
    najlepsze_rozwiazanie = X[:]
    najlepszy_makespan = calculate_makespan(najlepsze_rozwiazanie, tasks)

    temperatures = []  # Lista do przechowywania temperatur
    makespans = []  # Lista do przechowywania wartości makespan


    for epoka in range(max_epoki):
        for _ in range(liczba_prob):
            X_prim = generujsasiada(X[:], T, N)
            delta_f = calculate_makespan(X_prim, tasks) - calculate_makespan(X, tasks)

            if delta_f < 0:
                X = X_prim
                if calculate_makespan(X, tasks) < najlepszy_makespan:
                    najlepsze_rozwiazanie = X[:]
                    najlepszy_makespan = calculate_makespan(X, tasks)
            else:
                p = math.exp(-delta_f / T)
                if random.random() < p:
                    X = X_prim

        temperatures.append(T)
        makespans.append(calculate_makespan(X, tasks))

        T *= wspolczynnik_chlodzenia

    return najlepsze_rozwiazanie, najlepszy_makespan, temperatures, makespans

#Parametry algorytmu
temperatura_poczatkowa = 1000
wspolczynnik_chlodzenia = 0.95
liczba_prob = 1
max_epoki = 10000 

#Uruchomienie algorytmu
filepath = "rpq_500.txt"  # Zmień na aktualną ścieżkę do pliku
tasks = read_rpq_data(filepath)

najlepsze_rozwiazanie, najlepszy_makespan, temperatures, makespans = symulowane_wyzarzanie_rpq(tasks, temperatura_poczatkowa, wspolczynnik_chlodzenia, liczba_prob, max_epoki)
print("Najlepsze rozwiązanie:", najlepsze_rozwiazanie)
print("Najlepszy makespan:", najlepszy_makespan)

if set(najlepsze_rozwiazanie) == set(range(len(tasks))): # Sprawdzenie czy wszystkie zadania zostaly uzyte
    print("Wszystkie zadania zostaly uzyte")
else:
    print("Niektorych zadan brakuje")

print("Najlepsza sekwencja:", najlepsze_rozwiazanie)
print("Makespan:", najlepszy_makespan)

# Rysowanie wykresu
epoki = range(max_epoki)
plt.plot(epoki, makespans)
plt.xlabel('Epoki')
plt.ylabel('Makespan')
plt.title('Wykres zmian makespan w zależności od temperatury')
plt.show()
#sprawko opis teoretyczny funckje opisac co robia i wykresy 
#str encyklopedia algorytmow