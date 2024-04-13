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

def simulated_annealing(tasks, initial_temp, cooling_rate, iteration_per_temp, epochs):
    current_temp = initial_temp
    current_sequence = list(range(len(tasks)))
    random.shuffle(current_sequence)
    current_makespan = calculate_makespan(current_sequence, tasks)
    
    temperatures = []  # Lista do przechowywania temperatur
    makespans = []  # Lista do przechowywania wartości makespan

    for __ in range(epochs): 
        for _ in range(iteration_per_temp):
            new_sequence = current_sequence.copy()
            i, j = random.sample(range(len(tasks)), 2)
            new_sequence[i], new_sequence[j] = new_sequence[j], new_sequence[i]

            new_makespan = calculate_makespan(new_sequence, tasks)
            if new_makespan < current_makespan or random.random() < math.exp((new_makespan - current_makespan) / current_temp):
                current_sequence, current_makespan = new_sequence, new_makespan
        
        temperatures.append(current_temp)
        makespans.append(current_makespan)
        
        current_temp *= cooling_rate

    return current_sequence, current_makespan, temperatures, makespans

# Parametry algorytmu
initial_temp = 10000
cooling_rate = 0.95
iteration_per_temp = 150
epochs = 150 #<200

# Uruchomienie algorytmu
filepath = "rpq_100.txt"  # Zmień na aktualną ścieżkę do pliku
tasks = read_rpq_data(filepath)
best_sequence, best_makespan, temperatures, makespans = simulated_annealing(tasks, initial_temp, cooling_rate, iteration_per_temp, epochs)

if set(best_sequence) == set(range(len(tasks))): # Sprawdzenie czy wszystkie zadania zostaly uzyte
    print("Wszystkie zadania zostaly uzyte")
else:
    print("Niektorych zadan brakuje")

print("Najlepsza sekwencja:", best_sequence)
print("Makespan:", best_makespan)

# Rysowanie wykresu
plt.plot(temperatures, makespans)
#plt.gca().invert_xaxis()  # Odwrócenie osi X, aby temperatura malejąca była przedstawiona od lewej do prawej
plt.xlabel('Temperatura')
plt.ylabel('Makespan')
plt.title('Wykres zmian makespan w zależności od temperatury')
plt.show()

#sprawko opis teoretyczny funckje opisac co robia i wykresy 
#str encyklopedia algorytmow