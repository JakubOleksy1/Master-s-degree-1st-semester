import os

os.chdir("C:/Users/jakub/Visual Studio Code/IIMS")

import random
import matplotlib.pyplot as plt

# Prior 
prior = [
    random.random() 
    for _ in range(6)           #randomowe prawdopodobienstwa (>0)
    ]

prior = [
    p/sum(prior)                # Normalizuj sume prawdopodobienstwa do 1 
    for p in prior
    ]

print("Prior probabilities:", prior, "Prior sum", sum(prior))

probabilityPlot = [
    []
    for _ in range(6)       # 6 histogramow
    ]

y = 10000
iterations = []

for i in range(1, y + 1):
    x = random.randint(1, 6) - 1      # Losowanie liczby od 1 do 6 usun 1 bo indeksujemy od 0

    # Update prior 
    posterior = [
        prior[j] + (1 if j == x else 0) / i  # Dodaj 1 jesli wylosowana liczba jest rowna j
        for j in range(6)                    # i to numer iteracji
        ]

    # Posterior probabilities
    probability = [
        posterior[j] / sum(posterior) 
        for j in range(6)
        ]

    for j in range(6):
        probabilityPlot[j].append(probability[j])

    iterations.append(i)
    prior = posterior

print("Probabilities:", probability)

for j in range(6):
    plt.subplot(2, 3, j+1)
    plt.plot(iterations, probabilityPlot[j])
    plt.title(f'Probability of {j+1}')
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.grid()

plt.tight_layout()
plt.show()
