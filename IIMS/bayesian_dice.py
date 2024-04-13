import os

os.chdir("C:/Users/jakub/Visual Studio Code/IIMS")

import random
import matplotlib.pyplot as plt

# Prior 
prior = [
    random.random() 
    for _ in range(6)
    ]

prior = [
    p/sum(prior)                # Normalizuj sume do 1 
    for p in prior
    ]

print("Prior probabilities:", prior, "Prior sum", sum(prior))

probabilityPlot = [
    []
    for _ in range(6)
    ]

y = 10000
iterations = []

for i in range(1, y + 1):
    x = random.randint(1, 6) - 1  # Subtract 1 to use as index

    # Update prior 
    posterior = [
        prior[j] + (1 if j == x else 0) / i 
        for j in range(6)
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
