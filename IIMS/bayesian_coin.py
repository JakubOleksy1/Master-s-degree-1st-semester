import os

os.chdir("C:/Users/jakub/Visual Studio Code/IIMS")

import random
import matplotlib.pyplot as plt

# Prior 
prior_heads = random.random()
prior_tails = 1 - prior_heads
print("Prior probabilities - Heads:", prior_heads, "Tails:", prior_tails)

headsPlot = []
tailsPlot = []

y = 10000
iterations = []

for i in range(1, y + 1):
    x = random.randint(0, 1)
    if x == 1:
        heads = 1
        tails = 0 
    else:  
        heads = 0
        tails = 1

    # Update prior 
    posterior_heads = prior_heads + heads / i             
    posterior_tails = prior_tails + tails / i

    # Posterior probabilities
    probability_heads = posterior_heads / (posterior_heads + posterior_tails)
    probability_tails = 1 - probability_heads

    headsPlot.append(probability_heads)
    tailsPlot.append(probability_tails)
    iterations.append(i)

    prior_heads = posterior_heads
    prior_tails = posterior_tails


print("Probability of heads:", probability_heads)
print("Probability of tails:", probability_tails)
plt.subplot(1, 2, 1)
plt.plot(iterations, headsPlot)
plt.title('Probability of heads')
plt.xlabel('Iteration')
plt.ylabel('Probability')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(iterations, tailsPlot)
plt.title('Probability of tails')
plt.xlabel('Iteration')
plt.ylabel('Probability')
plt.grid()

plt.tight_layout()
plt.show()

#zrobic dla kostki