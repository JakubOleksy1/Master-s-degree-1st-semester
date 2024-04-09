import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("C:/Users/jakub/Visual Studio Code/AUM")

# Sigmoidalna funkcja aktywacji
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Pochodna sigmoidalnej funkcji aktywacji
def sigmoid_derivative(x):
    return x * (1 - x)

# Funkcja uczenia sieci neuronowej
def train(X, y, hidden_layer_size, learning_rate, epochs):
    input_layer_size = X.shape[1]
    output_layer_size = y.shape[1]

    # Inicjalizacja wag i biasów
    W1 = np.random.randn(input_layer_size, hidden_layer_size)
    b1 = np.zeros((1, hidden_layer_size))
    W2 = np.random.randn(hidden_layer_size, output_layer_size)
    b2 = np.zeros((1, output_layer_size))

    errors_mse_output = []
    error_mse_hidden = []
    errors_classification = []
    weights_history = {'W1': [], 'b1': [], 'W2': [], 'b2': []}

    for _ in range(epochs):
        # Propagacja do przodu
        hidden_layer_input = np.dot(X, W1) + b1
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, W2) + b2
        output = sigmoid(output_layer_input)


        # Obliczenie błędów
        error = y - output
        errors_mse_output.append(np.mean(error**2))
        errors_classification.append(np.mean(np.abs(np.round(output) - y)))
        hidden_error = error * sigmoid_derivative(output_layer_input)
        hidden_layer_error = np.dot(hidden_error, W2.T)
        error_mse_hidden.append(np.mean(hidden_layer_error**2))

        # Propagacja wsteczna
        d_output = error * sigmoid_derivative(output)
        d_hidden_layer = np.dot(d_output, W2.T) * sigmoid_derivative(hidden_layer_output)

        # Aktualizacja wag i biasów
        W2 += np.dot(hidden_layer_output.T, d_output) * learning_rate
        b2 += np.sum(d_output) * learning_rate
        W1 += np.dot(X.T, d_hidden_layer) * learning_rate
        b1 += np.sum(d_hidden_layer) * learning_rate

        weights_history['W1'].append(W1.copy())
        weights_history['W2'].append(W2.copy())
        weights_history['b1'].append(b1.copy())
        weights_history['b2'].append(b2.copy()) 

    print("\nTesting XOR gate:")
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    output = sigmoid(output_layer_input)
    print(f"Predicted Output: {output}")

    return W1, b1, W2, b2, errors_mse_output, errors_classification, weights_history, error_mse_hidden

# Zbiór danych XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Parametry uczenia
learning_rate = 0.2
epochs = 1000
hidden_layer_size = 3

# Uczenie sieci neuronowej
W1, b1, W2, b2, errors_mse, errors_classification, weights_history, error_mse_hidden = train(X, y, hidden_layer_size, learning_rate, epochs)

# Wykres błędu MSE w obu warstwach
plt.figure()

plt.subplot(1,3,1)
plt.plot(errors_mse)
plt.title('Błąd MSE wyjściowy')
plt.xlabel('Epoki')
plt.ylabel('Błąd')
plt.grid()
plt.subplot(1,3,2)
plt.plot(error_mse_hidden)
plt.title('Błąd MSE w warstwie ukrytej')
plt.xlabel('Epoki')
plt.ylabel('Błąd')
plt.grid()
# Wykres błędu klasyfikacji
plt.subplot(1,3,3)
plt.plot(errors_classification)
plt.title('Błąd klasyfikacji')
plt.xlabel('Epoki')
plt.ylabel('Błąd')
plt.grid()
plt.draw()

# Wykresy wag w obu warstwach
plt.figure()
for layer in ['W1', 'b1', 'W2', 'b2']:
    weights = np.array(weights_history[layer])
    for i in range(weights.shape[1]):
        plt.plot(weights[:, i], label=f'{layer}{i+1}')

plt.title('Wagi w warstwie ukrytej')
plt.xlabel('Neurony')
plt.ylabel('Wejścia')
plt.legend()
plt.grid()
plt.show()


