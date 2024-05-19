class ReferenceModel:
    def __init__(self, natural_freq=0.03, damping=0.7) -> None:
        self.omega = natural_freq
        self.ksi = damping
        self.prev_y = 0
        self.prev_prev_y = 0

    def system_output(self, input, time_step):
        # Obliczanie aktualnej wartości wyjścia systemu
        T = time_step
        omega_n = self.omega
        ksi = self.ksi

        # Współczynniki równania różnicowego
        a0 = 1 + 2 * ksi * omega_n * T + (omega_n * T) ** 2
        a1 = 2 * (1 + ksi * omega_n * T)
        a2 = -1
        b0 = (omega_n * T)**2

        # Obliczanie aktualnej wartości wyjścia systemu
        y = (b0 * input + a1 * self.prev_y + a2 * self.prev_prev_y) / a0
        
        # Aktualizacja poprzednich wartości wyjścia
        self.prev_prev_y = self.prev_y
        self.prev_y = y
        
        return y

# Tworzenie instancji modelu referencyjnego
model = ReferenceModel() # natural_freq=1, damping=0.7

# Symulacja odpowiedzi systemu na przykładowe dane wejściowe
time_step = 0.1
input_signal = [1] * 5000  # Stałe wejście o wartości 1 przez 5000 kroków czasowych
output_signal = []

for input in input_signal:
    output = model.system_output(input, time_step)
    output_signal.append(output)

# Wyświetlanie wyników
import matplotlib.pyplot as plt

plt.plot(output_signal)
plt.xlabel('Kroki czasowe')
plt.ylabel('Odpowiedź systemu')
plt.title('Odpowiedź systemu na stałe wejście')
plt.grid(True)
plt.show()
