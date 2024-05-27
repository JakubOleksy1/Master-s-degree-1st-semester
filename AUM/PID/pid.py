class PIDController:
    def __init__(self, Kp, Ki, Kd):
        # Inicjalizacja wspolczynnikow regulatora PID po raz pierwszy potem update w osobnej funkcji
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # Inicjalizacja bledow regulacji
        self.prev_error = 0.0
        self.prev_prev_error = 0.0

        # inicjalizacja poprzedniego sterowania
        self.prev_u = 0.0

    def update_coefficients(self, Kp, Ki, Kd):
        # Update stalych Kp Ki Kd
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def update_controller(self, U_c, Y) -> float:
        # Obliczenie aktulanego bledu regulacji
        current_error = U_c - Y
        # Obliczenie sterowania na podstawie bledu regulacji
        u = self.prev_u + self.Kp * (current_error - self.prev_error) + self.Ki * current_error + self.Kd * (current_error - 2 * self.prev_error + self.prev_prev_error)

        # Update bledow regulacji
        self.prev_prev_error = self.prev_error
        self.prev_error = current_error
        self.prev_u = u

        return u

#///////////////////////////////////////////////////////////////////////////////////////////////////

import matplotlib.pyplot as plt
import random

if __name__ == "__main__":
    # Inicjalizacja regulatora PID
    pid = PIDController(Kp=0.2,Ki=0.9,Kd=0.2)

    # Inicjalizacja temperatury zadanej i aktualnej (zamienic na odpowiedni prrzyklad)
    desired_temp = 20.0
    actual_temp = 0.0

    # Inicjalizacja listy temperatur i sterowan
    temps = [actual_temp]
    outputs = []
    iterations = 100 

    # Petla symulujaca dzialanie regulatora PID
    for _ in range(iterations):
        # Obliczenie sterowania
        u = pid.update_controller(desired_temp, actual_temp)
        outputs.append(u)

        # Symulacja zmiany temperatury
        actual_temp += u + random.uniform(-1, 1)
        temps.append(actual_temp)
        print("Actual temperature: ", actual_temp)

        #pid.update_coefficients(new_Kp, new_Ki, new_Kd, y) # tylko jezeli tuner cos zwroci

    # Rysowanie wykresow 
    plt.plot([desired_temp]*(iterations+1), 'r--', label='Desired temperature')
    plt.plot(temps, label='Actual temperature')
    plt.plot(outputs, label='Control output')
    plt.legend()
    plt.show()

    """
    neural_network_tuner(u)
    musi zwracac kp ki kd po polaczeniu z tuneremtak by "petla" wykonala sie raz
    po czym doszlo do aktualizacji nowych kp ki kd (chyba wsadz w petle update kp ki kd )
    !!! UZYJ pid.update_coefficients(new_Kp, new_Ki, new_Kd, y) !!!
    """
