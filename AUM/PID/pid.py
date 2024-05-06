class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        total = 0
        for (weight, input) in zip(self.weights, inputs):
            total += weight * input
        return self.sigmoid(total + self.bias)

    def sigmoid(self, x):
        return 1 / (1 + pow(2.71828, -x))

class PIDController:
    def __init__(self):
        # Initialize the PID parameters
        self.Kp = 0.2
        self.Ki = 0.9
        self.Kd = 0.2

        # Initialize the error values
        self.prev_error = 0.0
        self.prev_prev_error = 0.0

        # Initialize the control output
        self.prev_u = 0.0

    def update_controller(self, current_error):
        # Compute the PID output
        u = self.prev_u + self.Kp * (current_error - self.prev_error) + self.Ki * current_error + self.Kd * (current_error - 2 * self.prev_error + self.prev_prev_error)

        # Update the previous errors and control
        self.prev_prev_error = self.prev_error
        self.prev_error = current_error
        self.prev_u = u

        return u
    
#//////////////////////////////////////////////////////////////////////////////

import matplotlib.pyplot as plt
import random

if __name__ == "__main__":
    # Initialize a PID controller
    pid = PIDController()

    # Initialize the desired and actual temperatures
    desired_temp = 20.0
    actual_temp = 0.0

    # Initialize a list to store the temperatures and outputs
    temps = [actual_temp]
    outputs = []

    # Run the simulation for 100 time steps
    for _ in range(100):
        # Compute the error
        error = desired_temp - actual_temp

        # Update the controller and get the new output
        u = pid.update_controller(error)
        outputs.append(u)

        # Update the actual temperature
        actual_temp += u + random.uniform(-1, 1)
        temps.append(actual_temp)

    # Plot the desired and actual temperatures and the control output
    plt.plot([desired_temp]*101, 'r--', label='Desired temperature')
    plt.plot(temps, label='Actual temperature')
    plt.plot(outputs, label='Control output')
    plt.legend()
    plt.show()