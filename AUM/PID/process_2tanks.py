import math
import matplotlib.pyplot as plt

'''
This class models two tank process described in Pirabakaran-Becerra
paper chapter 4 (fig. 4)
'''

class TwoTankProcess:
    def __init__(self, A1=289, A2=144, K1=30, K2=30, c=280):
        """
        Initialize model with process constants. If not called
        ten default values are given.

        Args:
            A1: First tank base area.
            A2: second tank base area.
            K1: Flow constant for Q12 valve. Corresponds to physical properties of valve, tank, and liquid.
            K2: Flow constant for Q2 valve. Corresponds to physical properties of valve, tank, and liquid.
            c: Flow constant for Q1 (input) flow. Corresponds to max input flow.

        """
        self.A1 = A1
        self.A2 = A2
        self.K1 = K1
        self.K2 = K2
        self.c = c
        self.h1 = 0
        self.h2 = 0
        self.prev_Vin = 0

    def process_run(self, V_in, delta_T):
        """
        Run one iteration of model simulation.

        Args:
            V_in: Input valve opening in percents.
            delta_T: simulation time step [s]

        Returns:
            h2: liquid level in second tank.

        """
        Q1 = (self.c * V_in) / 100
        Q2 = self.K2 * math.sqrt(self.h2) if self.h2 > 0 else 0
        Q12 = self.K1 * math.sqrt(max(self.h1 - self.h2, 0))
        
        h1_new = self.h1 + ((Q1 - Q12) / self.A1) * delta_T
        h2_new = self.h2 + ((Q12 - Q2) / self.A2) * delta_T

        if h1_new < 0: h1_new = 0
        if h2_new < 0: h2_new = 0
        
        self.h1 = h1_new
        self.h2 = h2_new
        self.prev_Vin = V_in
        
        return self.h2

    def get_dY_dU(self, delta_T, dU):
        """
        Calculates dY/dU derivative for system.

        Args:
            delta_T: simulation time step [s].
            dU: Input value diffrence

        Returns:
            h2: dY/dU derivative at given moment

        """
        # Current state
        V_in = self.prev_Vin
        Q1 = (self.c * V_in) / 100
        Q2 = self.K2 * math.sqrt(self.h2) if self.h2 > 0 else 0
        Q12 = self.K1 * math.sqrt(max(self.h1 - self.h2, 0))
        h1_1 = self.h1 + ((Q1 - Q12) / self.A1) * delta_T
        h2_1 = self.h2 + ((Q12 - Q2) / self.A2) * delta_T
        
        # Perturbed state
        V_in += dU
        Q1 = (self.c * V_in) / 100
        Q2 = self.K2 * math.sqrt(h2_1) if h2_1 > 0 else 0
        Q12 = self.K1 * math.sqrt(max(h1_1 - h2_1, 0))
        h1_2 = h1_1 + ((Q1 - Q12) / self.A1) * delta_T
        h2_2 = h2_1 + ((Q12 - Q2) / self.A2) * delta_T
        
        return (h2_2 - h2_1) / dU

if __name__ == "__main__":
    # Simulation setup
    tanks_sys = TwoTankProcess()
    time = list(range(30000))
    system_out = []
    dYdU = []

    timeStep = 0.1
    input = 100
    for t in time:
        system_out.append(tanks_sys.process_run(input, timeStep))
        dYdU.append(tanks_sys.get_dY_dU(timeStep, (0.003*timeStep)/(input/100)))

    # Plotting the results
    plt.plot(time, system_out, label="h2(t)")
    plt.plot(time, dYdU, label="dY/dU")
    plt.xlabel('Time [s]')
    plt.ylabel('Height / dY/dU')
    plt.legend()
    plt.title('Two Tank Process Simulation')
    plt.show()
