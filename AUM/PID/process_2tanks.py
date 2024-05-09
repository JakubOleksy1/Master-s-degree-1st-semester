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

    def process_run(self, V_in, delta_T) -> float:
        """
        Run one iteration of model simulation.

        Args:
            V_in: Input valve opening in percents.
            delta_T: simulation time step [s]

        Returns:
            h2: liquid level in second tank.

        """
        Q1 = (self.c * V_in) / 100
        Q2 = self.K2 * math.sqrt(self.h2)
        Q12 = self.K1 * math.sqrt(self.h1 - self.h2)
        h1_new = ((Q1 -Q2) / self.A1) * delta_T + self.h1
        h2_new = ((Q12 - Q2) / self.A2) * delta_T + self.h2
        self.h1 = h1_new
        self.h2 = h2_new
        self.prev_Vin = V_in

        return self.h2

    def get_dY_dU(self, delta_T, dU) -> float:
        """
        Calculates dY/dU derivative for system.

        Args:
            delta_T: simulation time step [s].
            dU: Input value diffrence

        Returns:
            h2: dY/dU derivative at given moment

        """
        # first iteration for moment n
        Q1 = self.prev_Vin + (self.c * dU) / 100
        Q2 = self.K2 * math.sqrt(self.h2)
        Q12 = self.K1 * math.sqrt(self.h1 - self.h2)
        h1_1 = ((Q1 -Q2) / self.A1) * delta_T + self.h1
        h2_1 = ((Q12 - Q2) / self.A2) * delta_T + self.h2

        # second iteration for moment n+1
        Q2 = self.K2 * math.sqrt(h2_1)
        Q12 = self.K1 * math.sqrt(h1_1 - h2_1)
        h2_2 = ((Q12 - Q2) / self.A2) * delta_T + h2_1
        
        #return dY(n+1)/dU(n)
        return (h2_2 - h2_1) / dU
    

tanks_sys = TwoTankProcess()
time = list(range(5000))
system_out = list()
dYdU = list()

for y in time:
    system_out.append(tanks_sys.process_run(100, 1))
    dYdU.append(tanks_sys.get_dY_dU(1, 0.01))

plt.plot(time, system_out)
plt.plot(time, dYdU)
plt.show()