from pid import PIDController
from process_2tanks import TwoTankProcess
import matplotlib.pyplot as plt
import numpy as np

def ref_signal(t):
    # Generate the square wave signal with amplitude 0.6 and period 500 s
    usq = 0.6 * (1 + np.sign(np.sin(2 * np.pi * t / 500)))
    # Generate the sinusoidal signal with amplitude 0.6 and frequency corresponding to a period of 50 s
    sinusoidal = 0.6 * np.sin(0.02 * t)
    # Generate the reference signal uc(t) by adding the square wave and sinusoidal signals, and adding a constant offset of 5
    return usq + sinusoidal + 5

#simulation of a PID control for 2 tanks system (without autotuning)

STEPS_AMT = 6000

tanks_sys = TwoTankProcess() # default data from the article
pid = PIDController(Kp=4.95,Ki=0.1,Kd=10.02) # data from the article

time_n = list(range(STEPS_AMT)) # amount of a time steps for the simulation
system_y = list() # system response
controlVal_u = list()
y = 0 # current system  output
h_d = 20 # desired height of liquid in 2nd tank (that is desired output of the system)

#system outputs
for t in time_n:
    #h_d = ref_signal(t)
    if(t == 3000): h_d = 50 # step on the input
    u = pid.update_controller(h_d, y)
    if u > 100: u = 100 # control value is 0% - 100% and corresponds to opening of the valve
    if u < 0: u = 0 # same as prev
    y = tanks_sys.process_run(u, 1)
    system_y.append(y)
    controlVal_u.append(u)

#t = np.arange(0, STEPS_AMT, 1)  # Time from 0 to 999 seconds
#uc = ref_signal(t)

plt.plot(time_n, system_y, label='liquid level - h_2')
#plt.plot(time_n, controlVal_u, label='control value - v')
h_d_toPlot = [h_d] * STEPS_AMT
plt.plot(time_n, h_d_toPlot, label='setpoint - h_d')
#plt.plot(t, uc, label='Reference Signal (uc)')
plt.xlabel('time')
plt.ylabel('values')
plt.legend()
plt.grid(True)
plt.show()