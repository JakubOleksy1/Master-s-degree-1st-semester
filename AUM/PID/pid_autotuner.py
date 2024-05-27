from pid import PIDController
from process_2tanks import TwoTankProcess
from autotuner_NN import NNPIDAutotuner
from reference_model import ReferenceModel
import matplotlib.pyplot as plt
import numpy as np
import copy

def ref_signal(t):
    # Generate the square wave signal with amplitude 0.6 and period 500 s
    usq = 0.6 * (1 + np.sign(np.sin(2 * np.pi * t / 500)))
    # Generate the sinusoidal signal with amplitude 0.6 and frequency corresponding to a period of 50 s
    sinusoidal = 0.6 * np.sin(0.02 * t)
    # Generate the reference signal uc(t) by adding the square wave and sinusoidal signals, and adding a constant offset of 5
    return usq + sinusoidal + 5

def squareWave(t):
    # Generate the square wave signal with amplitude 2 and period 2000 s
    return 2 * (1 + np.sign(np.sin(2 * np.pi * t / 2000))) + 5

'''
def predictNextModelErr(refModel: ReferenceModel, current_y, dy_du):
    model_cp = copy.deepcopy(refModel)
    y_next = model_cp.system_output()
'''

#simulation of a PID control for 2 tanks system

STEPS_AMT = 30000
TIME_STEP = 1
Y_0 = 0 #initial system state
H_D = 5 # desired height of liquid in 2nd tank (that is desired output of the system)
UC_0 = 0 #initial system input
AUTOTUNING_ON = True

# init objects
tanksSys = TwoTankProcess() # default data from the article
pid = PIDController(Kp=13,Ki=0.01,Kd=10.02) #random settings,  data from the article Kp=4.95,Ki=0.01,Kd=10.02 (Kp=5,Ki=1,Kd=10)
tunerNN = NNPIDAutotuner(2*10**(-7), 10**(-7), 0.00001) # data from paper
refModel = ReferenceModel()

# data samples arrays
time_n = np.arange(0, STEPS_AMT, TIME_STEP) # amount of a time steps for the simulation
system_y = [] # system output
refSystem_ym = [] # reference model output
controlVal_u = []
systemErr_e = []
referenceSignal_uc = []
systemJacobian_dYdu = []
arr_Kp = []
arr_Ki = []
arr_Kd = []

#simulation
E = [0, 0, 0] # model errors at n, n-1, n-2
u_p = 0 # previous u
y_p = 0 # previous y
uc_p = 0 # previous u_c
# initial conditions
y = Y_0
u = 0
u_c = UC_0
for t in time_n:
    # predict K coeffcients updates
    if AUTOTUNING_ON: delta_Kp, delta_Ki, delta_Kd = tunerNN.predict([u, u_p, y, y_p, u_c, uc_p])
    else: delta_Kp, delta_Ki, delta_Kd = (0, 0, 0)
    pid.update_coefficients(pid.Kp + delta_Kp, pid.Ki + delta_Ki, pid.Kd + delta_Kd)
    uc_p = u_c # update u_c
    u_c = squareWave(t) # reference signal, tank nr 2 liquid height
    u_p = u # update u_p
    u = pid.update_controller(u_c, y)
    if u > 100: u = 100 # control value is 0% - 100% and corresponds to opening of the valve
    if u < 0: u = 0 # same as ^^^
    y_p = y # update y_p
    y = tanksSys.process_run(u, TIME_STEP)
    if u == 0: u_temp = 0.00001
    else: u_temp = u
    dY_du = tanksSys.get_dY_dU(TIME_STEP, 0.1)
    y_m = refModel.system_output(u_c, TIME_STEP)
    e = y - y_m # model error
    if AUTOTUNING_ON: tunerNN.train(dY_du, E, e)
    # update errors arr
    E[2] = E[1]
    E[1] = E[0]
    E[0] = e
    # for plotting
    system_y.append(y)
    controlVal_u.append(u)
    refSystem_ym.append(y_m)
    systemErr_e.append(e)
    referenceSignal_uc.append(u_c)
    systemJacobian_dYdu.append(dY_du)
    arr_Kp.append(pid.Kp)
    arr_Ki.append(pid.Ki)
    arr_Kd.append(pid.Kd)

plt.figure()
plt.plot(time_n, systemJacobian_dYdu, label='system Jacobian - dY_du')
plt.plot(time_n, system_y, label='liquid level - y/h_2', color='blue')
plt.legend()
plt.grid(True)

# Plotting all collected signals on one graph
plt.figure(figsize=(10, 6))

plt.plot(time_n, system_y, label='system output - y/h2', color='blue')
plt.plot(time_n, controlVal_u, label='control value - u/v', color='red')
plt.plot(time_n, refSystem_ym, label='reference model output - y_m', color='green')
plt.plot(time_n, systemErr_e, label='model error - e', color='orange',)
plt.plot(time_n, referenceSignal_uc, label='reference signal - u_c', color='purple')
#plt.plot(time_n, systemJacobian_dYdu, label='system Jacobian - dY_du', color='brown', linestyle='--')

plt.xlabel('time')
plt.ylabel('values')
plt.legend()
plt.grid(True)
plt.title('Signals Over Time')

plt.tight_layout()

# Plotting all collected signals seperately
plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.plot(time_n, system_y, label='liquid level - y/h_2')
plt.xlabel('time')
plt.ylabel('liquid level')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(time_n, controlVal_u, label='control value - u/v')
plt.xlabel('time')
plt.ylabel('control value')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(time_n, refSystem_ym, label='reference model output - y_m')
plt.xlabel('time')
plt.ylabel('reference model output')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(time_n, systemErr_e, label='model error - e')
plt.xlabel('time')
plt.ylabel('model error')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(time_n, referenceSignal_uc, label='reference signal - u_c')
plt.xlabel('time')
plt.ylabel('reference signal')
plt.legend()
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(time_n, systemJacobian_dYdu, label='system Jacobian - dY_du')
plt.xlabel('time')
plt.ylabel('system Jacobian')
plt.legend()
plt.grid(True)

plt.tight_layout()


plt.figure()
#Kp 
plt.subplot(3, 1, 1)
plt.plot(time_n, arr_Kp, label='Kp')
plt.xlabel('time')
plt.ylabel('Kp')
plt.legend()
plt.grid(True)
# Ki
plt.subplot(3, 1, 2)
plt.plot(time_n, arr_Ki, label='Ki')
plt.xlabel('time')
plt.ylabel('Kd')
plt.legend()
plt.grid(True)
# Kd
plt.subplot(3, 1, 3)
plt.plot(time_n, arr_Ki, label='Kd')
plt.xlabel('time')
plt.ylabel('Kd')
plt.legend()
plt.grid(True)


plt.show()