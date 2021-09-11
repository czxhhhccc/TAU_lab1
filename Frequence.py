import control
import control.matlab as matlab
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import math

# num = np.array([2,0])
# den = np.array([3,1])
# N = signal.TransferFunction(num, den)
# print(N)
# w, mag, phase = signal.bode(N)
# plt.plot(w, 10**(mag/20))
# plt.show()
h=matlab.tf([2,0],[3,1])
mag, phase, omega=control.bode(h, plot=False)
plt.plot(omega, phase)
print(type(omega))
phase1=[]
for i in omega:
    phase1.append(math.atan(1/(3*i))-2*np.pi)
    pass
plt.plot(omega, phase1)
plt.show()
# hh=matlab.tf([2,0],[3,1])
# matlab.nyquist(hh)
# plt.show()