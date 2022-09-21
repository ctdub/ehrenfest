import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Hamiltonian
h = np.array([[1, 0.1], [0.1, 1]], dtype=complex)
# initial conditions
clast = np.array([1, 0], dtype=complex)
# time step
dt = 0.1
# total time
totalTime = 100
# array of time values
timeSteps = np.arange(0, totalTime, dt)
# coefficients at time step n
cn = np.zeros((timeSteps.size, clast.size), dtype=complex)
for n in range(timeSteps.size):
    k1 = -1j * h.dot(clast)
    k2 = -1j * h.dot(clast + dt * k1/2)
    k3 = -1j * h.dot(clast + dt * k2/2)
    k4 = -1j * h.dot(clast + dt * k3)
    clast = clast + (1/6) * (k1 + 2*k2 + 2*k3 + k4) * dt
    cn[n] = clast

plt.plot(timeSteps, np.real(np.multiply(np.conj(cn[:, 0]), cn[:, 0])))
plt.plot(timeSteps, np.real(np.multiply(np.conj(cn[:, 1]), cn[:, 1])))
plt.show()
