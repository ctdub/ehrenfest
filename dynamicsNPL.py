import numpy as np
from numba import jit
import matplotlib.pyplot as plt

def build_hamilton(qtrj):
    diagEn = np.zeros(Nm + 1)
    diagEn[0] = omega
    dsp = np.sqrt(2 * np.multiply(s, 1/pFreq))
    reorgE = np.multiply(0.5 * pFreq**2, (qtrj - dsp)**2)
    diagEn[1:] = eNPL + reorgE
    h = np.diag(diagEn)
    h[0, 1:] = g
    h[1:, 0] = g
    return h

def QMrk4(h, inCon):
    clast = inCon
    k1 = -1j * h.dot(clast)
    k2 = -1j * h.dot(clast + dt * k1 / 2)
    k3 = -1j * h.dot(clast + dt * k2 / 2)
    k4 = -1j * h.dot(clast + dt * k3)
    cn = clast + 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4) * dt
    return cn

def init_classical():
    q_zpe = np.zeros((Nm, Ntraj))
    p_zpe = np.zeros((Nm, Ntraj))
    for n in range(Nm):
        q_zpe[n, :] = np.random.normal(loc=0.0, scale=sigmaQ[n], size=Ntraj)
        p_zpe[n, :] = np.random.normal(loc=0.0, scale=sigmaP[n], size=Ntraj)
    return p_zpe, q_zpe

# time step
dt = 0.5
# total time
totalTime = 50
# total number of time steps
steps = int(np.ceil(totalTime/dt))

# Number of NPL
nNPL = 1
# NPL energy
eNPL = 1
# cavity frequency
omega = 1
# cavity-exciton coupling strength
g = 0.1

# Nubmer of trajectories
Ntraj = 1
# temperature
T = 290
# Boltzman constant in atomic units
Kb = 3.168 * 10 ** -6
# Number of phonon modes per NPL
Nm = 1
# Phonon mode frequencies
dW = 0.2
# pFreq = np.linspace(0, 0.01, 1)
pFreq = np.array([0.05])
# Creates the Wigner distribution standard deviations
sigmaQ = np.zeros(Nm)
sigmaP = np.zeros(Nm)
# initial classical distribution shifts
shift = np.zeros(Nm)
# Huang-Rhys factor
s = np.full(Nm, 0.5)
# NPL in their excited state. inES = 1 if the NPL is excited at time zero.
for n in range(pFreq.size):
    sigmaQ[n] = np.sqrt(1 / (2 * pFreq[n] * np.tanh(pFreq / (Kb * T))))
    sigmaP[n] = pFreq[n] * sigmaQ[n]

# define the initial conditions
cint = np.zeros(Nm + 1)
cint[1] = 1

initQ, initP = init_classical()
ham = build_hamilton(initQ)

coef = np.zeros((steps + 1, Nm + 1), dtype=complex)
coef[0] = cint
for n in range(steps):
    coef[n + 1, :] = QMrk4(ham, coef[n])

plt.plot(np.arange(0, totalTime + dt, dt), np.real(np.multiply(np.conj(coef[:, 0]), coef[:, 0])))
plt.plot(np.arange(0, totalTime + dt, dt), np.real(np.multiply(np.conj(coef[:, 1]), coef[:, 1])))
