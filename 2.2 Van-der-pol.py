import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def van_der_pol(y, t, mu):
    """
    Arguments:
    y : list of two elements [x, v]
        x - represents the cell cycle phase
        v - represents the rate of change of the cell cycle phase
    t : float
        Current time point (not used explicitly in this function, but required for odeint)
    mu : float
        Nonlinearity parameter, represents the strength of nonlinear damping in the cell cycle
    """
    x, v = y
    dydt = [v, mu * (1 - x**2) * v - x]
    return dydt

# Parameters
mu = 1.0  # nonlinearity parameter
y0 = [0.5, 0.5]  # initial conditions: [cell cycle phase, rate of change]
t = np.linspace(0, 50, 1000)

# Solve ODE
sol = odeint(van_der_pol, y0, t, args=(mu,))

# Plot
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(t, sol[:, 0])
plt.title('Cell Cycle Progression')
plt.xlabel('Time')
plt.ylabel('Cell Cycle Phase')

plt.subplot(122)
plt.plot(sol[:, 0], sol[:, 1])
plt.title('Cell Cycle Phase Space')
plt.xlabel('Cell Cycle Phase')
plt.ylabel('Rate of Change')
plt.grid(True)

plt.tight_layout()
plt.show()
