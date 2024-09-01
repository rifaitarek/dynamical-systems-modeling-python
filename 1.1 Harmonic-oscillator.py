import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def harmonic_oscillator(y, t, k):
    """
    Arguments:
    y : list of two elements [x, v]
        x - represents the expression level of Gene A
        v - represents the expression level of Gene B
    t : float
        Current time point (not used explicitly in this function, but required for odeint)
    k : float
        Spring constant, represents the strength of regulation between genes
    """
    x, v = y
    dydt = [v, -k*x]
    return dydt

# Parameters
k = 1.0  # spring constant (regulation strength)
y0 = [1, 0]  # initial conditions: [Gene A expression, Gene B expression]
t = np.linspace(0, 20, 1000)

# Solve ODE
sol = odeint(harmonic_oscillator, y0, t, args=(k,))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, sol[:, 0], label='Gene A Expression')
plt.plot(t, sol[:, 1], label='Gene B Expression')
plt.title('Gene Expression Oscillation (Harmonic Oscillator Model)')
plt.xlabel('Time')
plt.ylabel('Expression Level')
plt.legend()
plt.grid(True)
plt.show()
