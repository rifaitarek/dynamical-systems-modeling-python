import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def damped_oscillator(y, t, k, c):
    """
    Arguments:
    y : list of two elements [x, v]
        x - represents the protein level
        v - represents the rate of change of protein level
    t : float
        Current time point (not used explicitly in this function, but required for odeint)
    k : float
        Spring constant, represents the strength of regulation
    c : float
        Damping coefficient, represents the protein degradation rate
    """
    x, v = y
    dydt = [v, -k*x - c*v]
    return dydt

# Parameters
k = 1.0  # spring constant (regulation strength)
c = 0.1  # damping coefficient (degradation rate)
y0 = [1, 0]  # initial conditions: [protein level, rate of change]
t = np.linspace(0, 50, 1000)

# Solve ODE
sol = odeint(damped_oscillator, y0, t, args=(k, c))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, sol[:, 0], label='Protein Level')
plt.title('Protein Level Oscillation with Degradation (Damped Oscillator Model)')
plt.xlabel('Time')
plt.ylabel('Protein Level')
plt.legend()
plt.grid(True)
plt.show()
