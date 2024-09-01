import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def lotka_volterra(y, t, alpha, beta, delta, gamma):
    """
    Lotka-Volterra predator-prey model.

    Arguments:
    y : list of two elements [x, y]
        x - prey population
        y - predator population
    t : float
        Current time point (not used explicitly in this function, but required for odeint)
    alpha : float
        Prey growth rate in the absence of predators
    beta : float
        Rate at which predators destroy prey
    delta : float
        Predator death rate in the absence of prey
    gamma : float
        Rate at which predators increase by consuming prey

    Returns:
    list of two elements [dx/dt, dy/dt]
        Rates of change for prey and predator populations
    """
    x, y = y
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Parameters
alpha = 1.0  # Prey growth rate
beta = 0.1   # Predation rate
delta = 0.075  # Predator growth rate
gamma = 1.5  # Predator death rate
y0 = [10, 5]  # Initial populations: [prey, predators]
t = np.linspace(0, 100, 1000)

# Solve ODE
sol = odeint(lotka_volterra, y0, t, args=(alpha, beta, delta, gamma))

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.plot(t, sol[:, 0], label='Prey')
plt.plot(t, sol[:, 1], label='Predator')
plt.title('Population Dynamics')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid(True)

plt.subplot(122)
plt.plot(sol[:, 0], sol[:, 1])
plt.title('Phase Space')
plt.xlabel('Prey Population')
plt.ylabel('Predator Population')
plt.grid(True)

plt.tight_layout()
plt.show()