import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def sir_model(y, t, N, beta, gamma):
    """
    SIR (Susceptible-Infected-Recovered) Model.

    Arguments:
    y : list of three elements [S, I, R]
        S - number of susceptible individuals
        I - number of infected individuals
        R - number of recovered individuals
    t : float
        Current time point (not used explicitly in this function, but required for odeint)
    N : int
        Total population size
    beta : float
        Transmission rate
    gamma : float
        Recovery rate

    Returns:
    list of three elements [dS/dt, dI/dt, dR/dt]
        Rates of change for S, I, and R
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# Parameters
N = 10000     # Total population
I0, R0 = 100, 0  # Initial number of infected and recovered individuals
S0 = N - I0 - R0  # Initial number of susceptible
beta = 0.3    # Transmission rate
gamma = 0.1   # Recovery rate
t = np.linspace(0, 100, 1000)

# Solve ODE
sol = odeint(sir_model, [S0, I0, R0], t, args=(N, beta, gamma))
S, I, R = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.title('SIR Model')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.legend()
plt.grid(True)
plt.show()
