import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def seir_model(y, t, N, beta, sigma, gamma):
    """
    SEIR (Susceptible-Exposed-Infected-Recovered) Model.

    Arguments:
    y : list of four elements [S, E, I, R]
        S - number of susceptible individuals
        E - number of exposed individuals
        I - number of infected individuals
        R - number of recovered individuals
    t : float
        Current time point (not used explicitly in this function, but required for odeint)
    N : int
        Total population size
    beta : float
        Transmission rate
    sigma : float
        Rate at which exposed individuals become infected
    gamma : float
        Recovery rate

    Returns:
    list of four elements [dS/dt, dE/dt, dI/dt, dR/dt]
        Rates of change for S, E, I, and R
    """
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]

# Parameters
N = 10000        # Total population
I0, R0 = 100, 0  # Initial number of infected and recovered individuals
E0 = 1000        # Initial number of exposed individuals
S0 = N - I0 - R0 - E0  # Initial number of susceptible
beta = 0.3       # Transmission rate
sigma = 0.2      # Rate of exposed becoming infected
gamma = 0.1      # Recovery rate
t = np.linspace(0, 100, 1000)

# Solve ODE
sol = odeint(seir_model, [S0, E0, I0, R0], t, args=(N, beta, sigma, gamma))
S, E, I, R = sol.T

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.title('SEIR Model')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.legend()
plt.grid(True)
plt.show()
