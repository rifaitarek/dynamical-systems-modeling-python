import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def logistic_growth(N, t, r, K):
    """
    Logistic Growth Model.

    Arguments:
    N : float
        Current population size
    t : float
        Current time point (not used explicitly in this function, but required for odeint)
    r : float
        Intrinsic growth rate
    K : float
        Carrying capacity of the environment

    Returns:
    float: Rate of population change (dN/dt)
    """
    return r * N * (1 - N / K)

# Parameters
r = 0.1   # Intrinsic growth rate
K = 1000  # Carrying capacity
N0 = 10   # Initial population size
t = np.linspace(0, 100, 1000)

# Solve ODE
sol = odeint(logistic_growth, N0, t, args=(r, K))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t, sol)
plt.title('Logistic Growth Model')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.grid(True)
plt.show()
