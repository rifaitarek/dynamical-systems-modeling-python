import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint

def lorenz_system(y, t, sigma, rho, beta):
    """
    Arguments:
    y : list of three elements [x, y, z]
        x, y, z - represent the expression levels of three interacting genes
    t : float
        Current time point (not used explicitly in this function, but required for odeint)
    sigma : float
        Represents the rate of gene expression activation
    rho : float
        Represents the strength of genetic feedback loops
    beta : float
        Represents the rate of gene expression inhibition
    """
    x, y, z = y
    dydt = [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
    return dydt

# Parameters
sigma = 10
rho = 28
beta = 8/3
y0 = [1, 1, 1]  # initial conditions: [Gene X expression, Gene Y expression, Gene Z expression]
t = np.linspace(0, 50, 5000)

# Solve ODE
sol = odeint(lorenz_system, y0, t, args=(sigma, rho, beta))

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol[:, 0], sol[:, 1], sol[:, 2])
ax.set_title('Gene Regulatory Network (Lorenz System)')
ax.set_xlabel('Gene X Expression')
ax.set_ylabel('Gene Y Expression')
ax.set_zlabel('Gene Z Expression')
plt.show()
