import numpy as np
import matplotlib.pyplot as plt

# params
alpha = 2
beta = 1.1
gamma = 1
delta = 0.9
R_0 = 1
E_0 = 0.5

# equilibrium points
E_eq = alpha / beta
R_eq = gamma / delta

def Jacobian(state):
    R, E = state
    return np.array([
        [alpha - beta * E, -beta * R], 
        [delta * E, -gamma + delta * R]])

J_eq = Jacobian((R_eq, E_eq))
eigenvalues = np.linalg.eigvals(J_eq)
print(eigenvalues)

# stability analysis
if np.real(eigenvalues[0]) < 0:
    print("stable")
elif np.real(eigenvalues[0]) > 0:
    print("unstable")
else:
    print("inconclusive")

# plotting
R_vals = np.linspace(0, 3, 400)
E_vals = np.linspace(0, 3, 400)
R_grid, E_grid = np.meshgrid(R_vals, E_vals)

# derivatives at each point
dR_dt = alpha * R_grid - beta * R_grid * E_grid
dE_dt = -gamma * E_grid + delta * R_grid * E_grid

plt.figure(figsize=(8, 6))
plt.streamplot(R_grid, E_grid, dR_dt, dE_dt, color='lightblue', density=1.5)

# dR/dt = 0
plt.axhline(E_eq, color='red', linestyle='--', label=f'dR/dt=0 Nullcline (E={E_eq:.2f})')
# dE/dt = 0 
plt.axvline(R_eq, color='blue', linestyle='--', label=f'dE/dt=0 Nullcline (R={R_eq:.2f})')

# equilibrium point
plt.plot(R_eq, E_eq, 'ko', markersize=8, label='Equilibrium Point')

# initial state point
plt.plot(R_0, E_0, 'go', markersize=8, label=f'Initial State (R={R_0}, E={E_0})')

plt.title('Stream Plot for Metabolic Model')
plt.xlabel('Resource Concentration (R)')
plt.ylabel('Enzyme Concentration (E)')
plt.xlim(0, 3)
plt.ylim(0, 3)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()