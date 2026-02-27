import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


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

def dState_dt(state):
    R, E = state

    return np.array([
        alpha * R - beta * R * E,
        -gamma * E + delta * R * E
    ])

def RK4_step(state):
    k1 = dState_dt(state)
    k2 = dState_dt(state + .5 * k1 * dt)
    k3 = dState_dt(state + .5 * k2 * dt)
    k4 = dState_dt(state + k3 * dt)

    return state + (k1 + 2 * (k2 + k3) + k4) * dt / 6

dt = .01
N = 10000
t = np.arange(N) * dt

stateHist = np.empty((2, N), dtype=np.float64)
stateHist[0, 0] = 1
stateHist[1, 0] = .5

for i in range(1, N):
    stateHist[:, i] = RK4_step(stateHist[:, i - 1])

J_eq = Jacobian((R_eq, E_eq))
eigenvalues = np.linalg.eigvals(J_eq)
print(eigenvalues)

# stability analysis
if np.real(eigenvalues[0]) < 0:
    print("stable")
elif np.real(eigenvalues[0]) > 0:
    print("unstable")
else:
    print("saddle point")

# plotting
R_vals = np.linspace(0, 4, 400)
E_vals = np.linspace(0, 5, 400)
R_grid, E_grid = np.meshgrid(R_vals, E_vals)

# derivatives at each point
dR_dt = alpha * R_grid - beta * R_grid * E_grid
dE_dt = -gamma * E_grid + delta * R_grid * E_grid

plt.figure(figsize=(8, 6))
plt.streamplot(R_grid, E_grid, dR_dt, dE_dt, color='lightblue', density=1.5)
plt.plot(stateHist[0, :], stateHist[1, :], label="trajectory")

# dR/dt = 0
plt.axhline(E_eq, color='red', linestyle='--', label=f'dR/dt=0 Nullcline (E={E_eq:.2f})')
# dE/dt = 0 
plt.axvline(R_eq, color='blue', linestyle='--', label=f'dE/dt=0 Nullcline (R={R_eq:.2f})')

# equilibrium point
plt.plot(R_eq, E_eq, 'ko', markersize=8, label='Equilibrium Point')

# initial state point
plt.plot(R_0, E_0, 'go', markersize=8, label=f'Initial State (R={R_0}, E={E_0})')

plt.title('Stream Plot for Metabolic Model')
plt.xlabel('Metabolite concentration (M)')
plt.ylabel('Enzyme concentration (M)')
plt.xlim(0, 4)
plt.ylim(0, 5)
plt.legend(loc='upper right')
plt.grid(True)
plt.show()