import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


dt = .01

# ODE model params
m_A = 2.35
m_B = 2.35
gamma_A = 1
gamma_B = 1
k_PA = 1
k_PB = 1
theta_A = .21
theta_B = .21
n_A = 3
n_B = 3
delta_PA = 1
delta_PB = 1

def hill_p(x, theta, n):
    return x ** n / (x ** n + theta ** n)

def hill_n(x, theta, n):
    return theta ** n / (x ** n + theta ** n)

def dHill_p_dx(x, theta, n):
    return n * theta ** n * x ** (n - 1) * (x ** n + theta ** n) ** (-2)

def dHill_n_dx(x, theta, n):
    return n * theta ** (n - 1) * x ** n * (x ** n + theta ** n) ** (-2)

def Jacobian(state, theta_A=theta_A, n_A=n_A):
    P_A, r_A, P_B, r_B = state

    return np.array([
        [-delta_PA, k_PA, 0, 0],
        [0, -gamma_A, m_A * dHill_p_dx(P_B, theta_B, n_B), 0],
        [0, 0, -delta_PB, k_PB],
        [m_B * dHill_n_dx(P_A, theta_A, n_A), 0, 0, -gamma_B]
    ])

def dState_dt(state, healthy, theta_A=theta_A, n_A=n_A):
    P_A, r_A, P_B, r_B = state

    if healthy:
        # healthy function -> P_A inhibits r_B
        return np.array([
            k_PA * r_A - delta_PA * P_A,
            m_A * hill_p(P_B, theta_B, n_B) - gamma_A * r_A,
            k_PB * r_B - delta_PB * P_B,
            m_B * hill_n(P_A, theta_A, n_A) - gamma_B * r_B
        ])
    
    else:
        # cancerous function -> inhibition is removed
        return np.array([
            k_PA * r_A - delta_PA * P_A,
            m_A * hill_p(P_B, theta_B, n_B) - gamma_A * r_A,
            k_PB * r_B - delta_PB * P_B,
            m_B - gamma_B * r_B
        ])

def RK4_step(state, healthy=True, theta_A=theta_A, n_A=n_A):
    k1 = dState_dt(state, healthy, theta_A=theta_A, n_A=n_A)
    k2 = dState_dt(state + .5 * k1 * dt, healthy, theta_A=theta_A, n_A=n_A)
    k3 = dState_dt(state + .5 * k2 * dt, healthy, theta_A=theta_A, n_A=n_A)
    k4 = dState_dt(state + k3 * dt, healthy, theta_A=theta_A, n_A=n_A)

    return state + (k1 + 2 * (k2 + k3) + k4) * dt / 6

isHealthy = False

# find equilibrium point
rootFunc = lambda state : dState_dt(state, healthy=isHealthy)
root, _, success, _ = fsolve(rootFunc, (.2, .2, .2, .2), full_output=1)

P_A_eq, r_A_eq, P_B_eq, r_B_eq = root

N = 5000
t = np.arange(N) * dt

stateHist = np.empty((N, 4), dtype=np.float64)
stateHist[0, :] = .8

for i in range(1, N):
    stateHist[i, :] = RK4_step(stateHist[i - 1, :], healthy=isHealthy)

P_A, r_A, P_B, r_B = stateHist.T

# time series
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.plot(t, P_A, label="$P_A$")
ax1.plot(t, r_A, label="$r_A$")
ax1.set_xlabel("time (s)")
ax1.set_ylabel("Concentration (M)")
ax1.legend()

ax2.plot(t, P_B, label="$P_B$")
ax2.plot(t, r_B, label="$r_B$")
ax2.set_xlabel("time (s)")
ax2.set_ylabel("Concentration (M)")
ax2.legend()

# phase plot

plt.figure()
plt.plot(P_A, P_B)
plt.scatter(P_A[0], P_B[0], label="Start", c="green", zorder=2)
plt.scatter(P_A[-1], P_B[-1], label="End", c="red", zorder=2)
plt.scatter(P_A_eq, P_B_eq, label="Equilibrium point", c="black", zorder=2)
plt.xlabel("$P_A$ (M)")
plt.ylabel("$P_B$ (M)")
plt.legend()


plt.tight_layout()
plt.show()


# instead of just disabling inhibition, gradually remove it by raising theta 
# a Hopf bifurcation occurs, where the limit cycle dies and the system settles
# into a stable point

# stability analysis
N_grid = 100

theta_grid = np.linspace(.21, 2, N_grid)
n_grid = np.linspace(1, 6, N_grid)

stability = np.empty((N_grid, N_grid), dtype=np.int32)

ding = np.empty(N_grid)

for i, theta in enumerate(theta_grid):
    for j, n in enumerate(n_grid):

        rootFunc = lambda state : dState_dt(state, healthy=True, theta_A=theta, n_A=n)
        root, _, success, _ = fsolve(rootFunc, (.2, .2, .2, .2), full_output=1)

        if (success == 1):
            J_stable = Jacobian(root, theta_A=theta, n_A=3)
            eigens = np.linalg.eigvals(J_stable)

            stabilityType = np.sum(np.real(eigens) < 0)
            if (np.sum(np.abs(np.real(eigens)) < 1e-4) != 0):
                print(eigens)

            if (stabilityType == 4):
                stability[i, j] = 0

            elif (stabilityType == 0):
                stability[i, j] = 2

            else:
                stability[i, j] = 1

            if (j == 0):
                ding[i] = np.max(np.real(eigens))
        
        else:
            stability[i, j] = -1

im = plt.pcolor(theta_grid, n_grid, stability.T)
plt.colorbar(im)
plt.text(.75, 3.5, "Saddle-point", backgroundcolor="white")
plt.text(1.5, 2, "Stable", backgroundcolor="white")
plt.xlabel("$\\theta_A$ (M)")
plt.ylabel("$n_A$")
plt.show()

# Hopf bifurcation
theta_grid = [.21, .5, .7, 1.4]

fig, axes = plt.subplots(1, 4)

for i in range(4):
    rootFunc = lambda state : dState_dt(state, healthy=True, theta_A=theta_grid[i])
    root, _, success, _ = fsolve(rootFunc, (.2, .2, .2, .2), full_output=1)

    P_A_eq, r_A_eq, P_B_eq, r_B_eq = root

    ax = axes[i]

    stateHist = np.empty((N, 4), dtype=np.float64)
    stateHist[0, :] = .8

    for j in range(1, N):
        stateHist[j, :] = RK4_step(stateHist[j - 1, :], healthy=True, theta_A=theta_grid[i])

    P_A, r_A, P_B, r_B = stateHist.T

    ax.plot(P_A, P_B)
    ax.set_title(f"$\\theta_A = ${theta_grid[i]} M")
    ax.scatter(P_A[0], P_B[0], label="Start", c="green", zorder=2)
    ax.scatter(P_A[-1], P_B[-1], label="End", c="red", zorder=2)
    ax.scatter(P_A_eq, P_B_eq, label="Eq. point", c="black", zorder=2)
    ax.set_xlabel("$P_A$ (M)")
    if (i == 0):
        ax.set_ylabel("$P_B$ (M)")
    ax.legend()

# plt.tight_layout()
plt.show()