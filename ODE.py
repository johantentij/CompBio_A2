import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

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

def Jacobian(state):
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

N = 10000
t = np.arange(N) * dt

stateHist = np.empty((N, 4), dtype=np.float64)
stateHist[0, :] = .8

for i in range(1, N):
    stateHist[i, :] = RK4_step(stateHist[i - 1, :], healthy=True)

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

# phase plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1.plot(r_A, P_A)
ax1.set_xlabel("$r_A$")
ax1.set_ylabel("$P_A$")

ax2.plot(P_A, P_B)
ax2.set_xlabel("$P_A$")
ax2.set_ylabel("$P_B$")

ax3.plot(r_B, P_B)
ax3.set_xlabel("$r_B$")
ax3.set_ylabel("$P_B$")

plt.tight_layout()
plt.show()


# instead of just disabling inhibition, gradually remove it by raising theta 
# a Hopf bifurcation occurs, where the limit cycle dies and the system settles
# into a stable point

# stability analysis
N_grid = 100

theta_grid = np.linspace(.21, 2, N_grid)

maxReEigen = np.empty(N_grid, dtype=np.float64)
for i, theta in enumerate(theta_grid):
    rootFunc = lambda state : dState_dt(state, healthy=True, theta_A=theta, n_A=3)
    root, _, success, _ = fsolve(rootFunc, (.5, .5, .5, .5), fprime=Jacobian, full_output=1)

    if (success == 1):
        J_stable = Jacobian(root)
        eigens = np.linalg.eigvals(J_stable)

        maxReEigen[i] = np.max(np.real(eigens))
    
    else:
        maxReEigen[i] = np.nan

theta_crit = theta_grid[np.argmin(np.abs(maxReEigen))]

plt.plot(theta_grid, maxReEigen)
plt.vlines(theta_crit, np.min(maxReEigen), np.max(maxReEigen), 
           linestyles="dashed", color="grey",
           label=r"$\theta_\mathrm{crit} = $" + f"{theta_crit:.3f}")
plt.xlabel("$\\theta_A$")
plt.ylabel("max $\\lambda$")
plt.legend()
plt.show()

# Hopf bifurcation
theta_grid = [.21, .5, .7]

fig, axes = plt.subplots(1, 3)

for i in range(3):
    ax = axes[i]

    stateHist = np.empty((N, 4), dtype=np.float64)
    stateHist[0, :] = .8

    for j in range(1, N):
        stateHist[j, :] = RK4_step(stateHist[j - 1, :], healthy=True, theta_A=theta_grid[i])

    P_A, r_A, P_B, r_B = stateHist.T

    ax.plot(r_A, P_A)
    ax.set_title(f"$\\theta_A = ${theta_grid[i]}")
    ax.set_xlabel("$r_A$")
    ax.set_ylabel("$P_A$")

plt.tight_layout()
plt.show()