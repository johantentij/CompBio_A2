import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

# set bigger text for matpotlib
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


###--- identifying sequences ---###

print("\nRunning Viterbi algorithm")
def viterbi(X,A,E):
    """Given a single sequence, with Transition and Emission probabilities,
    return the most probable state path, the corresponding P(X), and trellis."""

    allStates = A.keys()
    emittingStates = E.keys()
    L = len(X) + 1

    # Initialize
    V = {k:[0] * L for k in allStates}
    V['B'][0] = 1.

    # Middle columns
    for i,s in enumerate(X):
        for l in emittingStates:
            terms = [V[k][i] * A[k][l] for k in allStates]
            V[l][i+1] = max(terms) * E[l][s]

    best_state = max(emittingStates, key=lambda state: V[state][len(X)])
    P = V[best_state][len(X)]

    pi = best_state
    l = best_state
    for i in range(len(X)-1,0,-1): # iterating backwards through the sequence
        for k in emittingStates:
            if abs(V[k][i] * A[k][l] * E[l][X[i]] - V[l][i+1]) < 1e-10:
                pi = k + pi
                l = k
                break

    return(pi,P,V) # Return the state path, Viterbi probability, and Viterbi trellis

set_X = [
        "AGCGC",
        "AUUAU"
    ]
labels = ["seq1", "seq2"]

A = {
    'B': {'B': 0.0, 'E': 0.5, 'I': 0.5},
    'E': {'B': 0.0, 'E': 0.9, 'I': 0.1},
    'I': {'B': 0.0, 'E': 0.2, 'I': 0.8}
}
E = {
    'E': {'A': 0.25, 'U': 0.25, 'G': 0.25, 'C': 0.25},
    'I': {'A': 0.4, 'U': 0.4, 'G': 0.05, 'C': 0.15}
}

for j,X in enumerate(set_X):
    print("sequence:", X)
    Q, P, T = viterbi(X,A,E)
    label = labels[j]
    print("most likely path:", Q)
    print("probability:", P)
    print("\n")


###--- example hill functions ---###
print("\nMaking example hill function plots")
def hill_p(x, theta, n):
    return x ** n / (x ** n + theta ** n)

def hill_n(x, theta, n):
    return theta ** n / (x ** n + theta ** n)

n_vals = [1, 3, 10]

theta = .5

N = 100
P = np.linspace(0, 1, N)

fig, (ax1, ax2) = plt.subplots(1, 2)

for n in n_vals:
    H_p = hill_p(P, theta, n)
    H_n = hill_n(P, theta, n)

    ax1.plot(P, H_p, label=f"n={n}")
    ax2.plot(P, H_n, label=f"n={n}")

ax1.vlines(theta, 0, 1, color="grey", linestyles="dashed", label="$\\theta$")
ax2.vlines(theta, 0, 1, color="grey", linestyles="dashed", label="$\\theta$")

ax1.set_title("Activation")
ax1.set_xlabel("$P$ (M)")
ax1.set_ylabel("$H^+(P, \\theta, n)$")
ax1.legend()

ax2.set_title("Inhibition")
ax2.set_xlabel("$P$ (M)")
ax2.set_ylabel("$H^-(P, \\theta, n)$")
ax2.legend()

plt.show()


###--- ODE model ---###

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

def ODE_plots(isHealthy=False):
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

    return

print("Making ODE plots for healthy cell")
ODE_plots(isHealthy=True)
print("Making ODE plots for cancerous cell")
ODE_plots(isHealthy=False)

###--- ODE stability analysis ---###
print("\nBifurcation analysis")
N = 5000
N_grid = 100

theta_grid = np.linspace(.21, 2, N_grid)
n_grid = np.linspace(1, 6, N_grid)

stability = np.empty((N_grid, N_grid), dtype=np.int32)

for i, theta in enumerate(theta_grid):
    for j, n in enumerate(n_grid):

        rootFunc = lambda state : dState_dt(state, healthy=True, theta_A=theta, n_A=n)
        root, _, success, _ = fsolve(rootFunc, (.2, .2, .2, .2), full_output=1)

        if (success == 1):
            J_stable = Jacobian(root, theta_A=theta, n_A=n)
            eigens = np.linalg.eigvals(J_stable)

            stabilityType = np.sum(np.real(eigens) < 0)
            if (np.sum(np.abs(np.real(eigens)) < 1e-4) != 0):
                print("almost zero real eigenvalue encountered:", eigens)

            if (stabilityType == 4):
                stability[i, j] = 0

            elif (stabilityType == 0):
                stability[i, j] = 2

            else:
                stability[i, j] = 1
        
        else:
            stability[i, j] = -1

plt.pcolor(theta_grid, n_grid, stability.T)
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

plt.show()


###--- SDEVelo model ---###

print("\nRunning SDEVelo model")
dt = .01

# SDEVelo model params
a = np.array([1.0, .25])
b = np.array([.0005, .0005])
c = np.array([2.0, .5])
beta = np.array([2.35, 2.35])
gamma = np.array([1.0, 1.0])
sigma_1 = np.array([.05, .05])
sigma_2 = np.array([.05, .05])

m = np.array([2.35, 2.35])
gamma = np.array([1.0, 1.0])
k = np.array([1.0, 1.0])
delta = np.array([1.0, 1.0])

# Hill function params
theta_A = .21
theta_B = .21
n_A = 3
n_B = 3

def Euler_Maruyama_step(state, t):
    P, S, U = state

    P_A, P_B = P

    a_t = c / (1 + np.exp(b * (t - a)))

    beta_star = beta * np.array([
        hill_p(P_B, theta_B, n_B),
        hill_n(P_A, theta_A, n_A)
    ])

    dP = (k * S - delta * P) * dt

    dB_S = np.sqrt(dt) * np.random.normal(0, 1, 2)
    dB_U = np.sqrt(dt) * np.random.normal(0, 1, 2)

    dS = (beta_star * U - gamma * S) * dt + sigma_2 * dB_S
    dU = (a_t - beta_star * U) * dt + sigma_1 * dB_U
    
    dState = np.array([dP, dS, dU])

    return np.clip(state + dState, a_min=0, a_max=None)

N = 5000
t = np.arange(N) * dt

N_repeats = 100

stateHistEnsemble = np.empty((N_repeats, 3, 2, N))

for n in range(N_repeats):
    if (n % 10 == 0):
        print("SDEVelo run %d out of %d" % (n, N_repeats))

    stateHist = stateHistEnsemble[n]
    stateHist[0, :, 0] = np.array([.8, .8])     # P_A, P_B
    stateHist[1, :, 0] = np.array([.8, .8])     # s_A, s_B
    stateHist[2, :, 0] = np.array([.8, .8])     # u_A, u_B

    for i in range(1, N):
        stateHist[:, :, i] = Euler_Maruyama_step(stateHist[:, :, i - 1], t[i])

stateHistLower, stateHistUpper = np.percentile(stateHistEnsemble, (2.5, 97.5), axis=0)
stateHistMean = np.mean(stateHistEnsemble, axis=0)

P, S, U = stateHistMean
P_lower, S_lower, U_lower = stateHistLower
P_upper, S_upper, U_upper = stateHistUpper

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.fill_between(t, P_lower[0], P_upper[0], alpha=.5)
ax1.plot(t, P[0], label="$P_A$")
ax1.fill_between(t, S_lower[0], S_upper[0], alpha=.5)
ax1.plot(t, S[0], label="$S_A$")
ax1.fill_between(t, U_lower[0], U_upper[0], alpha=.5)
ax1.plot(t, U[0], label="$U_A$")
ax1.set_title("A")
ax1.set_xlabel("time (s)")
ax1.set_ylabel("Concentration (M)")
ax1.legend()

ax2.fill_between(t, P_lower[1], P_upper[1], alpha=.5)
ax2.plot(t, P[1], label="$P_A$")
ax2.fill_between(t, S_lower[1], S_upper[1], alpha=.5)
ax2.plot(t, S[1], label="$S_A$")
ax2.fill_between(t, U_lower[1], U_upper[1], alpha=.5)
ax2.plot(t, U[1], label="$U_A$")
ax2.set_title("B")
ax2.set_xlabel("time (s)")
ax2.set_ylabel("Concentration (M)")
ax2.legend()
plt.show()

plt.plot(P[0], P[1])
plt.scatter(P[0, 0], P[1, 0], label="Start", c="green", zorder=2)
plt.scatter(P[0, -1], P[1, -1], label="End", c="red", zorder=2)
plt.title("Mean trajectory")
plt.xlabel("$P_A$ (M)")
plt.ylabel("$P_B$ (M)")
plt.legend()
plt.show()


###--- BONUS: metabolic model ---###

print("\nMaking plots for bonus part")
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
print("Eigenvalues of the equilibrium point:", eigenvalues)

print("Nature of the equilibrium point:")
# stability analysis
if np.real(eigenvalues[0]) < 0:
    print("stable")
elif np.real(eigenvalues[0]) > 0:
    print("unstable")
else:
    print("inconclusive")

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