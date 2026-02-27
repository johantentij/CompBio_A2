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

def hill_p(x, theta, n):
    return x ** n / (x ** n + theta ** n)

def hill_n(x, theta, n):
    return theta ** n / (x ** n + theta ** n)

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