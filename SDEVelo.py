import numpy as np
import matplotlib.pyplot as plt

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

def Euler_Maruyama_step(state, t, healthy=True):
    P, S, U = state

    P_A, P_B = P

    a_t = a * c / (1 + np.exp(b * (t - a)))
    if healthy:
        # healthy function -> P_A inhibits U_B
        a_star = a_t * np.array([
            hill_p(P_B, theta_B, n_B),
            hill_n(P_A, theta_A, n_A)
        ])

    else:
        # cancerous function -> inhibition is removed
        a_star = a * c / (1 + np.exp(b * (t - a))) * np.array([
            hill_p(P_B, theta_B, n_B),
            1.0
        ])

    beta_star = beta * np.array([
        hill_p(P_B, theta_B, n_B),
        hill_n(P_A, theta_A, n_A)
    ])

    dP = (k * S - delta * P) * dt

    dB_S = np.sqrt(dt) * np.random.normal(0, 1, 2)
    dB_U = np.sqrt(dt) * np.random.normal(0, 1, 2)

    dS = (beta_star * U - gamma * S) * dt + sigma_2 * dB_S
    dU = (a_star - beta_star * U) * dt + sigma_1 * dB_U
    
    dState = np.array([dP, dS, dU])

    return state + dState

N = 1000
t = np.arange(N) * dt

N_repeats = 100

stateHistEnsemble = np.empty((N_repeats, 3, 2, N))

for n in range(N_repeats):
    stateHist = stateHistEnsemble[n]
    stateHist[0, :, 0] = np.array([.8, .8])     # P_A, P_B
    stateHist[1, :, 0] = np.array([.8, .8])     # s_A, s_B
    stateHist[2, :, 0] = np.array([.8, .8])     # u_A, u_B

    for i in range(1, N):
        stateHist[:, :, i] = Euler_Maruyama_step(stateHist[:, :, i - 1], t[i], healthy=False)

stateHistLower, stateHistUpper = np.percentile(stateHistEnsemble, (2.5, 97.5), axis=0)
stateHistMean = np.mean(stateHistEnsemble, axis=0)

P, S, U = stateHistMean
P_lower, S_lower, U_lower = stateHistLower
P_upper, S_upper, U_upper = stateHistUpper

plt.fill_between(t, U_lower[0], U_upper[0], alpha=.5)
plt.fill_between(t, U_lower[1], U_upper[1], alpha=.5)
plt.plot(t, U[0])
plt.plot(t, U[1])
plt.show()