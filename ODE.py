import numpy as np
import matplotlib.pyplot as plt

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

def dState_dt(state, healthy):
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

def RK4_step(state, healthy=True):
    k1 = dState_dt(state, healthy)
    k2 = dState_dt(state + .5 * k1 * dt, healthy)
    k3 = dState_dt(state + .5 * k2 * dt, healthy)
    k4 = dState_dt(state + k3 * dt, healthy)

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