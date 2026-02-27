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

