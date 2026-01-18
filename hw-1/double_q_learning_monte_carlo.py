import numpy as np
import matplotlib.pyplot as plt

Nmax = 1_000_000
R = 100  # number of Monte Carlo trials (tune)

sum_running_max = np.zeros(Nmax, dtype=np.float64)

rng = np.random.default_rng(0)
for r in range(R):
    eps = rng.standard_normal(Nmax)          # N(0,1)
    runmax = np.maximum.accumulate(eps)      # runmax[n-1] = max_{i<=n} eps[i]
    sum_running_max += runmax

fhat = 1.0 + (sum_running_max / R)
ghat = np.ones(Nmax)  # exactly g(n)=1, but included for plotting

n = np.arange(1, Nmax + 1)

plt.figure()
plt.semilogx(n, fhat, label="Monte Carlo  f(n)=E[max X]")
plt.semilogx(n, ghat, label="g(n)=max E[X]=1")
plt.xlabel("n (log scale)")
plt.ylabel("value")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
