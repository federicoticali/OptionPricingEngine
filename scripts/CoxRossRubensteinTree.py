"""
Cox-Ross-Rubinstein tree convergence to Black-Scholes.
 
The binomial tree is DETERMINISTIC (no confidence interval): its error is the
deterministic discretisation error, which lives on a price-vs-steps plot rather
than on an error-bar comparison.
 
Two panels:
  left  - the tree price vs the number of steps for small n, coloured by parity,
          showing the characteristic odd-even oscillation around Black-Scholes.
  right - |tree price - BS| vs n on log-log, showing the ~1/n decay, and how
          averaging two consecutive trees (n, n+1) cancels the oscillation.
 
Run:      python crr_convergence.py
Requires: numpy, matplotlib, and the classes in OptionPricing.py.
"""
 
import numpy as np
import matplotlib.pyplot as plt

from OptionPricing import (
    MarketData, VanillaOption, BlackScholesPricer, CoxRossRubinsteinPricer,
)
 
Market = MarketData(S = 49, r = 0.05, sigma = 0.2)
Option = VanillaOption(K = 50, T = 20 / 52, option_type = "call")
BS = BlackScholesPricer(Market, Option).price()
 
 
def tree_price(n: int) -> float:
    t = CoxRossRubinsteinPricer(Market, Option)
    t.n_steps = int(n)
    return t.CoxRossRubinsteinTree()
 

steps_small = np.arange(2, 81)
px_small = np.array([tree_price(n) for n in steps_small])
even = steps_small % 2 == 0
 
steps_big = np.unique(np.logspace(0.6, 3.0, 45).astype(int))     # ~4 .. 1000
px_big = np.array([tree_price(n) for n in steps_big])
err_raw = np.abs(px_big - BS)
px_avg = np.array([0.5 * (tree_price(n) + tree_price(n + 1)) for n in steps_big])
err_avg = np.abs(px_avg - BS)
 
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
 
ax1.axhline(BS, ls = "--", lw = 1.4, color = "#222222", zorder = 1,
            label = f"Black\u2013Scholes = {BS:.4f}")
ax1.plot(steps_small, px_small, "-", color = "#d9d9d9", lw = 0.9, zorder = 1)
ax1.scatter(steps_small[even], px_small[even], s = 26, color = "#2ca02c",
            zorder = 3, label = "even number of steps")
ax1.scatter(steps_small[~even], px_small[~even], s = 26, color = "#d62728",
            zorder = 3, label = "odd number of steps")
ax1.set_xlabel("number of steps  n")
ax1.set_ylabel("CRR tree price")
ax1.set_title("Odd\u2013even oscillation around Black\u2013Scholes", fontweight = "bold")
ax1.legend(frameon = False, fontsize = 9)
ax1.grid(alpha = 0.3)

ax2.loglog(steps_big, err_raw, "-o", ms = 3.5, color = "#1f77b4",
           label = "|CRR \u2212 BS|  (raw)")
ax2.loglog(steps_big, err_avg, "-s", ms=3.5, color = "#ff7f0e",
           label = "averaged tree (n, n+1)")
ref = err_raw[0] * steps_big[0] / steps_big          # ~ 1/n guide through first point
ax2.loglog(steps_big, ref, ":", color = "#888888", lw = 1.3, label = "~1/n reference")
ax2.set_xlabel("number of steps  n")
ax2.set_ylabel("|price \u2212 Black\u2013Scholes|")
ax2.set_title("Convergence rate: error vs steps", fontweight = "bold")
ax2.legend(frameon = False, fontsize = 9)
ax2.grid(alpha = 0.3, which = "both")
 
fig.suptitle("Cox\u2013Ross\u2013Rubinstein tree: convergence to Black\u2013Scholes",
             fontsize = 14, fontweight = "bold", y = 0.99)
fig.text(0.5, 0.005, "deterministic method: no confidence interval \u2014 the error is "
                     "pure discretisation error, oscillating as O(1/n)",
         ha = "center", fontsize = 9, color = "#555555")
fig.tight_layout(rect = (0, 0.03, 1, 0.96))
fig.savefig("../figures/crr_convergence.png", dpi = 150)
print("saved crr_convergence.png")
# plt.show()
