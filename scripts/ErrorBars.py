"""
Monte Carlo estimator comparison for the European Option Pricing Engine.
 
Three panels (one per discretisation scheme: exact / euler / milstein). In each
panel the x-axis lists the Monte Carlo methods (standard + antithetic variates +
control variates + importance sampling variance reduction techniques), and each 
point shows the price estimate with its 95% confidence interval (+/-1.96*SE). 
The dashed line is the analytic Black-Scholes price: every method should land on 
it, and better variance reduction = shorter error bar.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
 
from OptionPricing import (
    MarketData, VanillaOption, BlackScholesPricer, MonteCarloPricer,
)

SEED = 12345
N_PATHS = 200_000
N_STEPS = 500                      
SCHEMES = ["exact", "euler", "milstein"]
METHODS = [                   
    (None,          "Standard",   "#9aa0a6"),
    ("antithetic",  "Antithetic", "#1f77b4"),
    ("control",     "Control",    "#2ca02c"),
    ("importance",  "Importance", "#9467bd"),
]
 
Market = MarketData(S = 49, r = 0.05, sigma = 0.2)
Option = VanillaOption(K = 50, T = 20 / 52, option_type = "call")
BS = BlackScholesPricer(Market, Option).price()
 
np.random.seed(SEED)
results = {s: {} for s in SCHEMES}
for scheme in SCHEMES:
    mc = MonteCarloPricer(Market, Option)
    mc.scheme, mc.n_paths, mc.n_steps = scheme, N_PATHS, N_STEPS
    for vr, label, _ in METHODS:
        results[scheme][label] = mc.PricingMC(variance_reduction = vr)
 
all_lo, all_hi = [], []
for scheme in SCHEMES:
    for label, (p, se) in results[scheme].items():
        all_lo.append(p - 1.96 * se); all_hi.append(p + 1.96 * se)
lo, hi = min(all_lo + [BS]), max(all_hi + [BS])
pad = 0.18 * (hi - lo)
ylim = (lo - pad, hi + pad)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize = (15, 5.6), sharey=True)
 
x = np.arange(len(METHODS))
for ax, scheme in zip(axes, SCHEMES):
    ax.axhline(BS, ls = "--", lw = 1.4, color = "#222222", zorder=1)
    for i, (vr, label, col) in enumerate(METHODS):
        price, se = results[scheme][label]
        hw = 1.96 * se
        ax.errorbar(i, price, yerr = hw, fmt = "o", ms = 10, color = col,
                    capsize = 7, capthick = 1.8, elinewidth = 2.2,
                    mec = "white", mew = 1.3, zorder = 3)
        ax.annotate(f"\u00b1{hw:.4f}", (i, price), textcoords = "offset points",
                    xytext = (14, 0), va = "center", fontsize = 8, color = col)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in METHODS], fontsize = 10)
    ax.set_xlim(-0.5, len(METHODS) - 0.5 + 0.6)
    ax.set_title(scheme.capitalize(), fontsize = 13, fontweight = "bold", pad = 8)
    ax.grid(axis = "y", alpha = 0.3)
    ax.tick_params(axis = "x", length = 0)
 
axes[0].set_ylabel("Option price", fontsize = 12)
axes[0].set_ylim(*ylim)

axes[0].annotate(f"Black\u2013Scholes = {BS:.4f}", xy = (-0.45, BS),
                 xytext=(-0.45, BS + 0.30 * (ylim[1] - BS)),
                 fontsize = 9, color = "#222222",
                 arrowprops=dict(arrowstyle = "-", color = "#222222", lw = 1))
 
fig.suptitle("Monte Carlo estimators vs analytic Black\u2013Scholes, by scheme",
             fontsize = 15, fontweight = "bold", y = 0.99)
fig.text(0.5, 0.005,
         f"error bars = 95% CI (\u00b11.96\u00b7SE)   |   N = {N_PATHS:,} paths"
         f"   |   euler/milstein: {N_STEPS} steps   |   shorter bar = better variance reduction",
         ha = "center", fontsize = 9, color = "#555555")
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
fig.savefig("../figures/methods_comparison.png", dpi = 150)
print("saved methods_comparison.png")
# plt.show()