"""
Convergence & validation plots for the European Option Pricing Engine.
 
Class-based: built on the OOP API in OptionPricing.py
  (MarketData, VanillaOption, BlackScholesPricer, MonteCarloPricer, CoxRossRubensteinPricer).
 
Produces two figures:
  1) price_convergence.png - the Monte Carlo estimators (standard, antithetic,
     control) with their +/-1.96*SE bands, and the Cox-Ross-Rubinstein tree, all
     converging to the analytic Black-Scholes price.
  2) greeks_fd_error.png - the finite-difference error of Delta and Gamma vs
     the bump size h: the classic "U-curve" (truncation error ~ h^2 on the way
     down, floating-point round-off on the way up).
 
"""
 
import numpy as np
import matplotlib.pyplot as plt
 
from OptionPricing import (
    MarketData, VanillaOption, BlackScholesPricer,
    MonteCarloPricer, CoxRossRubensteinPricer,
)

Market = MarketData(S = 49, r = 0.05, sigma = 0.2)
Option = VanillaOption(K = 50, T = 20 / 52, option_type = "call")
BS = BlackScholesPricer(Market, Option).price()
 
C = {"Standard": "#9aa0a6", "Antithetic": "#1f77b4", "Control": "#2ca02c",
     "CRR tree": "#d62728", "BS": "#111111"}

SEED = 12345
np.random.default_rng(SEED) 

def mc_estimate(n_paths, variance_reduction):
    """One Monte Carlo estimate (price, std_error) at the given budget (exact scheme)."""
    mc = MonteCarloPricer(Market, Option)
    mc.scheme = "exact"
    mc.n_paths = n_paths
    return mc.PricingMC(variance_reduction = variance_reduction)
 
 
def tree_estimate(n_steps):
    tree = CoxRossRubensteinPricer(Market, Option)
    tree.n_steps = n_steps
    return tree.CoxRossRubensteinTree()
 
 
def bs_price_at(spot):
    """Black-Scholes price with the spot bumped to `spot` (MarketData is frozen)."""
    bumped = MarketData(S = spot, r = Market.r, sigma = Market.sigma)
    return BlackScholesPricer(bumped, Option).price()
 

# Figure 1
def price_convergence_plot(seed=12345, fname="price_convergence.png"):
    """MC estimators (with CI bands) and the CRR tree converging to Black-Scholes."""
    Ns = np.unique(np.logspace(2.5, 6, 16).astype(int))     # 316 to 1,000,000
    methods = {"Standard": None, "Antithetic": "antithetic", "Control": "control"}
 
    fig, ax = plt.subplots(figsize = (9, 5.5))
 
    for label, vr in methods.items():
        prices, ses = [], []
        for N in Ns:
            np.random.seed(seed)
            n = N // 2 if vr == "antithetic" else N
            p, se = mc_estimate(n, vr)
            prices.append(p); ses.append(se)
        prices, ses = np.array(prices), np.array(ses)
        ax.fill_between(Ns, prices - 1.96 * ses, prices + 1.96 * ses,
                        color = C[label], alpha = 0.18, linewidth = 0)
        ax.plot(Ns, prices, "-o", ms = 3.5, lw = 1.2, color = C[label],
                label = f"{label} MC  (\u00b11.96\u00b7SE)")
 
    steps = np.unique(np.logspace(1, 3.3, 22).astype(int))
    tree = [tree_estimate(int(m)) for m in steps]
    ax.plot(steps, tree, "-s", ms = 3, lw = 1.0, color = C["CRR tree"], alpha=0.85,
            label = "CRR tree")
 
    ax.axhline(BS, color = C["BS"], ls = "--", lw = 1.3, label = f"Black-Scholes = {BS:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("N   (Monte Carlo payoff budget   /   tree steps)")
    ax.set_ylabel("Option price")
    ax.set_title("Convergence to Black\u2013Scholes: MC variance reduction & CRR tree")
    ax.set_ylim(BS - 0.30, BS + 0.30)
    ax.grid(alpha = 0.3)
    ax.legend(frameon = False, fontsize = 9, loc = "upper right")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"saved {fname}")
    return fig
 
 
# Figure 2
def greeks_fd_error_plot(fname="greeks_fd_error.png"):
    """Finite-difference error of Delta and Gamma vs bump size h: the U-curve."""
    hs = np.logspace(-10, 0, 41)
    bs = BlackScholesPricer(Market, Option)
    d_exact, g_exact, mid = bs.delta(), bs.gamma(), bs.price()
 
    d_err, g_err = [], []
    for h in hs:
        up = bs_price_at(Market.S + h)
        dn = bs_price_at(Market.S - h)
        d_err.append(abs((up - dn) / (2 * h) - d_exact))          # Delta error
        g_err.append(abs((up - 2 * mid + dn) / h**2 - g_exact))   # Gamma error
 
    fig, ax = plt.subplots(figsize = (8, 5.5))
    ax.loglog(hs, d_err, "-o", ms=3, color="#1f77b4", label="Delta  (1st central diff)")
    ax.loglog(hs, g_err, "-s", ms=3, color="#d62728", label="Gamma  (2nd central diff)")
    ax.set_xlabel("bump size  h")
    ax.set_ylabel("|finite difference  \u2212  analytic|")
    ax.set_title("Finite-difference error vs bump size: the U-curve")
    ax.grid(alpha = 0.3, which = "both")
    ax.legend(frameon = False, fontsize = 10)
    fig.text(0.5, 0.01, "right of the minimum: truncation error ~h^2   |   "
                        "left: floating-point round-off dominates",
             ha = "center", fontsize = 8, color = "#555555")
    fig.tight_layout(rect = (0, 0.03, 1, 1))
    fig.savefig(fname, dpi = 150)
    print(f"saved {fname}")
    return fig
 
 
if __name__ == "__main__":
    price_convergence_plot()
    greeks_fd_error_plot()
    # plt.show()
