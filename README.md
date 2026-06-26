# European OptionPricingEngine

A small, self-contained Python engine that prices European vanilla options in three independent
ways — Black–Scholes closed-form, Monte Carlo (several variance-reduction techniques and
discretisation schemes), and a Cox–Ross–Rubinstein binomial tree — plus a set of
convergence/validation plots showing all three methods agreeing on the same price.

The design is object-oriented and immutable at the data layer: market and contract parameters are
frozen dataclasses (MarketData, VanillaOption), and each pricing method is its own class
sharing those same inputs.


The default example throughout the code (S=49, K=50, r=5%, σ=20%, T=20/52) is John Hull's
textbook example (Chapter 18) — the analytic call is $\simeq2.40$, which makes it a convenient published
reference for testing (see Validation).



# Features


Analytic Black–Scholes price + full Greeks (Δ, Γ, Θ, ν, ρ) for calls and puts.
Monte Carlo pricer with:

three path schemes: exact (exact GBM step), euler, milstein;
four estimators: standard, antithetic variates, control variates (terminal stock
price as control), importance sampling;
a 95% confidence interval (±1.96·SE) reported on every estimate.



Cox–Ross–Rubinstein binomial tree (vectorised over the terminal layer via the binomial pmf).
Validation suite: three plotting scripts demonstrating convergence to the analytic price and
the finite-difference Greek-error "U-curve".


Methods at a glance

MethodTypeError behaviourGood forBlack–Scholesclosed formexactground truth / fast GreeksMonte CarlostochasticO(1/√N), CI shrinks with variance reductionpath-dependent generalisations, flexibilityCRR treedeterministicO(1/n), with odd–even oscillationearly-exercise generalisations, intuition


Project structure

.
├── OptionPricing.py          # core engine: MarketData, VanillaOption, the three pricers
├── ConvergencePlot.py        # Fig 1: MC + tree converging to BS; Fig 2: Greek FD-error U-curve
├── CoxRossRubensteinTree.py  # CRR odd–even oscillation + O(1/n) convergence rate
├── ErrorBars.py              # MC estimator comparison across schemes (price ± 95% CI)
└── README.md

The three plotting scripts all import from OptionPricing.py, so keep them next to it (or install
the package — see the roadmap).


Installation

bashgit clone https://github.com/<you>/<repo>.git
cd <repo>
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

requirements.txt:

numpy
scipy
matplotlib


Quick start

pythonfrom OptionPricing import (
    MarketData, VanillaOption,
    BlackScholesPricer, MonteCarloPricer, CoxRossRubensteinPricer,
)

# Hull's textbook example: S=49, K=50, r=5%, sigma=20%, T=20 weeks
market = MarketData(S=49, r=0.05, sigma=0.20)
option = VanillaOption(K=50, T=20 / 52, option_type="call")

# 1) Analytic Black–Scholes + Greeks
bs = BlackScholesPricer(market, option)
print(bs.price_n_greeks())          # price ~ 2.40

# 2) Monte Carlo with control variates
mc = MonteCarloPricer(market, option)
mc.scheme, mc.n_paths = "exact", 500_000
price, se = mc.PricingMC(variance_reduction="control")
print(f"{price:.4f} ± {1.96 * se:.4f}")

# 3) Cox–Ross–Rubinstein binomial tree
tree = CoxRossRubensteinPricer(market, option)
tree.n_steps = 500
print(tree.CoxRossRubensteinTree())


Reproducing the figures

bashpython ConvergencePlot.py          # -> price_convergence.png, greeks_fd_error.png
python CoxRossRubensteinTree.py    # -> crr_convergence.png
python ErrorBars.py                # -> methods_comparison.png


price_convergence.png — the three MC estimators (with ±1.96·SE bands) and the CRR tree all
converging to the analytic Black–Scholes line as the budget grows.
greeks_fd_error.png — the finite-difference error of Δ and Γ vs the bump size h: the
classic U-curve (truncation error ~h² on the way down, floating-point round-off on the way up).
crr_convergence.png — left: the odd–even oscillation of the tree price around BS; right:
|tree − BS| decaying as O(1/n), and how averaging consecutive trees (n, n+1) cancels the
oscillation.
methods_comparison.png — one panel per scheme (exact / euler / milstein); each estimator
shown as a point with its 95% CI. Shorter bar = better variance reduction.


