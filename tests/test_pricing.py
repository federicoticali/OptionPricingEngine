"""Test suite for the European option pricing engine.
 
Strategy: the analytic Black-Scholes price is the ground truth. Black-Scholes
itself is checked against John Hull's published textbook example, the Greeks
against finite differences of the price, and Monte Carlo / the CRR tree against
the analytic price (within their respective error sizes).
"""
 
import math
 
import numpy as np
import pytest
 
from option_pricing import (
    MarketData,
    VanillaOption,
    BlackScholesPricer,
    MonteCarloPricer,
    CoxRossRubensteinPricer,
)
 
SEED = 12345
 
# Hull, "Options, Futures, and Other Derivatives": S=49, K=50, r=5%, sigma=20%,
# T=20 weeks. The textbook rounds the call to 2.40; the precise analytic value
# (pinned here as a regression guard) is 2.4005. Greeks: delta ~ 0.522,
# gamma ~ 0.066, vega ~ 12.1, theta ~ -4.31, rho ~ 8.91.
HULL_CALL_PRICE = 2.4005
 
 
@pytest.fixture
def market():
    return MarketData(S=49, r=0.05, sigma=0.20)
 
 
@pytest.fixture
def call_opt():
    return VanillaOption(K=50, T=20 / 52, option_type="call")
 
 
@pytest.fixture
def put_opt():
    return VanillaOption(K=50, T=20 / 52, option_type="put")
 
 
# ----------------------------- Black-Scholes ------------------------------- #
 
def test_bs_call_matches_hull(market, call_opt):
    price = BlackScholesPricer(market, call_opt).price()
    assert round(price, 2) == 2.40                       # Hull's published figure
    assert price == pytest.approx(HULL_CALL_PRICE, abs=1e-3)
 
 
def test_put_call_parity(market, call_opt, put_opt):
    c = BlackScholesPricer(market, call_opt).price()
    p = BlackScholesPricer(market, put_opt).price()
    forward = market.S - call_opt.K * math.exp(-market.r * call_opt.T)
    assert (c - p) == pytest.approx(forward, abs=1e-12)
 
 
def test_call_intrinsic_bounds(market, call_opt):
    # A European call sits between its discounted intrinsic value and the spot.
    price = BlackScholesPricer(market, call_opt).price()
    lower = max(market.S - call_opt.K * math.exp(-market.r * call_opt.T), 0.0)
    assert lower <= price <= market.S
 
 
# ------------------------ Greeks via finite differences -------------------- #
 
def _bump_spot(market, ds):
    return MarketData(S=market.S + ds, r=market.r, sigma=market.sigma)
 
 
def _bump_sigma(market, dv):
    return MarketData(S=market.S, r=market.r, sigma=market.sigma + dv)
 
 
def _bump_rate(market, dr):
    return MarketData(S=market.S, r=market.r + dr, sigma=market.sigma)
 
 
def test_delta_matches_central_difference(market, call_opt):
    h = 1e-4
    up = BlackScholesPricer(_bump_spot(market, +h), call_opt).price()
    dn = BlackScholesPricer(_bump_spot(market, -h), call_opt).price()
    fd = (up - dn) / (2 * h)
    assert BlackScholesPricer(market, call_opt).delta() == pytest.approx(fd, abs=1e-6)
 
 
def test_gamma_matches_central_difference(market, call_opt):
    h = 1e-3
    bs = BlackScholesPricer(market, call_opt)
    up = BlackScholesPricer(_bump_spot(market, +h), call_opt).price()
    dn = BlackScholesPricer(_bump_spot(market, -h), call_opt).price()
    fd = (up - 2 * bs.price() + dn) / h**2
    assert bs.gamma() == pytest.approx(fd, abs=1e-4)
 
 
def test_vega_matches_central_difference(market, call_opt):
    h = 1e-5
    up = BlackScholesPricer(_bump_sigma(market, +h), call_opt).price()
    dn = BlackScholesPricer(_bump_sigma(market, -h), call_opt).price()
    fd = (up - dn) / (2 * h)
    assert BlackScholesPricer(market, call_opt).vega() == pytest.approx(fd, rel=1e-5)
 
 
def test_rho_matches_central_difference(market, call_opt):
    h = 1e-6
    up = BlackScholesPricer(_bump_rate(market, +h), call_opt).price()
    dn = BlackScholesPricer(_bump_rate(market, -h), call_opt).price()
    fd = (up - dn) / (2 * h)
    assert BlackScholesPricer(market, call_opt).rho() == pytest.approx(fd, rel=1e-5)
 
 
def test_theta_matches_central_difference(market, call_opt):
    # theta = dV/dt. Bumping T increases time-to-maturity, so dV/dt = -dV/dT.
    h = 1e-5
    up = BlackScholesPricer(
        market, VanillaOption(K=call_opt.K, T=call_opt.T + h, option_type="call")
    ).price()
    dn = BlackScholesPricer(
        market, VanillaOption(K=call_opt.K, T=call_opt.T - h, option_type="call")
    ).price()
    fd = -(up - dn) / (2 * h)
    assert BlackScholesPricer(market, call_opt).theta() == pytest.approx(fd, rel=1e-4)
 
 
# ------------------------------ Monte Carlo -------------------------------- #
 
@pytest.mark.parametrize("vr", [None, "antithetic", "control", "importance"])
def test_mc_exact_within_confidence_interval(market, call_opt, vr):
    """Every estimator (exact scheme) must land within a few SE of Black-Scholes."""
    bs = BlackScholesPricer(market, call_opt).price()
    np.random.seed(SEED)
    mc = MonteCarloPricer(market, call_opt)
    mc.scheme, mc.n_paths = "exact", 200_000
    price, se = mc.PricingMC(variance_reduction=vr)
    assert se > 0
    assert abs(price - bs) < 4 * se
 
 
@pytest.mark.parametrize("scheme", ["euler", "milstein"])
def test_mc_discretised_schemes_close_to_bs(market, call_opt, scheme):
    """Euler / Milstein carry a small discretisation bias but stay near BS."""
    bs = BlackScholesPricer(market, call_opt).price()
    np.random.seed(SEED)
    mc = MonteCarloPricer(market, call_opt)
    mc.scheme, mc.n_paths, mc.n_steps = scheme, 200_000, 500
    price, se = mc.PricingMC(variance_reduction=None)
    assert price == pytest.approx(bs, abs=0.05)
 
 
def test_variance_reduction_shrinks_standard_error(market, call_opt):
    """Control variates should give a smaller SE than plain Monte Carlo."""
    np.random.seed(SEED)
    mc = MonteCarloPricer(market, call_opt)
    mc.scheme, mc.n_paths = "exact", 100_000
    _, se_standard = mc.PricingMC(variance_reduction=None)
    np.random.seed(SEED)
    _, se_control = mc.PricingMC(variance_reduction="control")
    assert se_control < se_standard
 
 
def test_mc_put_within_confidence_interval(market, put_opt):
    bs = BlackScholesPricer(market, put_opt).price()
    np.random.seed(SEED)
    mc = MonteCarloPricer(market, put_opt)
    mc.scheme, mc.n_paths = "exact", 200_000
    price, se = mc.PricingMC(variance_reduction="control")
    assert abs(price - bs) < 4 * se
 
 
# --------------------------- Cox-Ross-Rubinstein --------------------------- #
 
def test_crr_converges_to_bs_call(market, call_opt):
    bs = BlackScholesPricer(market, call_opt).price()
    tree = CoxRossRubensteinPricer(market, call_opt)
    tree.n_steps = 2000
    assert tree.CoxRossRubensteinTree() == pytest.approx(bs, abs=5e-3)
 
 
def test_crr_converges_to_bs_put(market, put_opt):
    bs = BlackScholesPricer(market, put_opt).price()
    tree = CoxRossRubensteinPricer(market, put_opt)
    tree.n_steps = 2000
    assert tree.CoxRossRubensteinTree() == pytest.approx(bs, abs=5e-3)
 
 
def test_crr_error_decreases_with_steps(market, call_opt):
    bs = BlackScholesPricer(market, call_opt).price()
    coarse = CoxRossRubensteinPricer(market, call_opt)
    coarse.n_steps = 10
    fine = CoxRossRubensteinPricer(market, call_opt)
    fine.n_steps = 1000
    err_coarse = abs(coarse.CoxRossRubensteinTree() - bs)
    err_fine = abs(fine.CoxRossRubensteinTree() - bs)
    assert err_fine < err_coarse
 
 
# ----------------------------- Input validation ---------------------------- #
 
@pytest.mark.parametrize("bad", [dict(S=-1, r=0.05, sigma=0.2),
                                 dict(S=49, r=0.05, sigma=0.0)])
def test_marketdata_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):
        MarketData(**bad)
 
 
@pytest.mark.parametrize("bad", [dict(K=-1, T=0.5),
                                 dict(K=50, T=0.0),
                                 dict(K=50, T=0.5, option_type="banana"),
                                 dict(K=50, T=0.5, t_in=1.0)])
def test_vanillaoption_rejects_bad_inputs(bad):
    with pytest.raises(ValueError):