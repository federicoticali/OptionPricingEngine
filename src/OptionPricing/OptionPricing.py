from scipy.stats import norm, binom
import numpy as np
from dataclasses import dataclass
import warnings

@dataclass(frozen = True)
class MarketData:
    S: float
    r: float
    sigma: float

    def __post_init__(self):
        if self.S <= 0 or self.sigma <= 0:
            raise ValueError("S and sigma must be greater than 0!")

@dataclass(frozen = True)
class VanillaOption:
    K: float
    T: float
    t_in: float = 0
    option_type: str = 'call'

    def __post_init__(self):
        if self.K <= 0 or self.T <= 0:
            raise ValueError("K and T must be greater than 0!")
        if self.option_type not in ('call','put'):
            raise ValueError(f"'option' or 'call' value for option_type expect, {self.option_type}, got instead.")
        if self.t_in > self.T:
            raise ValueError("Initial time cannot be larger than maturity time!")

class BlackScholesPricer:
    def __init__(self, market: MarketData, option: VanillaOption):
        self.option = option
        self.market = market
        self._d1_d2() 
    
    def _d1_d2(self):
        opt, mkt = self.option, self.market
        self.d1 = (np.log(mkt.S / opt.K) + (mkt.r + 1 / 2 * mkt.sigma**2) * (opt.T - opt.t_in)) / (mkt.sigma * np.sqrt(opt.T - opt.t_in))
        self.d2 = self.d1 - mkt.sigma * np.sqrt(opt.T - opt.t_in)
        self.N1 = norm.cdf(self.d1)
        self.N2 = norm.cdf(self.d2)
        self.N1prime = norm.pdf(self.d1)
        self.N1minus = norm.cdf(- self.d1)
        self.N2minus = norm.cdf(- self.d2)

    def price(self) -> float:
        opt, mkt = self.option, self.market
        if opt.option_type == 'call':
            return mkt.S * self.N1 - opt.K * np.exp(- mkt.r * (opt.T - opt.t_in)) * self.N2
        return opt.K * np.exp(- mkt.r * (opt.T - opt.t_in)) * self.N2minus - mkt.S * self.N1minus
    
    def delta(self) -> float:
        opt = self.option
        if opt.option_type == 'call':
            return self.N1
        return self.N1 - 1
    
    def theta(self) -> float:
        opt, mkt = self.option, self.market
        fact_1 = - mkt.S * self.N1prime * mkt.sigma / (2 * np.sqrt(opt.T - opt.t_in))
        fact_2 = - mkt.r * opt.K * np.exp(- mkt.r * (opt.T - opt.t_in)) * self.N2
        if opt.option_type == 'call':
            return  fact_1 + fact_2
        return fact_1 + fact_2 + mkt.r * opt.K * np.exp(- mkt.r * (opt.T - opt.t_in))

    def gamma(self) -> float:
        opt, mkt = self.option, self.market
        return self.N1prime / (mkt.S * mkt.sigma * np.sqrt(opt.T - opt.t_in))
    
    def vega(self) -> float:
        opt, mkt = self.option, self.market
        return mkt.S * np.sqrt(opt.T - opt.t_in) * self.N1prime
    
    def rho(self) -> float:
        opt, mkt = self.option, self.market
        if opt.option_type == 'call':
            return opt.K * (opt.T - opt.t_in) * np.exp(- mkt.r * (opt.T - opt.t_in)) * self.N2
        return - opt.K * (opt.T - opt.t_in) * np.exp(- mkt.r * (opt.T - opt.t_in)) * self.N2minus
    
    def price_n_greeks(self) -> dict[str, float]:
        return {
            "price": self.price(),
            "delta": self.delta(),
            "theta": self.theta(),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "rho": self.rho()
        }
    
class MonteCarloPricer:
    scheme: str = 'exact'
    n_paths: int = 200000
    n_steps: int = 1000
    mu: float = None

    def __init__(self, market: MarketData, option: VanillaOption):
        self.option = option
        self.market = market
        self._d1_d2()
        if self.scheme not in ('exact','euler','milstein'):
            raise ValueError(f"'exact', 'euler' or 'milstein' expected in scheme variable, {self.scheme} got instead.")  
    
    def _d1_d2(self):
        opt, mkt = self.option, self.market
        self.d1 = (np.log(mkt.S / opt.K) + (mkt.r + 1 / 2 * mkt.sigma**2) * (opt.T - opt.t_in)) / (mkt.sigma * np.sqrt(opt.T - opt.t_in))
        self.d2 = self.d1 - mkt.sigma * np.sqrt(opt.T - opt.t_in)
        self.N1 = norm.cdf(self.d1)
        self.N2 = norm.cdf(self.d2)
        self.N1prime = norm.pdf(self.d1)

    def _payoff(self, S: float):
        opt = self.option
        if opt.option_type == 'call':
            return np.maximum(S - opt.K, 0)
        return np.maximum(opt.K - S, 0)
    
    def StandardMC(self):
        opt, mkt = self.option, self.market
        if self.scheme == 'exact':
            Z = np.random.standard_normal(self.n_paths)
            S_T = mkt.S * np.exp(mkt.sigma * np.sqrt(opt.T - opt.t_in) * Z + (mkt.r - 1 / 2 * mkt.sigma**2) * (opt.T - opt.t_in))
        else:
            dt = (opt.T - opt.t_in) / self.n_steps
            S = np.full(self.n_paths, float(mkt.S))
            for _ in range(self.n_steps):
                Z = np.random.standard_normal(self.n_paths)
                increment = mkt.r * S * dt + mkt.sigma * S * np.sqrt(dt) * Z
                if self.scheme == 'milstein':
                    increment += 1 / 2 * mkt.sigma**2 * S * dt * (Z**2 - 1)
                S += increment
            S_T = S
        discounted = np.exp(- mkt.r * (opt.T - opt.t_in)) * self._payoff(S_T)
        price = float(discounted.mean())
        std_error = float(discounted.std(ddof = 1)) / np.sqrt(self.n_paths)
        return price, std_error
    
    def AntitheticMC(self):
        opt, mkt = self.option, self.market
        if self.scheme == 'exact':
            Z = np.random.standard_normal(self.n_paths)
            Z_minus = - Z
            S_T = mkt.S * np.exp(mkt.sigma * np.sqrt(opt.T - opt.t_in) * Z + (mkt.r - 1 / 2 * mkt.sigma**2) * (opt.T - opt.t_in))
            S_Tminus = mkt.S * np.exp(mkt.sigma * np.sqrt(opt.T - opt.t_in) * Z_minus + (mkt.r - 1 / 2 * mkt.sigma**2) * (opt.T - opt.t_in))
        else:
            dt = (opt.T - opt.t_in) / self.n_steps
            S = np.full(self.n_paths // 2, float(mkt.S))
            S_minus = np.full(self.n_paths // 2, float(mkt.S))
            for _ in range(self.n_steps):
                Z = np.random.standard_normal(self.n_paths // 2)
                Z_minus = - Z
                increment = mkt.r * S * dt + mkt.sigma * S * np.sqrt(dt) * Z
                increment_minus = mkt.r * S_minus * dt + mkt.sigma * S_minus * np.sqrt(dt) * Z_minus
                if self.scheme == 'milstein':
                    increment += 1 / 2 * mkt.sigma**2 * S * dt * (Z**2 - 1)
                    increment_minus += 1 / 2 * mkt.sigma**2 * S_minus * dt * (Z_minus**2 - 1)
                S += increment
                S_minus += increment_minus
            S_T = S
            S_Tminus = S_minus
        P = self._payoff(S_T)
        P_minus = self._payoff(S_Tminus)
        P = (P + P_minus) / 2
        discounted = np.exp(- mkt.r * (opt.T - opt.t_in)) * P
        price = float(discounted.mean())
        std_error = float(discounted.std(ddof = 1) / np.sqrt(self.n_paths))
        return price, std_error
    
    def ControlMC(self):
        opt, mkt = self.option, self.market
        if self.scheme == 'exact':
            Z = np.random.standard_normal(self.n_paths)
            S_T = mkt.S * np.exp(mkt.sigma * np.sqrt(opt.T - opt.t_in) * Z + (mkt.r - 1 / 2 * mkt.sigma**2) * (opt.T - opt.t_in))
            ES_T = mkt.S * np.exp(mkt.r * (opt.T - opt.t_in))
        else:
            dt = (opt.T - opt.t_in) / self.n_steps
            S = np.full(self.n_paths, float(mkt.S))
            for _ in range(self.n_steps):
                Z = np.random.standard_normal(self.n_paths)
                increment = mkt.r * S * dt + mkt.sigma * S * np.sqrt(dt) * Z
                if self.scheme == 'milstein':
                    increment += 1 / 2 * mkt.sigma**2 * S * dt * (Z**2 - 1)
                S += increment
            S_T = S
            ES_T = mkt.S * (1 + mkt.r * dt)**self.n_steps
        discount = np.exp(- mkt.r * (opt.T - opt.t_in))
        Y = discount * self._payoff(S_T)
        X = discount * S_T
        EX = discount * ES_T
        cov = np.cov(Y, X, ddof = 1) 
        c_star = cov[0,1] / cov[1,1]
        Y_star = Y - c_star * (X - EX)
        price = float(Y_star.mean())
        std_error = float(Y_star.std(ddof = 1) / np.sqrt(self.n_paths))
        return price, std_error
    
    def ImportanceSamplingMC(self):
        opt, mkt = self.option, self.market
        discount = np.exp(- mkt.r * (opt.T - opt.t_in))
        mu = - self.d2 if self.mu is None else self.mu
        theta = mu / np.sqrt(self.n_steps)
        if self.scheme == 'exact':
            Z = np.random.standard_normal(self.n_paths)
            Y = Z + mu
            S_T = mkt.S * np.exp((mkt.r - 1 / 2 * mkt.sigma**2) * (opt.T - opt.t_in) + mkt.sigma * np.sqrt(opt.T - opt.t_in) * Y)
            L = np.exp(- mu * Y + 1 / 2 * mu**2)
        else:
            dt = (opt.T - opt.t_in) / self.n_steps
            S = np.full(self.n_paths, float(mkt.S))
            Y_total = np.zeros(self.n_paths)
            for _ in range(self.n_steps):
                Z = np.random.standard_normal(self.n_paths)
                Y = Z + theta
                Y_total += Y
                increment = mkt.r * S * dt + mkt.sigma * S * np.sqrt(dt) * Y
                if self.scheme == 'milstein':
                    increment += 1 / 2 * mkt.sigma**2 * S * dt * (Y**2 - 1)
                S += increment
            S_T = S
            L = np.exp(- theta * Y_total + 1 / 2 * self.n_steps * theta**2)
        weighted = discount * self._payoff(S_T) * L
        price = float(weighted.mean())
        std_error = float(weighted.std(ddof=1) / np.sqrt(self.n_paths))
        return price, std_error
    
    def PricingMC(self, variance_reduction: str = None):
        if variance_reduction is None:
            return self.StandardMC()
        elif variance_reduction == 'antithetic':
            return self.AntitheticMC()
        elif variance_reduction == 'control':
            return self.ControlMC()
        elif variance_reduction == 'importance':
            return self.ImportanceSamplingMC()
        else:
            warnings.warn("No valid variance reduction technique requested, StandardMC will be executed.")
            return self.StandardMC()

class CoxRossRubensteinPricer:
    n_steps: int = 1000

    def __init__(self, market: MarketData, option: VanillaOption):
        self.option = option
        self.market = market  

    def _payoff(self, S: float):
        opt = self.option
        if opt.option_type == 'call':
            return np.maximum(S - opt.K, 0)
        return np.maximum(opt.K - S, 0)

    def CoxRossRubensteinTree(self):
        opt, mkt = self.option, self.market
        dt = (opt.T - opt.t_in) / self.n_steps
        u = np.exp(mkt.sigma * np.sqrt(dt))
        d = 1 / u
        a = np.exp(mkt.r * dt)
        p = (a - d) / (u - d)
        f = 0
        S0 = mkt.S
        for n in range(self.n_steps + 1):
            S = S0 * u**n * d**(self.n_steps - n)
            f += binom.pmf(n, self.n_steps, p) * self._payoff(S)
        return np.exp(- mkt.r * (opt.T - opt.t_in)) * f
