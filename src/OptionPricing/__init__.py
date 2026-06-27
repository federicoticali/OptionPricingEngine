"""
European option pricing engine.
 
Public API:
    MarketData, VanillaOption        - immutable inputs (frozen dataclasses)
    BlackScholesPricer               - closed-form price + Greeks
    MonteCarloPricer                 - MC with variance reduction / schemes
    CoxRossRubinsteinPricer          - binomial tree
"""
 
from .OptionPricing import (
    MarketData,
    VanillaOption,
    BlackScholesPricer,
    MonteCarloPricer,
    CoxRossRubinsteinPricer,
)
 
__all__ = [
    "MarketData",
    "VanillaOption",
    "BlackScholesPricer",
    "MonteCarloPricer",
    "CoxRossRubinsteinPricer",
]
 
__version__ = "0.1.0"