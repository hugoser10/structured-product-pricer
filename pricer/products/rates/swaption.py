import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict, Any
from pricer.products.base.atomic_rate_product import AtomicRateProduct


class Swaption(AtomicRateProduct):
    """Swaption Black 76. payer = N·A(T)·[S0·N(d1) − K·N(d2)]."""

    def __init__(self, T: float, swap_maturity: float, K: float,
                 notional: float, sigma: float, frequency: int,
                 swaption_type: str, rate_curve):
        self.T = T
        self.swap_maturity = swap_maturity
        self.K = K
        self.notional = notional
        self.sigma = sigma
        self.frequency = frequency
        self.swaption_type = swaption_type.lower()
        self.rate_curve = rate_curve

    def _swap_rate_and_annuity(self) -> Tuple[float, float]:
        # S0 = (DF(T) - DF(T + swap_maturity)) / annuité ; annuité = Σ DF(t_i) * dt
        rc = self.rate_curve
        dt = 1.0 / self.frequency
        n = int(round(self.swap_maturity * self.frequency))
        dates = [self.T + dt * i for i in range(1, n + 1)]
        annuity = sum(rc.discount_factor(t) * dt for t in dates)
        df_start = rc.discount_factor(self.T)
        df_end = rc.discount_factor(self.T + self.swap_maturity)
        swap_rate = (df_start - df_end) / annuity if annuity > 0 else 0.0
        return swap_rate, annuity

    def price(self, **kwargs) -> float:
        S0, annuity = self._swap_rate_and_annuity()
        if self.T <= 0 or self.sigma <= 0 or S0 <= 0:
            payoff = max(S0 - self.K, 0) if self.swaption_type == "payer" else max(self.K - S0, 0)
            return payoff * annuity * self.notional
        sq = np.sqrt(self.T)
        d1 = (np.log(S0 / self.K) + 0.5 * self.sigma**2 * self.T) / (self.sigma * sq)
        d2 = d1 - self.sigma * sq
        if self.swaption_type == "payer":
            return float(self.notional * annuity * (S0 * norm.cdf(d1) - self.K * norm.cdf(d2)))
        return float(self.notional * annuity * (self.K * norm.cdf(-d2) - S0 * norm.cdf(-d1)))

    def _compute_greeks(self, **kwargs) -> Dict[str, float]:
        S0, annuity = self._swap_rate_and_annuity()
        p = self.price(**kwargs)
        if self.T <= 0 or self.sigma <= 0 or S0 <= 0:
            return {"price": p, "dv01": 0.0}
        sq = np.sqrt(self.T)
        d1 = (np.log(S0 / self.K) + 0.5 * self.sigma**2 * self.T) / (self.sigma * sq)
        if self.swaption_type == "payer":
            delta_rate = self.notional * annuity * norm.cdf(d1)
        else:
            delta_rate = -self.notional * annuity * norm.cdf(-d1)
        return {"price": p, "dv01": delta_rate * 1e-4, "delta_rate": delta_rate,
                "swap_rate": S0, "annuity": annuity}

    def to_dict(self) -> Dict[str, Any]:
        return {"type": f"Swaption({self.swaption_type})", "T": self.T,
                "swap_maturity": self.swap_maturity, "K": self.K, "notional": self.notional}
