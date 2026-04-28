import numpy as np
from scipy.stats import norm
from typing import Dict, Any
from pricer.products.base.atomic_rate_product import AtomicRateProduct


class Caplet(AtomicRateProduct):
    """Caplet/Floorlet par formule de Black 76."""

    def __init__(self, T1: float, T2: float, K: float, notional: float,
                 sigma: float, rate_curve, caplet_type: str = "caplet"):
        self.T1, self.T2 = T1, T2
        self.K = K
        self.notional = notional
        self.sigma = sigma
        self.rate_curve = rate_curve
        self.caplet_type = caplet_type.lower()

    def price(self, **kwargs) -> float:
        rc = self.rate_curve
        F = rc.forward_rate(max(self.T1, 1e-4), self.T2)
        P1 = rc.discount_factor(self.T1)
        tau = self.T2 - self.T1

        if self.T1 <= 0 or self.sigma <= 0:
            payoff = max(F - self.K, 0) if self.caplet_type == "caplet" else max(self.K - F, 0)
            return payoff * self.notional * tau * rc.discount_factor(self.T2)

        sq = np.sqrt(self.T1)
        d1 = (np.log(F / self.K) + 0.5 * self.sigma**2 * self.T1) / (self.sigma * sq)
        d2 = d1 - self.sigma * sq
        if self.caplet_type == "caplet":
            return float(P1 * tau * self.notional * (F * norm.cdf(d1) - self.K * norm.cdf(d2)))
        return float(P1 * tau * self.notional * (self.K * norm.cdf(-d2) - F * norm.cdf(-d1)))

    def _compute_greeks(self, **kwargs) -> Dict[str, float]:
        # DV01 approximé : sensibilité linéaire à un déplacement parallèle de la courbe forward
        p = self.price()
        tau = self.T2 - self.T1
        F = self.rate_curve.forward_rate(max(self.T1, 1e-4), self.T2)
        P1 = self.rate_curve.discount_factor(self.T1)
        dv01 = -P1 * tau * self.notional * tau * 1e-4 if F > 0 else 0.0
        return {"price": p, "dv01": dv01}

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.caplet_type, "T1": self.T1, "T2": self.T2,
                "K": self.K, "notional": self.notional}
