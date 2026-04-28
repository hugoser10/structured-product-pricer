import numpy as np
from typing import Dict, Any
from pricer.products.base.atomic_rate_product import AtomicRateProduct

class ZeroCouponBond(AtomicRateProduct):
    """P = nominal · exp(-r(T)·T). Brique de tous les produits de taux."""

    def __init__(self, nominal: float, maturity: float, rate_curve):
        self.nominal = nominal
        self.maturity = maturity
        self.rate_curve = rate_curve

    def price(self, **kwargs) -> float:
        return self.nominal * np.exp(-self.rate_curve.zero_rate(self.maturity) * self.maturity)

    def _compute_greeks(self, **kwargs) -> Dict[str, float]:
        # DV01 = modified_duration · prix · 0.0001 
        # Pour un ZCB en taux continu : Macaulay = Modified = T.
        p = self.price()
        return {
            "price": p,
            "duration": self.maturity,
            "modified_duration": self.maturity,
            "dv01": self.maturity * p * 1e-4,
            "convexity": self.maturity ** 2,
            "zero_rate": self.rate_curve.zero_rate(self.maturity),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "ZeroCouponBond", "nominal": self.nominal, "maturity": self.maturity}
