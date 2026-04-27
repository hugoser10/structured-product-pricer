from typing import Dict, Any
from pricer.products.base.rate_product import RateProduct
from pricer.products.rates.zero_coupon_bond import ZeroCouponBond


class FloatingRateBond(RateProduct):
    """Coupons indexés sur le forward de la période + spread."""

    def __init__(self, nominal: float, maturity: float, spread: float,
                 frequency: int, rate_curve):
        super().__init__()
        self.nominal = nominal
        self.maturity = maturity
        self.spread = spread
        self.frequency = frequency
        self.rate_curve = rate_curve

        dt = 1.0 / frequency
        n = int(round(maturity * frequency))
        for i in range(1, n + 1):
            fwd = rate_curve.forward_rate(max(dt * (i - 1), 1e-4), dt * i)
            self._add_leg(ZeroCouponBond(nominal * (fwd + spread) * dt, dt * i, rate_curve))
        self._add_leg(ZeroCouponBond(nominal, maturity, rate_curve))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "FloatingRateBond", "nominal": self.nominal,
                "maturity": self.maturity, "spread": self.spread}
