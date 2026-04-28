from typing import Dict, Any
from pricer.products.base.rate_product import RateProduct
from pricer.products.rates.zero_coupon_bond import ZeroCouponBond


class FloatingRateBond(RateProduct):
    """
    Floating Rate Bond avec coupon C_t = (alpha·I_t + m) · D_t 
    """

    def __init__(self, nominal: float, maturity: float, spread: float,
                 frequency: int, rate_curve, alpha: float = 1.0):
        super().__init__()
        self.nominal = nominal
        self.maturity = maturity
        self.spread = spread
        self.alpha = alpha
        self.frequency = frequency
        self.rate_curve = rate_curve

        dt = 1.0 / frequency
        n = int(round(maturity * frequency))
        for i in range(1, n + 1):
            t_prev = dt * (i - 1)
            t_curr = dt * i
            df_prev = rate_curve.discount_factor(t_prev) if t_prev > 0 else 1.0
            df_curr = rate_curve.discount_factor(t_curr)
            simple_fwd = (df_prev / df_curr - 1.0) / dt
            coupon = nominal * (alpha * simple_fwd + spread) * dt
            self._add_leg(ZeroCouponBond(coupon, dt * i, rate_curve))
        self._add_leg(ZeroCouponBond(nominal, maturity, rate_curve))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "FloatingRateBond", "nominal": self.nominal,
                "maturity": self.maturity, "spread": self.spread, "alpha": self.alpha}
