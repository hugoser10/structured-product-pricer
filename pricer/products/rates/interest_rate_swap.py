from typing import Dict, Any
from pricer.products.base.rate_product import RateProduct
from pricer.products.rates.coupon_bond import CouponBond
from pricer.products.rates.floating_rate_bond import FloatingRateBond

class InterestRateSwap(RateProduct):
    """Swap = CouponBond(fixe) - FloatingRateBond (signe selon le sens)."""

    def __init__(self, nominal: float, fixed_rate: float, maturity: float,
                 frequency: int, pay_fixed: bool, rate_curve):
        super().__init__()

        self.nominal = nominal
        self.fixed_rate = fixed_rate
        self.maturity = maturity
        self.frequency = frequency
        self.pay_fixed = pay_fixed
        self.rate_curve = rate_curve

        fixed = CouponBond(nominal, fixed_rate, maturity, frequency, rate_curve)
        flt = FloatingRateBond(nominal, maturity, 0.0, frequency, rate_curve)

        # sens 
        if pay_fixed:
            self._add_leg(flt, +1.0)
            self._add_leg(fixed, -1.0)
        else:
            self._add_leg(fixed, +1.0)
            self._add_leg(flt, -1.0)

    def par_rate(self) -> float:
        # Taux fixe d'équilibre : (1 - DF(T)) / annuité
        dt = 1.0 / self.frequency
        n = int(round(self.maturity * self.frequency))
        ann = sum(self.rate_curve.discount_factor(dt * i) * dt for i in range(1, n + 1))
        return (1 - self.rate_curve.discount_factor(self.maturity)) / ann if ann else 0.0

    def greeks(self, **kwargs) -> Dict[str, float]:
        g = super().greeks(**kwargs)
        g["par_rate"] = self.par_rate()
        return g

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "InterestRateSwap", "nominal": self.nominal,
                "fixed_rate": self.fixed_rate, "maturity": self.maturity,
                "pay_fixed": self.pay_fixed}
