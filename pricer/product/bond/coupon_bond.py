import numpy as np
from scipy.optimize import brentq
from typing import Dict, Any
from pricer.products.base.rate_product import RateProduct
from pricer.products.bond.zero_coupon_bond import ZeroCouponBond


class CouponBond(RateProduct):
    """sum ZCB(coupon_i) + ZCB(nominal). YTM par bisection, duration de Macaulay."""

    def __init__(self, nominal: float, coupon_rate: float, maturity: float,
                 frequency: int, rate_curve):
        super().__init__()
        self.nominal = nominal
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.frequency = frequency
        self.rate_curve = rate_curve

        coupon = nominal * coupon_rate / frequency
        dt = 1.0 / frequency
        n = int(round(maturity * frequency))
        for i in range(1, n + 1):
            self._add_leg(ZeroCouponBond(coupon, dt * i, rate_curve))
        self._add_leg(ZeroCouponBond(nominal, maturity, rate_curve))

    def yield_to_maturity(self) -> float:
        # YTM = taux unique tel que sum coupon_i·exp(-y·t_i) + N·exp(-y·T) = prix
        p = self.price()
        coupon = self.nominal * self.coupon_rate / self.frequency
        dt = 1.0 / self.frequency
        n = int(round(self.maturity * self.frequency))
        dates = [dt * i for i in range(1, n + 1)]

        def f(y):
            pv = sum(coupon * np.exp(-y * t) for t in dates)
            pv += self.nominal * np.exp(-y * self.maturity)
            return pv - p

        # formule si l'analytique ne fonctionne pas
        try:
            return brentq(f, -0.2, 0.5)
        except Exception:
            return self.rate_curve.zero_rate(self.maturity)

    def greeks(self, **kwargs) -> Dict[str, float]:
        g = super().greeks(**kwargs)
        ytm = self.yield_to_maturity()
        p = g["price"]
        coupon = self.nominal * self.coupon_rate / self.frequency
        dt = 1.0 / self.frequency
        n = int(round(self.maturity * self.frequency))

        # Duration de Macaulay : sum (t_i · PV_i) / B 
        flows_t = [dt * i for i in range(1, n + 1)] + [self.maturity]
        flows_amt = [coupon] * n + [self.nominal]
        pvs = [a * np.exp(-ytm * t) for a, t in zip(flows_amt, flows_t)]
        mac = sum(t * pv for t, pv in zip(flows_t, pvs)) / p
        modified = mac / (1 + ytm / self.frequency)
        # Convexité : C = (1/B) sum exp(-r·t) F_t · t*t
        convex = sum(pv * t ** 2 for t, pv in zip(flows_t, pvs)) / p

        g["ytm"] = ytm
        g["duration"] = mac
        g["modified_duration"] = modified
        g["convexity"] = convex
        g["dv01"] = modified * p * 1e-4
        return g

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "CouponBond", "nominal": self.nominal,
                "coupon_rate": self.coupon_rate, "maturity": self.maturity,
                "frequency": self.frequency}
