from typing import Dict, Any
from pricer.products.base.rate_product import RateProduct
from pricer.products.rates.caplet import Caplet


class Floor(RateProduct):
    """sum Floorlet(T_{i-1}, T_i)."""

    def __init__(self, maturity: float, K: float, notional: float,
                 frequency: int, sigma: float, rate_curve):
        super().__init__()
        self.maturity = maturity
        self.K = K
        self.notional = notional
        self.frequency = frequency
        self.sigma = sigma

        dt = 1.0 / frequency
        n = int(round(maturity * frequency))
        for i in range(1, n + 1):
            self._add_leg(Caplet(dt * (i - 1), dt * i, K, notional, sigma,
                                 rate_curve, "floorlet"))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Floor", "maturity": self.maturity, "K": self.K, "notional": self.notional}
