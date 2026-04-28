import numpy as np
from typing import Dict, Any
from pricer.products.base.atomic_rate_product import AtomicRateProduct


class BondOption(AtomicRateProduct):
    """Option d'exercice anticipé sur obligation, pricée par Monte Carlo Hull-White."""

    def __init__(self, nominal: float, coupon_rate: float, maturity: float,
                 exercise_dates: list, strike: float, frequency: int,
                 hw_model, option_type: str = "call"):
        self.nominal = nominal
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.exercise_dates = sorted(exercise_dates)
        self.strike = strike
        self.frequency = frequency
        self.hw_model = hw_model
        self.option_type = option_type

    def price(self, n_simulations: int = 5000, n_steps: int = 100,
              seed: int = 42, **kwargs) -> float:
        rc = self.hw_model.rate_curve
        r0 = rc.zero_rate(0.25) if rc else 0.03
        paths = self.hw_model.simulate_paths(r0, self.maturity, n_steps, n_simulations, seed)
        dt = self.maturity / n_steps
        coupon = self.nominal * self.coupon_rate / self.frequency
        dt_c = 1.0 / self.frequency
        cdates = [dt_c * i for i in range(1, int(self.maturity * self.frequency) + 1)]
        vals = np.zeros(n_simulations)

        # Pour chaque simu : à chaque date d'exercice, valeur = max(continuation, exercice)
        # On simplifie en prenant la meilleure valeur d'exercice rencontrée (Bermudan max)
        for sim in range(n_simulations):
            cum_df = np.exp(-np.cumsum(paths[sim, :-1]) * dt)
            for ex in self.exercise_dates:
                step = min(int(round(ex / dt)), n_steps - 1)
                df_ex = cum_df[step]
                r_ex = paths[sim, step]
                resid = sum(coupon * np.exp(-r_ex * (t - ex)) for t in cdates if t > ex)
                resid += self.nominal * np.exp(-r_ex * (self.maturity - ex))
                v = (max(resid - self.strike, 0) if self.option_type == "call"
                     else max(self.strike - resid, 0))
                vals[sim] = max(vals[sim], v * df_ex)
        return float(np.mean(vals))

    def _compute_greeks(self, **kwargs) -> Dict[str, float]:
        return {"price": self.price(**kwargs), "dv01": 0.0}

    def to_dict(self) -> Dict[str, Any]:
        return {"type": f"BondOption({self.option_type})", "strike": self.strike}
