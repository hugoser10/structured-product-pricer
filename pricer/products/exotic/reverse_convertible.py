from typing import Dict, Any
from pricer.products.base.equity_product import EquityProduct
from pricer.products.rates.coupon_bond import CouponBond
from pricer.products.rates.zero_coupon_bond import ZeroCouponBond
from pricer.products.exotic.barrier_option import BarrierOption

class ReverseConvertible(EquityProduct):
    """
    SSPA 1230 — Barrier Reverse Convertible.
    Réplication : CouponBond(coupon élevé) - Put down-and-in à `barrier`.
    Payoff à maturité :
      - si la barrière n'a jamais été touchée OU S_T > strike : nominal
      - sinon : nominal · S_T / strike (perte en capital limitée à la baisse)
      Plus les coupons fixes versés tout au long de la vie.

    Le coupon élevé compense la vente du put barrière à l'investisseur.
    """

    def __init__(self, S: float, T: float, r: float,
                 strike: float, barrier: float,
                 coupon_rate: float = 0.0, frequency: int = 2,
                 nominal: float = 100.0, rate_curve=None,
                 vol_surface=None, heston=None):
        super().__init__()
        self.S, self.T, self.r = S, T, r
        self.strike, self.barrier = strike, barrier
        self.coupon_rate = coupon_rate
        self.nominal = nominal
        bt = "down-and-in" if barrier < S else "up-and-in"

        # Jambe taux : bond classique avec coupons (ou ZCB si coupon_rate = 0)
        if rate_curve is not None and coupon_rate > 0:
            self._add_leg(CouponBond(nominal, coupon_rate, T, frequency, rate_curve), +1.0)
        elif rate_curve is not None:
            self._add_leg(ZeroCouponBond(nominal, T, rate_curve), +1.0)

        # Jambe equity : short Put barrière -> l'investisseur supporte la baisse
        # (notionnel = nominal car perte plafonnée au capital investi)
        self._add_leg(BarrierOption(S, strike, T, r, barrier,
                                     barrier_type=bt, option_type="put",
                                     heston=heston, vol_surface=vol_surface),
                      -nominal / strike)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "ReverseConvertible", "S": self.S, "T": self.T,
                "strike": self.strike, "barrier": self.barrier,
                "coupon_rate": self.coupon_rate, "sspa": 1230}
