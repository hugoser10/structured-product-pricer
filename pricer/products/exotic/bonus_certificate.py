import numpy as np
from typing import Dict, Any
from pricer.products.base.equity_product import EquityProduct
from pricer.products.equity.option import Option
from pricer.products.exotic.barrier_option import BarrierOption
from pricer.products.rates.zero_coupon_bond import ZeroCouponBond

class BonusCertificate(EquityProduct):
    """
    SSPA 1320 — Bonus Certificate.
    Réplication : Call(K=bonus) + ZCB(bonus) - Put_KI(K=bonus, barrier).
    Payoff :
      - si la barrière n'a jamais été touchée : max(S_T, bonus) = Call(K=bonus) + bonus
      - sinon : S_T  (la garantie est annulée par le Put_KI)
    """

    def __init__(self, S: float, T: float, r: float,
                 barrier: float, bonus_level: float = None,
                 rate_curve=None, vol_surface=None, heston=None):
        super().__init__()
        self.S, self.T, self.r = S, T, r
        self.barrier = barrier
        self.bonus_level = bonus_level if bonus_level is not None else S
        self.rate_curve = rate_curve
        bt = "down-and-in" if barrier < S else "up-and-in"

        # Long Call(K=bonus) : participation à la hausse au-delà du niveau garanti
        self._add_leg(Option(S, self.bonus_level, T, r, "call",
                              vol_surface=vol_surface, heston=heston), +1.0)
        # ZCB(bonus) : garantit le niveau bonus à maturité
        if rate_curve is not None:
            self._add_leg(ZeroCouponBond(self.bonus_level, T, rate_curve), +1.0)
        else:
            # Actualisation au taux flat si pas de courbe disponible
            self._zcb_value = self.bonus_level * np.exp(-r * T)

        # Short Put_KI(K=bonus) : annule la garantie si la barrière est touchée
        self._add_leg(BarrierOption(S, self.bonus_level, T, r, barrier,
                                     barrier_type=bt, option_type="put",
                                     heston=heston, vol_surface=vol_surface), -1.0)

    def price(self, **kwargs) -> float:
        p = super().price(**kwargs)
        if self.rate_curve is None:
            p += self._zcb_value
        return p

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "BonusCertificate", "S": self.S, "T": self.T,
                "barrier": self.barrier, "bonus_level": self.bonus_level,
                "sspa": 1320}
