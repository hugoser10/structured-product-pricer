from typing import Dict, Any
from pricer.products.base.equity_product import EquityProduct
from pricer.products.equity.option import Option
from pricer.products.equity.barrier_option import BarrierOption

class BonusCertificate(EquityProduct):
    """
    SSPA 1130 — Bonus Certificate.
    Réplication : Call(K=0) + Put(K=B_bonus) - Put down-and-in à barrier1.
    Payoff :
      - si la barrière n'a jamais été touchée : max(S_T, B_bonus)
      - sinon : S_T (perte du niveau garanti).
    Approximation utilisée : Call(K≈0) - Put_KI(K=B_bonus, barrier=barrier1).
    """

    def __init__(self, S: float, T: float, r: float,
                 barrier: float, bonus_level: float = None,
                 rate_curve=None, vol_surface=None, heston=None):
        super().__init__()
        self.S, self.T, self.r = S, T, r
        self.barrier = barrier
        self.bonus_level = bonus_level if bonus_level is not None else S
        bt = "down-and-in" if barrier < S else "up-and-in"
        # rate_curve accepté pour API uniforme mais non utilisé (pas de ZCB explicite)

        # Long Call(K=0) -> réplique S_T
        self._add_leg(Option(S, 1e-3, T, r, "call",
                              vol_surface=vol_surface, heston=heston), +1.0)
        # Long Put(K=bonus) → ajoute le niveau garanti 
        self._add_leg(Option(S, self.bonus_level, T, r, "put",
                              vol_surface=vol_surface, heston=heston), +1.0)
        #  annulé si la barrière est touchée (short Put down-and-in)
        self._add_leg(BarrierOption(S, self.bonus_level, T, r, barrier,
                                     barrier_type=bt, option_type="put",
                                     heston=heston, vol_surface=vol_surface), -1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "BonusCertificate", "S": self.S, "T": self.T,
                "barrier": self.barrier, "bonus_level": self.bonus_level,
                "sspa": 1130}
