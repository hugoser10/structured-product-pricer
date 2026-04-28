from typing import Dict, Any
from pricer.products.base.equity_product import EquityProduct
from pricer.products.rates.zero_coupon_bond import ZeroCouponBond
from pricer.products.equity.call_spread import CallSpread

class CappedCapitalProtection(EquityProduct):
    """
    SSPA 1220 — Capital Protection with Cap.
    Réplication : ZCB(nominal) + participation · CallSpread(strike, cap).
    Payoff à maturité :
      nominal + participation · (min(S_T, cap) - strike)^+
    Le ZCB garantit le capital à maturité, le CallSpread offre la participation
    bornée par le `cap`.
    """

    def __init__(self, S: float, T: float, r: float,
                 strike: float, cap: float,
                 participation: float = 1.0, nominal: float = 100.0,
                 rate_curve=None, vol_surface=None, heston=None):
        super().__init__()
        self.S, self.T, self.r = S, T, r
        self.strike, self.cap = strike, cap
        self.participation = participation
        self.nominal = nominal

        if rate_curve is not None:
            self._add_leg(ZeroCouponBond(nominal, T, rate_curve), +1.0)
        # Si cap <= strike, le CallSpread n'a pas de sens : on ne l'ajoute pas.
        if cap > strike:
            self._add_leg(CallSpread(S, strike, cap, T, r,
                                      vol_surface=vol_surface, heston=heston),
                          participation)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "CappedCapitalProtection", "S": self.S, "T": self.T,
                "strike": self.strike, "cap": self.cap,
                "participation": self.participation, "sspa": 1220}
