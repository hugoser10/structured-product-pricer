from typing import Dict, Any
from pricer.products.base.equity_product import EquityProduct
from pricer.products.rates.zero_coupon_bond import ZeroCouponBond
from pricer.products.equity.option import Option

class TrackerCertificate(EquityProduct):
    """
    SSPA 1300 — Tracker Certificate.
    Réplication : participation × Call(K≈0).
    Payoff à maturité : participation × S_T (réplique linéairement le sous-jacent).
    """

    def __init__(self, S: float, T: float, r: float,
                 participation: float = 1.0, nominal: float = 100.0,
                 rate_curve=None, vol_surface=None, heston=None):
        super().__init__()
        self.S, self.T, self.r = S, T, r
        self.participation = participation
        self.nominal = nominal

        # Call de strike quasi-nul -> réplique S_T à maturité (avec actualisation BSM)
        self._add_leg(Option(S, 1e-3, T, r, "call",
                              vol_surface=vol_surface, heston=heston),
                      participation)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "TrackerCertificate", "S": self.S, "T": self.T,
                "participation": self.participation, "sspa": 1300}
