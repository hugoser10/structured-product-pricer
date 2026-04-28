from typing import Dict, Any
from pricer.products.base.equity_product import EquityProduct
from pricer.products.equity.option import Option


class CallSpread(EquityProduct):
    """
    Call spread : achat d'un call K1 et vente d'un call K2, avec K1 < K2.

    Stratégie haussière à coût limité : on réduit la prime payée en vendant
    le call K2, mais on plafonne le gain au-delà de K2. Le profit maximal
    est atteint si le spot finit au-dessus de K2, la perte maximale est
    limitée à la prime nette débitée.
    """

    def __init__(self, S, K1, K2, T, r, sigma=0.2, q=0.0, vol_surface=None, heston=None):
        super().__init__()
        self.S, self.K1, self.K2, self.T, self.r = S, K1, K2, T, r
        self._add_leg(Option(S, K1, T, r, "call", sigma, q, vol_surface, heston), +1.0)
        self._add_leg(Option(S, K2, T, r, "call", sigma, q, vol_surface, heston), -1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "CallSpread", "S": self.S, "K1": self.K1, "K2": self.K2, "T": self.T}