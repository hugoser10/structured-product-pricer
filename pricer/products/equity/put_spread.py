from typing import Dict, Any
from pricer.products.base.equity_product import EquityProduct
from pricer.products.equity.option import Option


class PutSpread(EquityProduct):
    """
    Put spread : achat d'un put K2 et vente d'un put K1, avec K1 < K2.

    Stratégie baissière à coût limité : on réduit la prime payée en vendant
    le put K1, mais on plafonne le gain en dessous de K1. Le profit maximal
    est atteint si le spot finit en dessous de K1, la perte maximale est
    limitée à la prime nette débitée.
    """

    def __init__(self, S, K1, K2, T, r, sigma=0.2, q=0.0, vol_surface=None, heston=None):
        super().__init__()
        self.S, self.K1, self.K2, self.T, self.r = S, K1, K2, T, r
        self._add_leg(Option(S, K2, T, r, "put", sigma, q, vol_surface, heston), +1.0)
        self._add_leg(Option(S, K1, T, r, "put", sigma, q, vol_surface, heston), -1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "PutSpread", "S": self.S, "K1": self.K1, "K2": self.K2, "T": self.T}