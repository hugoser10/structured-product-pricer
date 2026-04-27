from typing import Dict, Any
from pricer.products.base.equity_product import EquityProduct
from pricer.products.equity.option import Option


class Straddle(EquityProduct):
    """
    Straddle : achat d'un call et d'un put au même strike K et à la même maturité.

    Stratégie qui parie sur un fort mouvement du sous-jacent dans un sens ou
    dans l'autre. Profitable si le spot s'éloigne suffisamment de K pour
    couvrir la double prime payée. Le point mort est atteint à K +/- prime totale.

    Le delta est proche de zéro si K est at-the-money, ce qui en fait une
    position quasi-pure sur la volatilité.
    """

    def __init__(self, S, K, T, r, sigma=0.2, q=0.0, vol_surface=None, heston=None):
        super().__init__()
        self.S, self.K, self.T, self.r = S, K, T, r
        self._add_leg(Option(S, K, T, r, "call", sigma, q, vol_surface, heston), +1.0)
        self._add_leg(Option(S, K, T, r, "put",  sigma, q, vol_surface, heston), +1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Straddle", "S": self.S, "K": self.K, "T": self.T}