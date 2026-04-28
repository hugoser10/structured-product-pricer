from typing import Dict, Any
from pricer.products.base.equity_product import EquityProduct
from pricer.products.equity.option import Option


class Strangle(EquityProduct):
    """
    Strangle : achat d'un call K2 et d'un put K1, avec K1 < K2.

    Même logique que le straddle — pari sur un fort mouvement du sous-jacent —
    mais les deux strikes sont out-of-the-money, ce qui rend la prime totale
    moins chère. En contrepartie, le spot doit s'éloigner davantage pour
    que la stratégie soit profitable : les points morts sont K1 - prime
    et K2 + prime.
    """

    def __init__(self, S, K1, K2, T, r, sigma=0.2, q=0.0, vol_surface=None, heston=None):
        super().__init__()
        self.S, self.K1, self.K2, self.T, self.r = S, K1, K2, T, r
        self._add_leg(Option(S, K2, T, r, "call", sigma, q, vol_surface, heston), +1.0)
        self._add_leg(Option(S, K1, T, r, "put",  sigma, q, vol_surface, heston), +1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Strangle", "S": self.S, "K1": self.K1, "K2": self.K2, "T": self.T}