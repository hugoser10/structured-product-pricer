from typing import Dict, Any
from pricer.products.base.equity_product import EquityProduct
from pricer.products.equity.option import Option


class Strip(EquityProduct):
    """
    Strip : achat de 1 call et 2 puts au même strike K.

    Variante asymétrique du straddle avec un biais baissier : on mise sur
    un fort mouvement du sous-jacent, mais on double la mise à la baisse.
    Profitable si le spot descend fortement en dessous de K, ou monte
    suffisamment pour couvrir la triple prime payée. Le gain à la baisse
    est deux fois plus rapide qu'à la hausse.
    """

    def __init__(self, S, K, T, r, sigma=0.2, q=0.0, vol_surface=None, heston=None):
        super().__init__()
        self.S, self.K, self.T, self.r = S, K, T, r
        self._add_leg(Option(S, K, T, r, "call", sigma, q, vol_surface, heston), +1.0)
        self._add_leg(Option(S, K, T, r, "put",  sigma, q, vol_surface, heston), +2.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Strip", "S": self.S, "K": self.K, "T": self.T}