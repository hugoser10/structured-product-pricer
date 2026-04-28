from typing import Dict, Any
from pricer.products.base.equity_product import EquityProduct
from pricer.products.equity.option import Option


class Butterfly(EquityProduct):
    """
    Butterfly spread : +1 call K1, -2 call K2, +1 call K3, avec K1 < K2 < K3.

    Stratégie qui parie sur une faible volatilité du sous-jacent : elle est
    profitable si le spot reste proche de K2 à maturité. Le coût net est limité
    (prime débitée) et le gain maximal est atteint exactement en K2.

    K2 est généralement choisi proche du spot (at-the-money), et K1, K3
    sont placés symétriquement de part et d'autre.
    """

    def __init__(self, S, K1, K2, K3, T, r, sigma=0.2, q=0.0, vol_surface=None, heston=None):
        super().__init__()
        self.S, self.K1, self.K2, self.K3, self.T, self.r = S, K1, K2, K3, T, r
        self._add_leg(Option(S, K1, T, r, "call", sigma, q, vol_surface, heston), +1.0)
        self._add_leg(Option(S, K2, T, r, "call", sigma, q, vol_surface, heston), -2.0)
        self._add_leg(Option(S, K3, T, r, "call", sigma, q, vol_surface, heston), +1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Butterfly", "S": self.S, "K1": self.K1,
                "K2": self.K2, "K3": self.K3, "T": self.T}