from abc import abstractmethod
from typing import Dict, Any
from pricer.products.base.composite_product import CompositeProduct

class EquityProduct(CompositeProduct):
    """compo famille equity. greeks() garantit delta, gamma, vega, theta, rho."""

    _EQUITY_GREEKS = ("delta", "gamma", "vega", "theta", "rho")

    def greeks(self, **kwargs) -> Dict[str, float]:
        g = super().greeks(**kwargs)
        for key in self._EQUITY_GREEKS:
            g.setdefault(key, 0.0)
        return g

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]: ...
