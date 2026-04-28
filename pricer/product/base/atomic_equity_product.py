from abc import abstractmethod
from typing import Dict
from pricer.products.base.product import Product

class AtomicEquityProduct(Product):
    """Produit equity atomique (formule fermee). greeks() garantit delta, gamma, vega, theta, rho."""

    _EQUITY_GREEKS = ("delta", "gamma", "vega", "theta", "rho")

    def greeks(self, **kwargs) -> Dict[str, float]:
        g = self._compute_greeks(**kwargs)
        g.setdefault("price", self.price(**kwargs))
        for key in self._EQUITY_GREEKS:
            g.setdefault(key, 0.0)
        return g

    @abstractmethod
    def _compute_greeks(self, **kwargs) -> Dict[str, float]: ...
