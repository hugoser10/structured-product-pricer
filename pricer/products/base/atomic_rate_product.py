from abc import abstractmethod
from typing import Dict
from pricer.products.base.product import Product


class AtomicRateProduct(Product):
    """Produit de taux atomique. greeks() garantit 'price' et 'dv01'."""

    def greeks(self, **kwargs) -> Dict[str, float]:
        g = self._compute_greeks(**kwargs)
        g.setdefault("price", self.price(**kwargs))
        g.setdefault("dv01", 0.0)
        return g

    @abstractmethod
    def _compute_greeks(self, **kwargs) -> Dict[str, float]: ...
