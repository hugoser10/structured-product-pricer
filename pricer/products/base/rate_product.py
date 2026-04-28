from abc import abstractmethod
from typing import Dict, Any
from pricer.products.base.composite_product import CompositeProduct


class RateProduct(CompositeProduct):
    """Compo de famille de taux. greeks() garantit 'dv01'."""

    def greeks(self, **kwargs) -> Dict[str, float]:
        g = super().greeks(**kwargs)
        g.setdefault("dv01", 0.0)
        return g

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]: ...
