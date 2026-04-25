from compositeProduct import CompositeProduct
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

class RateProduct(CompositeProduct):
    """
    Pour les produits de taux composés.
    Garantit que greeks() contient toujours 'dv01' (hérité des legs ZCB/Caplet).
    """

    def greeks(self, **kwargs) -> Dict[str, float]:
        g = super().greeks(**kwargs)
        g.setdefault("dv01", 0.0)
        return g

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass