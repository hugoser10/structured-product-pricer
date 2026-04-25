import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

class CompositeProduct(Product):
    """
    Produit dont price() et greeks() sont la somme pondérée des composantes.

    Les sous-classes déclarent leurs briques via _add_leg(produit, poids).
    to_dict() reste abstraite : chaque sous-classe décrit ses paramètres métier.

        price()  = Σ  w_i x leg_i.price()
        greeks() = Σ  w_i x leg_i.greeks()   (clé par clé)
    """

    def __init__(self):
        self._legs: List[Tuple[Product, float]] = []

    def _add_leg(self, product: Product, weight: float = 1.0):
        """Ajoute une composante pondérée."""
        self._legs.append((product, weight))

    def price(self, **kwargs) -> float:
        return sum(w * leg.price(**kwargs) for leg, w in self._legs)

    def greeks(self, **kwargs) -> Dict[str, float]:
        """Agrège les greeks de chaque composante pondérés par leur poids."""
        result: Dict[str, float] = {}
        for leg, w in self._legs:
            for key, val in leg.greeks(**kwargs).items():
                if key == "price":
                    continue
                result[key] = result.get(key, 0.0) + w * val
        result["price"] = self.price(**kwargs)
        return result

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass