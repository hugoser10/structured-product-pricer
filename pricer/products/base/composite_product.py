from abc import abstractmethod
from typing import List, Tuple, Dict, Any
from pricer.products.base.product import Product


class CompositeProduct(Product):
    """
    Classe de base pour les produits construits par combinaison de sous-produits.

    Chaque sous-produit (jambe) est associé à un poids : +1 pour une position
    longue, -1 pour une position courte, ou n'importe quel autre coefficient.
    Le prix et les Greeks du produit sont obtenus en faisant la somme pondérée
    des prix et Greeks de chaque jambe.

    Exemple : un CallSpread = +1 call K1 + (-1) call K2
    """

    def __init__(self):
        self._legs: List[Tuple[Product, float]] = []

    def _add_leg(self, product: Product, weight: float = 1.0):
        """Ajoute une jambe au produit avec son poids (défaut +1)."""
        self._legs.append((product, weight))

    def price(self, **kwargs) -> float:
        """Retourne la somme pondérée des prix de chaque jambe."""
        return sum(w * leg.price(**kwargs) for leg, w in self._legs)

    def greeks(self, **kwargs) -> Dict[str, float]:
        """Retourne la somme pondérée des Greeks de chaque jambe."""
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
        """Retourne les paramètres du produit sous forme de dictionnaire."""
        ...