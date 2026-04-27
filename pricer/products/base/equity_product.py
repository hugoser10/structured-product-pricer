from abc import abstractmethod
from typing import Dict, Any
from pricer.products.base.composite_product import CompositeProduct


class EquityProduct(CompositeProduct):
    """
    Classe de base pour les produits equity composites (CallSpread, Straddle...).

    Hérite de CompositeProduct : le prix et les Greeks sont calculés en sommant
    ceux de chaque jambe. Ajoute simplement la garantie que les cinq clés
    delta, gamma, vega, theta, rho sont toujours présentes dans le résultat
    de greeks(), avec 0.0 par défaut si une jambe ne les retourne pas.
    """

    _EQUITY_GREEKS = ("delta", "gamma", "vega", "theta", "rho")

    def greeks(self, **kwargs) -> Dict[str, float]:
        g = super().greeks(**kwargs)
        for key in self._EQUITY_GREEKS:
            g.setdefault(key, 0.0)
        return g

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Retourne les paramètres du produit sous forme de dictionnaire."""
        ...