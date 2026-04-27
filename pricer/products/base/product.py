from abc import ABC, abstractmethod
from typing import Dict, Any


class Product(ABC):
    """
    Interface commune à tous les produits financiers du pricer.

    Toute classe concrète (Option, CallSpread, Autocall...) doit implémenter
    les trois méthodes suivantes :

    - price() : retourne le prix du produit
    - greeks() : retourne un dictionnaire de sensibilités (delta, vega, dv01...)
    - to_dict() : retourne les paramètres du produit, utilisé pour l'affichage
                  et la sérialisation (inventaire, portfolio)
    """

    @abstractmethod
    def price(self, **kwargs) -> float:
        """Retourne le prix du produit."""
        ...

    @abstractmethod
    def greeks(self, **kwargs) -> Dict[str, float]:
        """Retourne les sensibilités du produit sous forme de dictionnaire."""
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Retourne les paramètres du produit sous forme de dictionnaire."""
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"