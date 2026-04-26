import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional

class Product(ABC):
    """
    Interface commune à tous les produits financiers.

    Impose trois méthodes à toutes les sous-classes :
        price(**kwargs)   - float : valeur de marché
        greeks(**kwargs)  - dict[str,float] :sensibilités aux facteurs de risque
        to_dict()         - dict[str,Any] : paramètres pour affichage/sérialisation
    """

    @abstractmethod
    def price(self, **kwargs) -> float:
        pass

    @abstractmethod
    def greeks(self, **kwargs) -> Dict[str, float]:
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"