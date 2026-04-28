from abc import abstractmethod
from typing import Dict
from pricer.products.base.product import Product


class AtomicEquityProduct(Product):
    """
    Classe de base pour les produits equity simples (Option, DigitalOption...).

    Un produit "atomique" est une brique de base qui calcule son prix directement
    par formule, sans se décomposer en sous-produits. C'est la différence avec
    EquityProduct (ex: CallSpread) qui combine plusieurs Options entre elles.

    greeks() s'assure que le dictionnaire retourné contient toujours les cinq
    clés delta, gamma, vega, theta, rho — avec 0.0 si elles ne sont pas
    calculées par _compute_greeks().
    """

    _EQUITY_GREEKS = ("delta", "gamma", "vega", "theta", "rho")

    def greeks(self, **kwargs) -> Dict[str, float]:
        g = self._compute_greeks(**kwargs)
        g.setdefault("price", self.price(**kwargs))
        for key in self._EQUITY_GREEKS:
            g.setdefault(key, 0.0)
        return g

    @abstractmethod
    def _compute_greeks(self, **kwargs) -> Dict[str, float]:
        """Calcule et retourne les Greeks du produit sous forme de dictionnaire."""
        ...