from abc import abstractmethod
from typing import Dict, Any
from pricer.products.base.product import Product


class PathDependentProduct(Product):
    """
    Classe de base pour les produits dont le pricing nécessite une simulation Monte Carlo
    (options barrières, autocalls...).

    Contrairement aux produits analytiques, il n'existe pas de formule fermée pour
    ces produits. Le prix est donc estimé par simulation, et les Greeks delta et gamma
    sont calculés par différences finies : on bumpe le spot de +/- 1%, on reprice,
    et on en déduit les dérivées numériquement.

    Les sous-classes doivent implémenter _get_spot() et _set_spot() pour permettre
    ce bumping, ainsi que price() pour la simulation elle-même.

    Note : seuls delta et gamma sont calculés. Vega, theta et rho nécessiteraient
    des simulations supplémentaires et ne sont pas implémentés ici.
    """

    @abstractmethod
    def _get_spot(self) -> float:
        """Retourne la valeur actuelle du spot."""
        ...

    @abstractmethod
    def _set_spot(self, value: float):
        """Met à jour la valeur du spot (utilisé pour le bumping des Greeks)."""
        ...

    def greeks(self, n_simulations: int = 5000, seed: int = 42, **kwargs) -> Dict[str, float]:
        """
        Calcule le prix, le delta et le gamma par différences finies.

        Le spot est bumpé de +/- 1% pour estimer les dérivées. Le seed est fixé
        pour que les trois simulations utilisent les mêmes tirages aléatoires
        et réduire le bruit de Monte Carlo.
        """
        eps = self._get_spot() * 0.01
        p = self.price(n_simulations=n_simulations, seed=seed, **kwargs)

        self._set_spot(self._get_spot() + eps)
        pu = self.price(n_simulations=n_simulations, seed=seed, **kwargs)

        self._set_spot(self._get_spot() - 2 * eps)
        pd_ = self.price(n_simulations=n_simulations, seed=seed, **kwargs)

        self._set_spot(self._get_spot() + eps)
        return {
            "price": p,
            "delta": (pu - pd_) / (2 * eps),
            "gamma": (pu - 2 * p + pd_) / eps ** 2,
        }

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Retourne les paramètres du produit sous forme de dictionnaire."""
        ...