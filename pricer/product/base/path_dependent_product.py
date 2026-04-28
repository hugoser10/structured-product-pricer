from abc import abstractmethod
from typing import Dict, Any
from pricer.products.base.product import Product


class PathDependentProduct(Product):
    """Pricing Monte Carlo. delta/gamma par différences finies sur le spot."""

    @abstractmethod
    def _get_spot(self) -> float: ...

    @abstractmethod
    def _set_spot(self, value: float): ...

    def greeks(self, n_simulations: int = 5000, seed: int = 42, **kwargs) -> Dict[str, float]:
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
    def to_dict(self) -> Dict[str, Any]: ...
