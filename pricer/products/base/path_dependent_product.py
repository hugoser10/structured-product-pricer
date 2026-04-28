from abc import abstractmethod
from typing import Dict, Any
from pricer.products.base.product import Product


class PathDependentProduct(Product):
    """
    Pricing par Monte Carlo. Greeks par différences finies sur le spot
    (delta, gamma) et, si les attributs correspondants sont présents, sur
    la volatilité (vega), la maturité (theta) et le taux (rho).
    """

    _PATH_GREEKS = ("delta", "gamma", "vega", "theta", "rho")

    @abstractmethod
    def _get_spot(self) -> float: ...

    @abstractmethod
    def _set_spot(self, value: float): ...

    def _get_sigma(self):
        return getattr(self, "_sigma", None)

    def _set_sigma(self, value):
        if hasattr(self, "_sigma"):
            self._sigma = value

    def _get_T(self):
        if hasattr(self, "T"):
            return self.T
        if hasattr(self, "maturity"):
            return self.maturity
        return None

    def _set_T(self, value):
        if hasattr(self, "T"):
            self.T = value
        elif hasattr(self, "maturity"):
            self.maturity = value

    def _get_r(self):
        return getattr(self, "r", None)

    def _set_r(self, value):
        if hasattr(self, "r"):
            self.r = value

    def greeks(self, n_simulations: int = 5000, seed: int = 42, **kwargs) -> Dict[str, float]:
        # par difference finie avec seed commun
        def _price():
            return self.price(n_simulations=n_simulations, seed=seed, **kwargs)

        #  Prix
        p = _price()

        #   Delta / Gamma 
        S0 = self._get_spot()
        eps_S = max(abs(S0) * 0.01, 1e-8)
        self._set_spot(S0 + eps_S);  pu = _price()
        self._set_spot(S0 - eps_S);  pd_ = _price()
        self._set_spot(S0)
        delta = (pu - pd_) / (2 * eps_S)
        gamma = (pu - 2 * p + pd_) / (eps_S ** 2)

        # Vega (bump de _sigma)
        vega = 0.0
        sigma0 = self._get_sigma()
        if sigma0 is not None:
            eps_sig = 0.01
            self._set_sigma(sigma0 + eps_sig);  pu_v = _price()
            self._set_sigma(max(sigma0 - eps_sig, 1e-6));  pd_v = _price()
            self._set_sigma(sigma0)
            vega = (pu_v - pd_v) / (2 * eps_sig) / 100.0

        # Theta (forward bump : ce qui se passe demain si rien d'autre ne bouge)
        theta = 0.0
        T0 = self._get_T()
        if T0 is not None and T0 > 1 / 365:
            self._set_T(T0 - 1 / 365);  p_tom = _price()
            self._set_T(T0)
            theta = p_tom - p  # P&L par jour ; négatif si décay

        # Rho (bump du taux)
        rho = 0.0
        r0 = self._get_r()
        if r0 is not None:
            eps_r = 1e-4
            self._set_r(r0 + eps_r);  pu_r = _price()
            self._set_r(r0 - eps_r);  pd_r = _price()
            self._set_r(r0)
            rho = (pu_r - pd_r) / (2 * eps_r) / 100.0

        return {
            "price": p,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]: ...
