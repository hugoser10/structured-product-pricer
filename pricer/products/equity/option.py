import numpy as np
from typing import Dict, Any
from pricer.products.base.atomic_equity_product import AtomicEquityProduct
from pricer.models.volatility.black_scholes import bsm_price, bsm_greeks


class Option(AtomicEquityProduct):
    """
    Option européenne vanille (call ou put).

    Trois modes de pricing disponibles, par ordre de priorité :
    - Heston (Fourier) si un modèle heston est fourni — le put est déduit
      du call par parité call-put.
    - BSM avec surface de volatilité si vol_surface est fournie — la vol
      est interpolée sur la surface au strike et à la maturité du produit.
    - BSM flat si seulement sigma est fourni (comportement par défaut).

    Note : les Greeks sont toujours calculés en BSM, même si le pricing
    utilise Heston.
    """

    def __init__(self, S: float, K: float, T: float, r: float,
                 option_type: str = "call",
                 sigma: float = 0.2, q: float = 0.0,
                 vol_surface=None, heston=None):
        self.S, self.K, self.T, self.r = S, K, T, r
        self.option_type = option_type.lower()
        self._sigma = sigma
        self.q = q
        self.vol_surface = vol_surface
        self.heston = heston

    @property
    def sigma(self) -> float:
        """Retourne la vol interpolée sur la surface si disponible, sinon la vol flat."""
        if self.vol_surface is not None:
            return self.vol_surface.get_vol(self.K, self.T)
        return self._sigma

    def price(self, **kwargs) -> float:
        """
        Calcule le prix de l'option selon le modèle disponible.
        À maturité (T <= 0), retourne directement le payoff intrinsèque.
        """
        if self.T <= 0:
            return float(max(self.S - self.K, 0) if self.option_type == "call"
                         else max(self.K - self.S, 0))
        if self.heston is not None:
            call_p = self.heston.price_call_fourier(self.S, self.K, self.T, self.r)
            if self.option_type == "call":
                return call_p
            # Le put est déduit du call par parité call-put
            return call_p - self.S * np.exp(-self.q * self.T) + self.K * np.exp(-self.r * self.T)
        return bsm_price(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.q)

    def _compute_greeks(self, **kwargs) -> Dict[str, float]:
        """Calcule les Greeks par formule fermée BSM."""
        return bsm_greeks(self.S, self.K, self.T, self.r, self.sigma, self.option_type, self.q)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": f"Option({self.option_type})", "S": self.S,
                "K": self.K, "T": self.T, "r": self.r, "sigma": self.sigma}