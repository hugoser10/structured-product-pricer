import numpy as np
from scipy.stats import norm
from typing import Dict, Any
from pricer.products.base.atomic_equity_product import AtomicEquityProduct
from pricer.models.volatility.black_scholes import bsm_d1_d2


class DigitalOption(AtomicEquityProduct):
    """
    Option digitale européenne, disponible en deux variantes :

    - cash-or-nothing : verse un montant fixe `payout` si l'option finit dans
      la monnaie (ITM), 0 sinon.
    - asset-or-nothing : verse la valeur du sous-jacent S_T si ITM, 0 sinon.

    Le pricing est analytique (formule fermée BSM). La volatilité peut être
    fournie directement via sigma, ou lue sur une surface de volatilité.

    Note : le modèle Heston n'est pas supporté pour ce produit.
    Note : theta et rho de la variante asset-or-nothing sont approximés à 0.
    """

    def __init__(self, S: float, K: float, T: float, r: float,
                 option_type: str = "call",
                 digital_type: str = "cash-or-nothing",
                 payout: float = 1.0,
                 sigma: float = 0.2, q: float = 0.0, vol_surface=None):
        self.S, self.K, self.T, self.r = S, K, T, r
        self.option_type = option_type.lower()
        self.digital_type = digital_type.lower()
        self.payout = payout
        self._sigma = sigma
        self.q = q
        self.vol_surface = vol_surface

    @property
    def sigma(self) -> float:
        """Retourne la vol de la surface si disponible, sinon la vol flat."""
        if self.vol_surface is not None:
            return self.vol_surface.get_vol(self.K, self.T)
        return self._sigma

    def price(self, **kwargs) -> float:
        """
        Calcule le prix par formule fermée BSM.
        À maturité (T <= 0), retourne directement le payoff.
        """
        if self.T <= 0:
            itm = (self.S > self.K) if self.option_type == "call" else (self.S < self.K)
            if not itm:
                return 0.0
            return self.payout if self.digital_type == "cash-or-nothing" else self.S
        d1, d2 = bsm_d1_d2(self.S, self.K, self.T, self.r, self.sigma, self.q)
        eq, dk = np.exp(-self.q * self.T), np.exp(-self.r * self.T)
        if self.digital_type == "cash-or-nothing":
            n = norm.cdf(d2) if self.option_type == "call" else norm.cdf(-d2)
            return float(self.payout * dk * n)
        # asset-or-nothing
        n = norm.cdf(d1) if self.option_type == "call" else norm.cdf(-d1)
        return float(self.S * eq * n)

    def _compute_greeks(self, **kwargs) -> Dict[str, float]:
        """
        Calcule les Greeks par dérivation analytique de la formule fermée.
        Retourne un dictionnaire vide si T <= 0 (à maturité).
        """
        if self.T <= 0:
            return {}
        d1, d2 = bsm_d1_d2(self.S, self.K, self.T, self.r, self.sigma, self.q)
        s, sq = self.sigma, np.sqrt(self.T)
        eq, dk = np.exp(-self.q * self.T), np.exp(-self.r * self.T)
        denom = self.S * s * sq
        sign = 1.0 if self.option_type == "call" else -1.0

        if self.digital_type == "cash-or-nothing":
            delta = sign * self.payout * dk * norm.pdf(d2) / denom
            gamma = -sign * self.payout * dk * norm.pdf(d2) * d1 / (denom * denom / self.S)
            vega = -sign * self.payout * dk * norm.pdf(d2) * d1 / (s * 100)
            theta = (self.r * self.price()
                     - sign * self.payout * dk * norm.pdf(d2)
                     * (np.log(self.S / self.K) / (2 * s * sq**3)
                        + (self.r - self.q) / (s * sq))) / 365
            rho = sign * self.payout * self.T * dk * norm.pdf(d2) / (denom * s * 100)
        else:
            delta = eq * (sign * norm.cdf(sign * d1) + norm.pdf(d1) / denom)
            gamma = sign * eq * norm.pdf(d1) * (1 / denom) * (1 - d1 / (s * sq)) / self.S
            vega = sign * self.S * eq * norm.pdf(d1) * sq * d2 / (s * 100)
            theta = 0.0  # approximé à 0 pour asset-or-nothing
            rho = 0.0    # approximé à 0 pour asset-or-nothing
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

    def to_dict(self) -> Dict[str, Any]:
        return {"type": f"Digital({self.digital_type},{self.option_type})",
                "S": self.S, "K": self.K, "T": self.T, "payout": self.payout}