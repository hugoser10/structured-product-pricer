import numpy as np
from scipy.stats import norm
from typing import Dict, Any
from pricer.products.base.path_dependent_product import PathDependentProduct


class OneTouchOption(PathDependentProduct):
    """
    Option one-touch ou no-touch, pricée par formule fermée (Reiner-Rubinstein).

    - One-touch : verse le payout si le spot touche la barrière H à un moment
      quelconque avant maturité, 0 sinon.
    - No-touch : verse le payout si le spot ne touche jamais la barrière H
      avant maturité, 0 sinon. C'est le complément du one-touch.

    La barrière peut être haute (H > S) ou basse (H < S), la formule s'adapte
    automatiquement. Le pricing est analytique sous GBM avec barrière surveillée
    en continu.

    Note : les Greeks sont calculés par différences finies héritées de
    PathDependentProduct. Seuls delta et gamma sont disponibles.
    """

    def __init__(self, S, H, T, r, touch_type="one-touch",
                 payout=1.0, sigma=0.2, q=0.0, vol_surface=None):
        self.S, self.H, self.T, self.r = S, H, T, r
        self.touch_type = touch_type.lower()
        self.payout = payout
        self._sigma = sigma
        self.q = q
        self.vol_surface = vol_surface

    @property
    def sigma(self) -> float:
        """Retourne la vol interpolée sur la surface si disponible, sinon la vol flat."""
        if self.vol_surface is not None:
            return self.vol_surface.get_vol(self.H, self.T)
        return self._sigma

    def _get_spot(self): return self.S
    def _set_spot(self, v): self.S = v

    def _one_touch_price(self) -> float:
        """
        Calcule le prix du one-touch par la formule de Reiner-Rubinstein.
        Les paramètres alpha et beta dépendent du taux, du dividende et de la vol.
        La formule diffère selon que la barrière est haute ou basse.
        """
        s, sq = self.sigma, np.sqrt(self.T)
        alpha = (self.r - self.q) / s**2 - 0.5
        beta = np.sqrt(alpha**2 + 2 * self.r / s**2)
        ratio = self.H / self.S
        z1 = (np.log(ratio) + beta * s**2 * self.T) / (s * sq)
        z2 = z1 - 2 * beta * s * sq
        if self.H < self.S:
            return float(self.payout * (ratio**(-alpha-beta) * norm.cdf(-z1)
                                        + ratio**(-alpha+beta) * norm.cdf(-z2)))
        return float(self.payout * (ratio**(-alpha-beta) * norm.cdf(z1)
                                    + ratio**(-alpha+beta) * norm.cdf(z2)))

    def price(self, **kwargs) -> float:
        """
        Retourne le prix du one-touch ou du no-touch.
        À maturité (T <= 0), retourne directement le payoff selon que la
        barrière est atteinte ou non. Le no-touch est le complément du
        one-touch : payout * e^(-rT) - prix_one_touch.
        """
        if self.T <= 0:
            hit = (self.S <= self.H) if self.H < self.S else (self.S >= self.H)
            val = self.payout if hit else 0.0
            return val if self.touch_type == "one-touch" else (self.payout - val)
        ot = self._one_touch_price()
        if self.touch_type == "one-touch":
            return ot
        return self.payout * np.exp(-self.r * self.T) - ot

    def to_dict(self) -> Dict[str, Any]:
        return {"type": f"OneTouchOption({self.touch_type})",
                "S": self.S, "H": self.H, "T": self.T}