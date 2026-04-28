import numpy as np
from typing import Dict, Any
from pricer.products.base.path_dependent_product import PathDependentProduct


class BarrierOption(PathDependentProduct):
    """
    Option à barrière européenne, pricée par simulation Monte Carlo.

    Quatre types de barrières disponibles via barrier_type :
    - down-and-out : l'option est annulée si le spot descend sous la barrière.
    - down-and-in  : l'option est activée si le spot descend sous la barrière.
    - up-and-out   : l'option est annulée si le spot monte au-dessus de la barrière.
    - up-and-in    : l'option est activée si le spot monte au-dessus de la barrière.

    En cas d'annulation (out) ou de non-activation (in), l'investisseur reçoit
    le rebate (0 par défaut). Le pricing utilise Heston si un modèle est fourni,
    sinon un GBM log-normal standard.

    Note : la barrière est surveillée en continu sur les pas de simulation.
    Plus n_steps est élevé, plus la surveillance est précise.
    """

    def __init__(self, S, K, T, r, barrier, barrier_type="down-and-out",
                 option_type="call", rebate=0.0, sigma=0.2, q=0.0,
                 heston=None, vol_surface=None):
        self.S, self.K, self.T, self.r = S, K, T, r
        self.barrier = barrier
        self.barrier_type = barrier_type.lower()
        self.option_type = option_type.lower()
        self.rebate = rebate
        self._sigma = sigma
        self.q = q
        self.heston = heston
        self.vol_surface = vol_surface

    @property
    def sigma(self) -> float:
        """Retourne la vol interpolée sur la surface si disponible, sinon la vol flat."""
        if self.vol_surface is not None:
            return self.vol_surface.get_vol(self.K, self.T)
        return self._sigma

    def _get_spot(self): return self.S
    def _set_spot(self, v): self.S = v

    def price(self, n_simulations=10000, n_steps=252, seed=42, **kwargs) -> float:
        """
        Calcule le prix par Monte Carlo.

        Simule n_simulations trajectoires, détecte si la barrière est touchée
        sur chaque trajectoire, puis calcule le payoff moyen actualisé.

        Args:
            n_simulations : nombre de trajectoires simulées (défaut 10 000).
            n_steps : nombre de pas de temps, détermine la précision de la
                      surveillance de la barrière (défaut 252, soit quotidien sur 1 an).
            seed : graine aléatoire pour la reproductibilité.
        """
        if seed is not None:
            np.random.seed(seed)
        if self.heston is not None:
            S_paths, _ = self.heston.simulate_paths(self.S, self.r, self.T,
                                                     n_steps, n_simulations, seed)
        else:
            # GBM : log-rendement (r - q - σ²/2)dt + σ√dt Z
            dt = self.T / n_steps
            Z = np.random.normal(0, 1, (n_simulations, n_steps))
            lr = (self.r - self.q - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z
            S_paths = np.hstack([np.full((n_simulations, 1), self.S),
                                  self.S * np.exp(np.cumsum(lr, axis=1))])

        ST = S_paths[:, -1]
        H = self.barrier
        hit = (np.any(S_paths <= H, axis=1) if "down" in self.barrier_type
               else np.any(S_paths >= H, axis=1))
        van = (np.maximum(ST - self.K, 0) if self.option_type == "call"
               else np.maximum(self.K - ST, 0))
        # KO : payoff vanille si barrière non touchée, rebate sinon
        # KI : payoff vanille si barrière touchée, rebate sinon
        pf = (np.where(hit, self.rebate, van) if "out" in self.barrier_type
              else np.where(hit, van, self.rebate))
        return float(np.mean(pf) * np.exp(-self.r * self.T))

    def to_dict(self) -> Dict[str, Any]:
        return {"type": f"BarrierOption({self.barrier_type})",
                "S": self.S, "K": self.K, "T": self.T,
                "barrier": self.barrier, "option_type": self.option_type}