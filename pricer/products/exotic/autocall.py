import numpy as np
from scipy.optimize import brentq
from typing import Dict, Any
from pricer.products.base.path_dependent_product import PathDependentProduct


class Autocall(PathDependentProduct):
    """
    Autocall à barrières observées à dates fixes, pricé par simulation Monte Carlo.

    À chaque date d'observation, trois scénarios possibles :
    - Si S(t_i) >= barrier_call[i] * strike : rappel anticipé, l'investisseur
      reçoit (1 + i * coupon) actualisé.
    - À maturité seulement, si pas de rappel et S(T) >= barrier_final * strike :
      remboursement du nominal (1.0).
    - À maturité seulement, si S(T) < barrier_final * strike : perte en capital,
      l'investisseur reçoit S(T) / strike.

    Trois modes de simulation disponibles :
    - GBM (par défaut) : trajectoires log-normales avec vol flat ou de surface.
    - Heston : trajectoires avec volatilité stochastique si heston est fourni.
    - Taux stochastiques : actualisation via Hull-White si hw_model est fourni,
      combinable avec GBM ou Heston.
    """

    def __init__(self, spot, strike, coupon, barrier_call,
                 barrier_final=0.5, maturity=5.0, n_obs=1,
                 r=0.03, sigma=0.2, q=0.0,
                 heston=None, vol_surface=None, hw_model=None):
        self.spot = spot
        self.strike = strike
        self.coupon = coupon
        self.barrier_call = barrier_call
        self.barrier_final = barrier_final
        self.maturity = maturity
        self.n_obs = n_obs
        self.r = r
        self._sigma = sigma
        self.q = q
        self.heston = heston
        self.vol_surface = vol_surface
        self.hw_model = hw_model

    @property
    def sigma(self) -> float:
        """Retourne la vol ATM de la surface si disponible, sinon la vol flat."""
        if self.vol_surface is not None:
            return self.vol_surface.get_atm_vol(self.maturity)
        return self._sigma

    def _get_spot(self): return self.spot
    def _set_spot(self, v): self.spot = v

    def price(self, n_simulations=20000, n_steps_per_obs=20, seed=42, **kwargs) -> float:
        """
        Calcule le prix par Monte Carlo.

        Pour chaque simulation, parcourt les dates d'observation dans l'ordre
        et applique la logique de rappel. Le prix est la moyenne des payoffs
        actualisés sur l'ensemble des simulations.

        Args:
            n_simulations : nombre de trajectoires simulées (défaut 20 000).
            n_steps_per_obs : nombre de pas de temps entre deux observations (défaut 20).
            seed : graine aléatoire pour la reproductibilité.
        """
        if seed is not None:
            np.random.seed(seed)
        total_obs = len(self.barrier_call)
        n_steps = total_obs * n_steps_per_obs
        T = self.maturity

        # --- Trajectoires du sous-jacent ---
        if self.heston is not None:
            S_paths, _ = self.heston.simulate_paths(self.spot, self.r, T,
                                                     n_steps, n_simulations, seed)
            oidx = [int((i+1) * n_steps_per_obs) for i in range(total_obs)]
            S_obs = S_paths[:, oidx]
        else:
            dt = T / n_steps
            Z = np.random.normal(0, 1, (n_simulations, n_steps))
            lr = (self.r - self.q - 0.5*self.sigma**2)*dt + self.sigma*np.sqrt(dt)*Z
            S_all = self.spot * np.exp(np.cumsum(lr, axis=1))
            oidx = [int((i+1)*n_steps_per_obs) - 1 for i in range(total_obs)]
            S_obs = S_all[:, oidx]

        # --- Facteurs d'actualisation ---
        dt_obs = T / total_obs
        if self.hw_model is not None:
            rc = self.hw_model.rate_curve
            r0 = rc.zero_rate(0.25) if rc else self.r
            rp = self.hw_model.simulate_paths(r0, T, n_steps, n_simulations, seed+1)
            dt_r = T / n_steps
            cr = np.cumsum(rp[:, :-1] * dt_r, axis=1)
            df_obs = np.exp(-cr[:, [int((i+1)*n_steps_per_obs)-1 for i in range(total_obs)]])
        else:
            df_obs = np.tile(np.exp(-self.r * np.arange(1, total_obs+1) * dt_obs),
                             (n_simulations, 1))

        # --- Boucle d'observation : rappel ou non ---
        payoffs = np.zeros(n_simulations)
        called = np.zeros(n_simulations, dtype=bool)

        for obs in range(total_obs):
            S_t = S_obs[:, obs]
            nb = obs + 1
            b_lvl = self.barrier_call.get(nb, 1.0) * self.strike
            df = df_obs[:, obs]
            active = ~called
            if obs < total_obs - 1:
                rec = active & (S_t >= b_lvl)
                payoffs[rec] = (1 + nb * self.coupon) * df[rec]
                called |= rec
            else:
                # Dernière observation : rappel, remboursement nominal, ou perte en capital
                c1 = active & (S_t >= b_lvl)
                c2 = active & ~c1 & (S_t >= self.barrier_final * self.strike)
                c3 = active & ~c1 & ~c2
                payoffs[c1] = (1 + nb * self.coupon) * df[c1]
                payoffs[c2] = 1.0 * df[c2]
                payoffs[c3] = (S_t[c3] / self.strike) * df[c3]

        return float(np.mean(payoffs))

    def optimal_coupon(self, target_price=1.0, n_simulations=10000, seed=42) -> float:
        """
        Calcule le coupon qui donne un prix cible, par défaut at par (1.0).
        Utilise la méthode de Brent pour trouver le zéro de price() - target_price.
        """
        def f(c):
            self.coupon = c
            return self.price(n_simulations=n_simulations, seed=seed) - target_price
        try:
            return brentq(f, 0.001, 0.5, xtol=1e-4)
        except Exception:
            return self.coupon

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Autocall", "spot": self.spot, "strike": self.strike,
                "coupon": self.coupon, "maturity": self.maturity,
                "barrier_final": self.barrier_final}