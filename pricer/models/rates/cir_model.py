import numpy as np
from scipy.optimize import differential_evolution
from pricer.models.rates.stochastic_rate_model import StochasticRateModel


class CIRModel(StochasticRateModel):
    """dr = a*(k - r)*dt + sigma*sqrt(r)*dW. MLE avec contrainte de Feller (2ak > sigma^2)."""

    def __init__(self, a: float = 0.1, k: float = 0.03, sigma: float = 0.05):
        self.a, self.k, self.sigma = a, k, sigma

    def calibrate(self, rate_series: np.ndarray, dt: float = 1/252,
                  window_years: int = 5):
        # MLE plutôt que OLS car la volatilité dépend de sqrt(r) (instabilité numérique sinon)
        r = np.array(rate_series)
        n_recent = min(len(r), int(window_years * 252))
        r = np.maximum(r[-n_recent:], 1e-6)
        r_t, r_t1 = r[:-1], r[1:]

        def neg_loglik(params):
            a, k, s = params
            if a <= 0 or k <= 0 or s <= 0:
                return 1e10
            if 2 * a * k <= s ** 2:                    # contrainte de Feller
                return 1e8
            mu = r_t + a * (k - r_t) * dt
            var = np.maximum(s ** 2 * r_t * dt, 1e-12)
            return 0.5 * float(np.sum(np.log(var) + (r_t1 - mu) ** 2 / var))

        bounds = [(0.01, 10.0), (1e-4, 0.5), (1e-4, 1.0)]
        res = differential_evolution(neg_loglik, bounds, seed=42, maxiter=300,
                                      tol=1e-8, polish=True)
        self.a, self.k, self.sigma = res.x
        return self

    def zero_bond_price(self, r0: float, T: float) -> float:
        # B(T) = Phi(T) * exp(-r0 * Gamma(T))
        a, k, s = self.a, self.k, self.sigma
        gamma = np.sqrt(a**2 + 2 * s**2)
        exp_gt = np.exp(gamma * T)
        denom = (gamma + a) * (exp_gt - 1) + 2 * gamma
        Gamma = 2 * (exp_gt - 1) / denom
        Phi_exp = (2 * gamma * np.exp((a + gamma) * T / 2) / denom) ** (2 * a * k / s**2)
        return Phi_exp * np.exp(-r0 * Gamma)

    def simulate_paths(self, r0: float, T: float, n_steps: int,
                       n_simulations: int, seed: int = None) -> np.ndarray:
        # Schéma Euler tronqué à zéro pour garantir r >= 0
        if seed is not None:
            np.random.seed(seed)
        dt = T / n_steps
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = r0
        Z = np.random.normal(0, 1, (n_simulations, n_steps))
        for i in range(n_steps):
            r = paths[:, i]
            paths[:, i + 1] = np.maximum(
                0, r + self.a * (self.k - r) * dt
                + self.sigma * np.sqrt(np.maximum(r, 0)) * np.sqrt(dt) * Z[:, i])
        return paths
