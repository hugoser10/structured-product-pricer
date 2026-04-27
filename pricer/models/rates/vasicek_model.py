import numpy as np
from pricer.models.rates.stochastic_rate_model import StochasticRateModel


class VasicekModel(StochasticRateModel):
    """dr = a*(k - r)*dt + sigma*dW. Calibration OLS sur la discrétisation."""

    def __init__(self, a: float = 0.1, k: float = 0.03, sigma: float = 0.01):
        self.a, self.k, self.sigma = a, k, sigma

    def calibrate(self, rate_series: np.ndarray, dt: float = 1/252,
                  window_years: int = 5):
        # OLS sur r(t+dt) = alpha*r(t) + beta + eps  →  a = (1-alpha)/dt, k = beta/(a*dt)
        r = np.array(rate_series)
        n_recent = min(len(r), int(window_years * 252))
        r = r[-n_recent:]
        X = np.column_stack([r[:-1], np.ones(len(r) - 1)])
        y = r[1:]
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha, beta = coeffs
        self.a = max((1 - alpha) / dt, 1e-4)
        self.k = beta / (self.a * dt) if self.a > 1e-4 else float(np.mean(r))
        residuals = y - X @ coeffs
        self.sigma = np.std(residuals) / np.sqrt(dt)
        return self

    def zero_bond_price(self, r0: float, T: float) -> float:
        a, k, s = self.a, self.k, self.sigma
        sigma_b = s * (1 - np.exp(-a * T)) / a
        A = np.exp((sigma_b / s - T) * (k - s**2 / (2 * a**2)) - sigma_b**2 / (4 * a))
        return A * np.exp(-sigma_b / s * r0)

    def simulate_paths(self, r0: float, T: float, n_steps: int,
                       n_simulations: int, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        dt = T / n_steps
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = r0
        Z = np.random.normal(0, 1, (n_simulations, n_steps))
        for i in range(n_steps):
            r = paths[:, i]
            paths[:, i + 1] = r + self.a * (self.k - r) * dt + self.sigma * np.sqrt(dt) * Z[:, i]
        return paths
