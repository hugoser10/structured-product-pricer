import numpy as np
from pricer.models.rates.stochastic_rate_model import StochasticRateModel
from pricer.models.rates.vasicek_model import VasicekModel


class HullWhiteModel(StochasticRateModel):
    """dr = (theta(t) - a*r)*dt + sigma*dW. theta(t) calibré exactement sur la courbe initiale."""

    def __init__(self, a: float = 0.1, sigma: float = 0.01, rate_curve=None):
        self.a, self.sigma, self.rate_curve = a, sigma, rate_curve

    def _theta(self, t: float) -> float:
        # theta(t) = a*f(0,t) + df(0,t)/dt + sigma^2/(2a)*(1 - exp(-2at))
        if self.rate_curve is None:
            return self.a * 0.03
        dt = 0.001
        f0 = self.rate_curve.forward_rate(max(t - dt, 1e-6), t + dt)
        df = (self.rate_curve.forward_rate(t, t + 2*dt)
              - self.rate_curve.forward_rate(max(t - dt, 1e-6), t + dt)) / dt
        return self.a * f0 + df + (self.sigma**2 / (2 * self.a)) * (1 - np.exp(-2 * self.a * t))

    def zero_bond_price(self, r0: float, T: float) -> float:
        if self.rate_curve is None:
            return VasicekModel(self.a, 0.03, self.sigma).zero_bond_price(r0, T)
        a, s = self.a, self.sigma
        sigma_t = s * (1 - np.exp(-a * T)) / a
        P0T = self.rate_curve.discount_factor(T)
        f0T = self.rate_curve.forward_rate(max(T - 0.001, 1e-6), T + 0.001)
        log_A = (np.log(P0T) + sigma_t * f0T
                 - (s**2 / (4 * a)) * (np.exp(-2*a*T) - 1) * (np.exp(-a*T) - 1)**2 / a**2)
        return np.exp(log_A - sigma_t * r0)

    def simulate_paths(self, r0: float, T: float, n_steps: int,
                       n_simulations: int, seed: int = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = r0
        Z = np.random.normal(0, 1, (n_simulations, n_steps))
        for i in range(n_steps):
            r = paths[:, i]
            theta_t = self._theta(times[i])
            paths[:, i + 1] = r + (theta_t - self.a * r) * dt + self.sigma * np.sqrt(dt) * Z[:, i]
        return paths
