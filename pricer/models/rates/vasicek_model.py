import numpy as np
from pricer.models.rates.stochastic_rate_model import StochasticRateModel

class VasicekModel(StochasticRateModel):
    """
    Modèle de Vasicek : taux court à retour à la moyenne.

        dr(t) = a * (k - r(t)) * dt + sigma * dW(t)

    Paramètres :
        a  : vitesse de retour à la moyenne
        k  : taux d'équilibre long terme
        sigma : volatilité du taux court

    Le modèle admet des taux négatifs (processus gaussien).
    Formule fermée pour les obligations ZC disponible.

    Calibration : régression OLS
        r(t+Δt) = alpha * r(t) + beta + epsilon, où alpha = 1 - a*Δt, beta = a*k*Δt.
    """

    def __init__(self, a: float = 0.1, k: float = 0.03, sigma: float = 0.01):
        self.a = a
        self.k = k
        self.sigma = sigma

    def calibrate(self, rate_series: np.ndarray, dt: float = 1/252,
                  window_years: int = 5):
        """
        Calibration par OLS sur la série historique de taux courts.
        Résout la régression linéaire : r(t+dt) = alpha·r(t) + beta + espilon
        Utilise les window_years dernières années pour rester dans
        un régime de taux homogène.
        """
        r = np.array(rate_series)
        n_recent = min(len(r), int(window_years * 252))
        r = r[-n_recent:]

        # Régression par OLS
        X = np.column_stack([r[:-1], np.ones(len(r) - 1)])
        y = r[1:]
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        alpha, beta = coeffs

        # Récupération des paramètres structurels depuis alpha et beta
        self.a = max((1 - alpha) / dt, 1e-4)
        self.k = beta / (self.a * dt) if self.a > 1e-4 else float(np.mean(r))

        # sigma estimé via l'écart-type des résidus, annualisé
        residuals = y - X @ coeffs
        self.sigma = np.std(residuals) / np.sqrt(dt)
        return self

    def zero_bond_price(self, r0: float, T: float) -> float:
        """
        Prix analytique d'une obligation ZC de maturité T.
            B(T) = A(T) * exp(sigma_b/sigma * r0)
        """
        a, k, s = self.a, self.k, self.sigma

        # Sensibilité du ZC au taux court
        sigma_b = s * (1 - np.exp(-a * T)) / a

        # Facteur d'ajustement de la convexité
        A = np.exp((sigma_b / s - T) * (k - s**2 / (2 * a**2)) - sigma_b**2 / (4 * a))

        return A * np.exp(-sigma_b / s * r0)

    def simulate_paths(self, r0: float, T: float, n_steps: int,
                       n_simulations: int, seed: int = None) -> np.ndarray:
        """
        Simulation Euler-Maruyama :
            r(t+dt) = r(t) + a*(k - r(t))*dt + sigma*sqrt(dt)*Z,  Z ~ N(0,1)
        """
        if seed is not None:
            np.random.seed(seed)
        dt = T / n_steps

        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = r0

        # Chocs gaussiens pour toutes les simulations et tous les steps
        Z = np.random.normal(0, 1, (n_simulations, n_steps))
        for i in range(n_steps):
            r = paths[:, i]
            paths[:, i + 1] = r + self.a * (self.k - r) * dt + self.sigma * np.sqrt(dt) * Z[:, i]

        return paths