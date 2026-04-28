import numpy as np
from pricer.models.rates.stochastic_rate_model import StochasticRateModel
from scipy.optimize import differential_evolution

class CIRModel(StochasticRateModel):
    """
    Modèle CIR (Cox-Ingersoll-Ross, 1985) — taux court à variance stochastique.

        dr(t) = a * (k - r(t)) * dt + sigma * sqrt(r(t)) * dW(t)

    Avantage par rapport à Vasicek : les taux restent positifs sous la
    condition de Feller : 2*a*k > sigma^2.

    Formule fermée pour les obligations ZC (voir notes de cours p.42).

    Calibration : régression linéaire sur la forme transformée
        (r(t+dt) - r(t)) / sqrt(r(t)) = beta1 * dt/sqrt(r(t)) - a * sqrt(r(t))*dt + epsilon
    """

    def __init__(self, a: float = 0.1, k: float = 0.03, sigma: float = 0.05):
        self.a = a
        self.k = k
        self.sigma = sigma

    def calibrate(self, rate_series: np.ndarray, dt: float = 1/252,
                  window_years: int = 5):
        """
        Calibration CIR par maximum de vraisemblance : MLE avec contrainte pour garantir des taux
        strictement positifs.

        On utilise par défaut les window_years dernières années pour éviter
        les régimes de taux très différents du passé.
        """
        r = np.array(rate_series)

        n_recent = min(len(r), int(window_years * 252))
        r = r[-n_recent:]
        r = np.maximum(r, 1e-6)  # contrainte de positivité

        r_t  = r[:-1]
        r_t1 = r[1:]

        def neg_loglik(params):
            a, k, s = params

            if a <= 0 or k <= 0 or s <= 0:
                return 1e10
            
            # Condition de Feller : 2ak > sigma^2 (garantit r(t) > 0)
            if 2 * a * k <= s ** 2:
                return 1e8
            
            # Approximation gaussienne de la vraisemblance
            mu     = r_t + a * (k - r_t) * dt
            var    = np.maximum(s ** 2 * r_t * dt, 1e-12)

            return 0.5 * float(np.sum(np.log(var) + (r_t1 - mu) ** 2 / var))

        bounds = [(0.01, 10.0), (1e-4, 0.5), (1e-4, 1.0)]
        res = differential_evolution(neg_loglik, bounds, seed=42,
                                      maxiter=300, tol=1e-8, polish=True)
        self.a, self.k, self.sigma = res.x
        return self

    def zero_bond_price(self, r0: float, T: float) -> float:
        """
        Prix analytique d'une obligation ZC
            B(T) = Phi(T) * exp(-r0 * Gamma(T))
        """
        a, k, s = self.a, self.k, self.sigma

        # gamma : racine caractéristique de l'EDO du modèle CIR
        gamma = np.sqrt(a**2 + 2 * s**2)
        exp_gt = np.exp(gamma * T)
        denom = (gamma + a) * (exp_gt - 1) + 2 * gamma

        # Gamma : ensibilité du prix au taux court r0
        Gamma = 2 * (exp_gt - 1) / denom

        # Facteur d'actualisation ajusté du risque de taux
        Phi_exp = (2 * gamma * np.exp((a + gamma) * T / 2) / denom) ** (2 * a * k / s**2)
        return Phi_exp * np.exp(-r0 * Gamma)

    def simulate_paths(self, r0: float, T: float, n_steps: int,
                       n_simulations: int, seed: int = None) -> np.ndarray:
        """
        Simulation Euler-Milstein avec troncature à 0 pour garantir r >= 0 :
            r(t+dt) = max(0, r(t) + a*(k-r(t))*dt + sigma*sqrt(r(t))*sqrt(dt)*Z)
        """
        if seed is not None:
            np.random.seed(seed)
        dt = T / n_steps
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = r0
        Z = np.random.normal(0, 1, (n_simulations, n_steps))

        for i in range(n_steps):
            r = paths[:, i]

            # max(r, 0) dans le terme diffusif évite sqrt d'un nombre négatif
            paths[:, i + 1] = np.maximum(
                0, r + self.a * (self.k - r) * dt + self.sigma * np.sqrt(np.maximum(r, 0)) * np.sqrt(dt) * Z[:, i]
            )
        return paths