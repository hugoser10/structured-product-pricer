import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from abc import ABC, abstractmethod

MATURITY_MAP = {
    "1M": 1/12, "2M": 2/12, "3M": 3/12, "4M": 4/12,
    "6M": 0.5, "9M": 0.75, "1Y": 1, "2Y": 2, "3Y": 3,
    "4Y": 4, "5Y": 5, "6Y": 6, "7Y": 7, "8Y": 8, "9Y": 9,
    "10Y": 10, "15Y": 15, "20Y": 20, "25Y": 25, "30Y": 30, "50Y": 50,
}

def maturity_converter(code: str | None = None):
    '''Converts string maturity code to a number in years'''
    if code is not None:
        if code[-1].upper() == 'M':
            return float(code[:-1]) / 12
        if code[-1].upper() == 'Y':
            return float(code[:-1])


# Rate Curve with Nelson-Siegel-Svensson
class RateCurve:
    """
    Rate curve calibration with Nelson-Siegel-Svensson (NSS) for a given country and date

    NSS formula gives the zero rate for a given maturity based on 6 parameters as:
        y(t) = β0
             + β1 * (1 - exp(-λ1 t)) / (λ1 t)
             + β2 * [(1 - exp(-λ1 t)) / (λ1 t) - exp(-λ1 t)]
             + β3 * [(1 - exp(-λ2 t)) / (λ2 t) - exp(-λ2 t)]

    The calibration is based on the minimization of the squared error between observed and predicted rates
    """

    def __init__(self, country: str = "United States", date: str = None,
                 data_path: str = "data/rate_curves.csv"):
        self.country = country
        self.params = None      # (β0, β1, β2, β3, λ1, λ2)
        self._spline = None
        self._maturities = None
        self._rates = None
        self._load_and_calibrate(data_path, date)

    def _load_and_calibrate(self, path: str, date: str):
        """Loads relevant data and performs calibration"""
        df = pd.read_csv(path)
        df = df[df["country"] == self.country].copy()
        df["maturity_y"] = df["maturity"].apply(lambda mat: maturity_converter(mat))
        df = df.dropna(subset=["maturity_y"])
        df["date"] = pd.to_datetime(df["date"])

        # Sélection de la date
        if date is None:
            target = df["date"].max()
        else:
            target = pd.to_datetime(date)
        df = df[df["date"] == target].sort_values("maturity_y")

        if df.empty:
            raise ValueError(f"Aucune donnée pour {self.country} à la date {target}")

        self._maturities = df["maturity_y"].values
        self._rates = df["rate"].values / 100   # passage en décimal

        # Spline cubique de secours
        self._spline = CubicSpline(self._maturities, self._rates)

        # Calibration NSS
        self._calibrate_nss()

    @staticmethod
    def _nss(t, b0, b1, b2, b3, lam1, lam2):
        """Nelson-Siegel-Svensson's formula"""
        t = np.maximum(t, 1e-6)
        f1 = (1 - np.exp(-lam1 * t)) / (lam1 * t)
        f2 = f1 - np.exp(-lam1 * t)
        f3 = (1 - np.exp(-lam2 * t)) / (lam2 * t) - np.exp(-lam2 * t)
        return b0 + b1 * f1 + b2 * f2 + b3 * f3

    def _calibrate_nss(self):
        """Minimisation des moindres carrés pour les 6 paramètres NSS."""
        def objective(params):
            pred = self._nss(self._maturities, *params)
            return np.sum((pred - self._rates) ** 2)

        # Plusieurs initialisations pour éviter les minima locaux
        best_result, best_val = None, np.inf
        inits = [
            [self._rates[-1], -0.02, 0.01, 0.01, 1.0, 3.0],
            [0.03, -0.01, 0.02, 0.005, 0.5, 2.0],
            [0.04, 0.01, -0.01, 0.02, 1.5, 4.0],
        ]
        for x0 in inits:
            res = minimize(objective, x0,
                           bounds=[(0, 0.2), (-0.2, 0.2), (-0.2, 0.2),
                                   (-0.2, 0.2), (0.01, 5), (0.01, 10)],
                           method="L-BFGS-B")
            if res.fun < best_val:
                best_val, best_result = res.fun, res

        self.params = best_result.x

    def zero_rate(self, maturity: float) -> float:
        """
        Returns the zero rate (decimal) interpolated by NSS for a given maturity
        If NSS was not calibrated, the cubic spline interpolation of the curve is returned
        """
        if self.params is not None:
            return float(self._nss(np.array([maturity]), *self.params)[0])
        else:
            raise ValueError('')

    def discount_factor(self, maturity: float) -> float:
        """Returns the discount factor for a given maturity using continuous compounding"""
        return np.exp(-self.zero_rate(maturity) * maturity)

    def forward_rate(self, t1: float, t2: float) -> float:
        """Returns the forward rate between the two given maturities"""
        r1, r2 = self.zero_rate(t1), self.zero_rate(t2)
        return (r2 * t2 - r1 * t1) / (t2 - t1)

    def get_curve_data(self, maturities=None):
        """Returns a DataFrame of maturities and rates for chosen maturities (iterable)"""
        if maturities is None:
            maturities = np.linspace(0.1, 30, 200)
        rates = [self.zero_rate(t) * 100 for t in maturities]
        return pd.DataFrame({"maturity": maturities, "rate": rates})

data = {
    "mat": ["1M", "2M", "3M"],
    "rate": [0.5, 0.6, 0.9]
}
df = pd.DataFrame(data)
df["mat_y"] = df["mat"].apply(maturity_converter)
print(df)

# Stochastic Models
class StochasticRateModel(ABC):
    """
    Abstract class for stochastic rate curve models

    - calibrate(): estimation des paramètres sur les données historiques
    - zero_bond_price(): calcul du prix d'un ZC
    - simulate_paths(): simulations de Monte Carlo de chemins de taux
        - return: array(n_simulations, n_steps+1)
    """

    @abstractmethod
    def simulate_paths(self, r0: float, T: float, n_steps: int,
                       n_simulations: int, seed: int = None) -> np.ndarray:
        pass

    @abstractmethod
    def zero_bond_price(self, r0: float, T: float) -> float:
        pass

class VasicekModel():
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
        from scipy.optimize import differential_evolution

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