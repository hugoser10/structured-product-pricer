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