import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline


MATURITY_MAP = {
    "1M": 1/12, "2M": 2/12, "3M": 3/12, "4M": 4/12,
    "6M": 0.5, "9M": 0.75, "1Y": 1, "2Y": 2, "3Y": 3,
    "4Y": 4, "5Y": 5, "6Y": 6, "7Y": 7, "8Y": 8, "9Y": 9,
    "10Y": 10, "15Y": 15, "20Y": 20, "25Y": 25, "30Y": 30, "50Y": 50,
}


class RateCurve:
    """Courbe zéro-coupon calibrée par Nelson-Siegel-Svensson (NSS)."""

    def __init__(self, country: str = "United States", date: str = None,
                 data_path: str = "pricer/data/rate_curves.parquet"):
        self.country = country
        self.params = None
        self._spline = None
        self._maturities = None
        self._rates = None
        self._load_and_calibrate(data_path, date)

    def _load_and_calibrate(self, path: str, date: str):
        df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
        df = df[df["country"] == self.country].copy()
        df["maturity_y"] = df["maturity"].map(MATURITY_MAP)
        df = df.dropna(subset=["maturity_y"])
        df["date"] = pd.to_datetime(df["date"])
        target = pd.to_datetime(date) if date else df["date"].max()
        df = df[df["date"] == target].sort_values("maturity_y")
        if df.empty:
            raise ValueError(f"Aucune donnée pour {self.country} à {target}")
        self._maturities = df["maturity_y"].values
        self._rates = df["rate"].values / 100
        self._spline = CubicSpline(self._maturities, self._rates)
        self._calibrate_nss()

    @staticmethod
    def _nss(t, b0, b1, b2, b3, lam1, lam2):
        # y(t) = b0 + b1*f1 + b2*(f1 - exp(-l1*t)) + b3*(f3 - exp(-l2*t))
        t = np.maximum(t, 1e-6)
        f1 = (1 - np.exp(-lam1 * t)) / (lam1 * t)
        f2 = f1 - np.exp(-lam1 * t)
        f3 = (1 - np.exp(-lam2 * t)) / (lam2 * t) - np.exp(-lam2 * t)
        return b0 + b1 * f1 + b2 * f2 + b3 * f3

    def _calibrate_nss(self):
        # 3 initialisations pour éviter les minima locaux
        def obj(p): return float(np.sum((self._nss(self._maturities, *p) - self._rates) ** 2))
        bounds = [(0, 0.2), (-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2), (0.01, 5), (0.01, 10)]
        inits = [
            [self._rates[-1], -0.02, 0.01, 0.01, 1.0, 3.0],
            [0.03, -0.01, 0.02, 0.005, 0.5, 2.0],
            [0.04, 0.01, -0.01, 0.02, 1.5, 4.0],
        ]
        best, val = None, np.inf
        for x0 in inits:
            res = minimize(obj, x0, bounds=bounds, method="L-BFGS-B")
            if res.fun < val:
                val, best = res.fun, res
        self.params = best.x

    def zero_rate(self, maturity: float) -> float:
        if self.params is not None:
            return float(self._nss(np.array([maturity]), *self.params)[0])
        return float(self._spline(maturity))

    def discount_factor(self, maturity: float) -> float:
        return np.exp(-self.zero_rate(maturity) * maturity)

    def forward_rate(self, t1: float, t2: float) -> float:
        # f(t1, t2) = (r(t2)*t2 - r(t1)*t1) / (t2 - t1)
        r1, r2 = self.zero_rate(t1), self.zero_rate(t2)
        return (r2 * t2 - r1 * t1) / (t2 - t1)

    def get_curve_data(self, maturities=None) -> pd.DataFrame:
        if maturities is None:
            maturities = np.linspace(0.1, 30, 200)
        rates = [self.zero_rate(t) * 100 for t in maturities]
        return pd.DataFrame({"maturity": maturities, "rate": rates})
