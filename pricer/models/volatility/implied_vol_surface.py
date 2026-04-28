import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from pricer.models.volatility.black_scholes import implied_vol_newton


class ImpliedVolSurface:
    """Surface sigma(K, T) construite par inversion BSM Newton-Raphson + spline bicubique."""

    def __init__(self, ticker: str = "MSFT", date: str = None,
                 data_path: str = "pricer/data/options.csv", r: float = 0.045):
        self.ticker, self.r = ticker, r
        self._ivol_grid = None
        self._strikes_grid = None
        self._maturities_grid = None
        self._interp = None
        self.spot = None
        self._build_surface(data_path, date)

    def _build_surface(self, path: str, date: str):
        # Convention marché : puts pour K<S, calls pour K>S, moyenne des deux à l'ATM
        df = pd.read_csv(path, sep=";")
        df = df[df["ticker"] == self.ticker].copy()
        if df.empty:
            raise ValueError(f"Aucune option pour {self.ticker}")
        self.spot = df["underlyingPrice"].iloc[0]
        self._reference_date = date or df["date"].max()
        df["T"] = df["dte"] / 365.0

        results = []
        for _, row in df.iterrows():
            K, T, S = row["strike"], row["T"], self.spot
            mny = K / S
            if mny < 0.99 and row["side"] != "put":
                continue
            if mny > 1.01 and row["side"] != "call":
                continue
            iv = implied_vol_newton(row["mid"], S, K, T, self.r, row["side"])
            if iv is not None and 0.01 < iv < 5.0:
                results.append({"strike": K, "T": T, "iv": iv, "moneyness": mny,
                                "side": row["side"]})

        if not results:
            raise ValueError("Aucune volatilité implicite valide.")

        df_iv = pd.DataFrame(results)
        df_iv = (df_iv.groupby(["strike", "T"], as_index=False)
                      .agg(iv=("iv", "mean"), moneyness=("moneyness", "first")))

        strikes = np.array(sorted(df_iv["strike"].unique()))
        maturities = np.array(sorted(df_iv["T"].unique()))
        grid = np.full((len(maturities), len(strikes)), np.nan)
        for _, row in df_iv.iterrows():
            i = np.searchsorted(maturities, row["T"])
            j = np.searchsorted(strikes, row["strike"])
            if i < len(maturities) and j < len(strikes):
                grid[i, j] = row["iv"]

        # Combler les NaN par interpolation 1D sur strike puis sur maturité
        for i in range(len(maturities)):
            r_ = grid[i, :]
            mask = ~np.isnan(r_)
            if mask.sum() >= 2:
                grid[i, np.isnan(r_)] = np.interp(strikes[np.isnan(r_)], strikes[mask], r_[mask])
            elif mask.sum() == 1:
                grid[i, :] = r_[mask][0]
        for j in range(len(strikes)):
            c = grid[:, j]
            mask = ~np.isnan(c)
            if mask.sum() >= 2:
                grid[np.isnan(c), j] = np.interp(maturities[np.isnan(c)], maturities[mask], c[mask])
            elif mask.sum() == 1:
                grid[:, j] = c[mask][0]

        self._strikes_grid = strikes
        self._maturities_grid = maturities
        self._ivol_grid = grid
        if len(maturities) >= 2 and len(strikes) >= 4:
            kx = min(3, len(maturities) - 1)
            ky = min(3, len(strikes) - 1)
            self._interp = RectBivariateSpline(maturities, strikes, grid, kx=kx, ky=ky)
        else:
            self._interp = None

    def get_vol(self, K: float, T: float) -> float:
        K_ = float(np.clip(K, self._strikes_grid[0], self._strikes_grid[-1]))
        T_ = float(np.clip(T, self._maturities_grid[0], self._maturities_grid[-1]))
        if self._interp is not None:
            vol = float(np.array(self._interp(T_, K_)).flat[0])
            return max(vol, 0.01)
        i = np.clip(np.searchsorted(self._maturities_grid, T_), 0, len(self._maturities_grid) - 1)
        j = np.clip(np.searchsorted(self._strikes_grid, K_), 0, len(self._strikes_grid) - 1)
        return max(float(self._ivol_grid[i, j]), 0.01)

    def get_atm_vol(self, T: float) -> float:
        return self.get_vol(self.spot, T)

    def get_surface_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, T in enumerate(self._maturities_grid):
            for j, K in enumerate(self._strikes_grid):
                iv = self._ivol_grid[i, j]
                if not np.isnan(iv):
                    rows.append({"strike": K, "maturity": T, "iv": iv * 100,
                                 "moneyness": K / self.spot})
        return pd.DataFrame(rows)
