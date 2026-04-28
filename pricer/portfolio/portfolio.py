import numpy as np
import pandas as pd
from typing import List, Tuple, Any, Dict


class Portfolio:
    """Liste de positions (produit, quantité, label) avec agrégations."""

    def __init__(self, name: str = "Portfolio"):
        self.name = name
        self._positions: List[Tuple[Any, float, str]] = []

    def add(self, product, quantity: float = 1.0, label: str = None):
        if label is None:
            label = f"{product.__class__.__name__}_{len(self._positions) + 1}"
        self._positions.append((product, quantity, label))
        return self

    @property
    def positions(self):
        return list(self._positions)

    def total_value(self) -> float:
        return sum(p.price() * q for p, q, _ in self._positions)

    def risk_report(self) -> pd.DataFrame:
        rows = []
        for product, qty, label in self._positions:
            try:
                g = product.greeks()
            except Exception:
                g = {"price": product.price()}
            row = {
                "label": label,
                "type": product.__class__.__name__,
                "quantity": qty,
                "unit_price": g.get("price", 0),
                "total_value": g.get("price", 0) * qty,
            }
            for key in ("delta", "gamma", "vega", "theta", "rho", "dv01",
                        "duration", "ytm", "par_rate"):
                if key in g:
                    row[key] = g[key] * qty
            rows.append(row)
        return pd.DataFrame(rows)

    def bucketed_risk(self, maturities: List[float] = None) -> pd.DataFrame:
        """DV01 et delta agrégés par pilier de maturité."""
        if maturities is None:
            maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 15, 20, 30]
        labels = ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y",
                  "10Y", "15Y", "20Y", "30Y"][:len(maturities)]
        dv01_b = {b: 0.0 for b in labels}
        delta_b = {b: 0.0 for b in labels}

        for product, qty, _ in self._positions:
            try:
                g = product.greeks()
            except Exception:
                continue
            mat = getattr(product, "maturity", getattr(product, "T", None))
            if mat is None:
                continue
            idx = int(np.argmin([abs(mat - m) for m in maturities]))
            b = labels[idx]
            if "dv01" in g:
                dv01_b[b] += g["dv01"] * qty
            if "delta" in g:
                delta_b[b] += g["delta"] * qty

        return pd.DataFrame({
            "maturity_bucket": labels,
            "maturity_year": maturities[:len(labels)],
            "dv01": [dv01_b[b] for b in labels],
            "delta": [delta_b[b] for b in labels],
        })

    def bucketed_strike_risk(self, strike_pct=None) -> pd.DataFrame:
        """Greeks equity agrégés par bucket de moneyness K/S."""
        if strike_pct is None:
            strike_pct = [0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3]
        labels = ["<70%", "70-85%", "85-95%", "ATM", "105-115%", "115-130%", ">130%"]
        buckets = {b: {"delta": 0.0, "gamma": 0.0, "vega": 0.0} for b in labels}

        for product, qty, _ in self._positions:
            S = getattr(product, "S", getattr(product, "spot", None))
            K = getattr(product, "K", getattr(product, "strike",
                       getattr(product, "K2", None)))
            if S is None or K is None or S == 0:
                continue
            try:
                g = product.greeks()
            except Exception:
                continue
            mny = K / S
            if mny < 0.7: b = labels[0]
            elif mny < 0.85: b = labels[1]
            elif mny < 0.95: b = labels[2]
            elif mny < 1.05: b = labels[3]
            elif mny < 1.15: b = labels[4]
            elif mny < 1.30: b = labels[5]
            else: b = labels[6]
            for k in ("delta", "gamma", "vega"):
                buckets[b][k] += g.get(k, 0.0) * qty
        return pd.DataFrame([{"strike_bucket": b, **buckets[b]} for b in labels])

    def summary(self) -> Dict[str, float]:
        totals = {"total_value": 0.0, "total_delta": 0.0, "total_gamma": 0.0,
                  "total_vega": 0.0, "total_dv01": 0.0}
        for product, qty, _ in self._positions:
            try:
                g = product.greeks()
                totals["total_value"] += g.get("price", 0) * qty
                totals["total_delta"] += g.get("delta", 0) * qty
                totals["total_gamma"] += g.get("gamma", 0) * qty
                totals["total_vega"] += g.get("vega", 0) * qty
                totals["total_dv01"] += g.get("dv01", 0) * qty
            except Exception:
                totals["total_value"] += product.price() * qty
        return {"name": self.name, "n_positions": len(self._positions), **totals}

    def pnl_attribution(self, ds: float = 0.0, dr: float = 0.0,
                        dsigma: float = 0.0, dt: float = 1/365) -> Dict[str, float]:
        """ΔV ≈ θ·Δt + Δ·ΔS + ½·Γ·ΔS² + ν·Δσ + ρ·Δr"""
        totals = {"theta": 0, "delta": 0, "gamma": 0, "vega": 0, "rho": 0,
                  "dv01": 0, "convexity": 0}
        for product, qty, _ in self._positions:
            try:
                g = product.greeks()
                for k in totals:
                    totals[k] += g.get(k, 0) * qty
            except Exception:
                pass
        dv01_pnl = totals["dv01"] * dr * 10000
        convex_pnl = 0.5 * totals["convexity"] * dr ** 2
        return {
            "theta_pnl": totals["theta"] * dt * 365,
            "delta_pnl": totals["delta"] * ds,
            "gamma_pnl": 0.5 * totals["gamma"] * ds**2,
            "vega_pnl": totals["vega"] * dsigma * 100,
            "rho_pnl": totals["rho"] * dr * 10000,
            "dv01_pnl": dv01_pnl,
            "convex_pnl": convex_pnl,
            "total_pnl": (totals["theta"] * dt * 365 + totals["delta"] * ds
                          + 0.5 * totals["gamma"] * ds**2
                          + totals["vega"] * dsigma * 100
                          + totals["rho"] * dr * 10000
                          + dv01_pnl + convex_pnl),
        }

    def __len__(self):
        return len(self._positions)

    def __repr__(self):
        return f"Portfolio('{self.name}', {len(self)} positions, value={self.total_value():.2f})"
