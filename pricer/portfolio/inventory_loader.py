import os
import pandas as pd
from datetime import datetime
from typing import Dict
from pricer.portfolio.portfolio import Portfolio
from pricer.products.rates import (
    InterestRateSwap, FloatingRateBond, ZeroCouponBond, CouponBond,
)
from pricer.products.equity import (
    Option, CallSpread, PutSpread, Butterfly,
)
from pricer.products.exotic import BarrierOption, Autocall


# Codes SSPA → mapping vers les produits du pricer (réplication statique)
SSPA_LABELS = {
    1100: "Tracker Certificate",
    1130: "Bonus Certificate",
    1220: "Capped Capital Protection",
    1320: "Reverse Convertible",
}


class InventoryLoader:
    """
    Parse l'Inventaire.xlsx (4 feuilles : Swap, Options, Autocall, Notes structurées)
    et construit un dict {nom_feuille: Portfolio} avec les produits du pricer.
    """

    def __init__(self, path: str = None,
                 rate_curve=None, vol_surface=None, spot: float = None,
                 r: float = None, hw_model=None, heston=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "..", "data", "Inventaire.xlsx")
        self.path = path
        self.rate_curve = rate_curve
        self.vol_surface = vol_surface
        self.spot = spot
        self.r = r
        self.hw_model = hw_model
        self.heston = heston
        self._sheets: Dict[str, pd.DataFrame] = {}
        self._load_sheets()

    def _load_sheets(self):
        xls = pd.ExcelFile(self.path)
        for s in xls.sheet_names:
            self._sheets[s] = pd.read_excel(xls, sheet_name=s)

    def get_sheet(self, name: str) -> pd.DataFrame:
        return self._sheets.get(name, pd.DataFrame()).copy()

    @staticmethod
    def _years_between(d1, d2) -> float:
        if not isinstance(d1, datetime):
            d1 = pd.to_datetime(d1)
        if not isinstance(d2, datetime):
            d2 = pd.to_datetime(d2)
        return max((d2 - d1).days / 365.25, 1e-3)

    @staticmethod
    def _freq_to_int(s) -> int:
        if pd.isna(s) or s is None:
            return 2
        s = str(s).upper()
        return {"1M": 12, "3M": 4, "6M": 2, "1Y": 1}.get(s, 2)

    # Constructions par feuille

    def build_swaps(self) -> Portfolio:
        port = Portfolio("Inventaire — Swaps")
        df = self.get_sheet("Swap")
        for _, row in df.iterrows():
            T = self._years_between(row["Date Valorisation"], row["Maturité"])
            freq_fixe = self._freq_to_int(row.get("Fréquence fixe"))
            nominal = float(row["Nominal"])
            fixed_rate = row.get("Taux fixe")
            # Cas 1 : swap fixe contre variable → IRS classique
            # Cas 2 : variable vs variable (ligne sans taux fixe) → on approxime par un FRN
            if pd.notna(fixed_rate):
                irs = InterestRateSwap(
                    nominal=nominal, fixed_rate=float(fixed_rate),
                    maturity=T, frequency=freq_fixe,
                    pay_fixed=True, rate_curve=self.rate_curve,
                )
                port.add(irs, quantity=1,
                         label=f"Swap {T:.1f}Y fixe={fixed_rate*100:.1f}% N={nominal:,.0f}")
            else:
                frn = FloatingRateBond(
                    nominal=nominal, maturity=T, spread=0.0,
                    frequency=freq_fixe, rate_curve=self.rate_curve,
                )
                port.add(frn, quantity=1,
                         label=f"Basis Swap {T:.1f}Y N={nominal:,.0f}")
        return port

    def build_options(self) -> Portfolio:
        port = Portfolio("Inventaire — Options & Stratégies")
        df = self.get_sheet("Options")
        for _, row in df.iterrows():
            T = self._years_between(row["Date Valorisation"], row["Maturité"])
            qty = int(row["Quantité"])
            prod_type = str(row["Produit"]).strip()
            K1 = row.get("Strike 1")
            K2 = row.get("Strike 2")
            K3 = row.get("Strike 3")
            barrier_type = row.get("Type Barrière")
            barrier_lvl = row.get("Niveau Barrière")
            S, r = self.spot, self.r

            if prod_type == "Call Spread":
                p = CallSpread(S, float(K1), float(K2), T, r, vol_surface=self.vol_surface)
                lbl = f"Call Spread K={K1:.0f}/{K2:.0f}"
            elif prod_type == "Put Spread":
                p = PutSpread(S, float(K1), float(K2), T, r, vol_surface=self.vol_surface)
                lbl = f"Put Spread K={K1:.0f}/{K2:.0f}"
            elif prod_type == "Butterfly":
                p = Butterfly(S, float(K1), float(K2), float(K3), T, r,
                              vol_surface=self.vol_surface)
                lbl = f"Butterfly K={K1:.0f}/{K2:.0f}/{K3:.0f}"
            elif prod_type in ("Call", "Put") and pd.notna(barrier_type):
                # Option vanille avec barrière -> BarrierOption KO/KI
                bt = "up-and-out" if str(barrier_type).upper() == "OUT" else "up-and-in"
                if float(barrier_lvl) < S:
                    bt = bt.replace("up", "down")
                p = BarrierOption(S, float(K1), T, r, float(barrier_lvl),
                                   barrier_type=bt, option_type=prod_type.lower(),
                                   heston=self.heston, vol_surface=self.vol_surface)
                lbl = f"{prod_type} Barrier {bt} K={K1:.0f} H={barrier_lvl:.0f}"
            elif prod_type in ("Call", "Put"):
                p = Option(S, float(K1), T, r, option_type=prod_type.lower(),
                           vol_surface=self.vol_surface)
                lbl = f"{prod_type} K={K1:.0f}"
            else:
                continue
            port.add(p, quantity=qty, label=lbl)
        return port

    def build_autocalls(self) -> Portfolio:
        """Une ligne = une observation. On regroupe par ID Produit pour reconstruire l'autocall."""
        port = Portfolio("Inventaire — Autocalls")
        df = self.get_sheet("Autocall")
        if df.empty:
            return port

        for pid, grp in df.groupby("ID Produit"):
            grp = grp.sort_values("Date Observation").reset_index(drop=True)
            ref_date = grp["Date Référence"].iloc[0]
            val_date = grp["Date Valorisation"].iloc[0]
            last_obs = grp["Date Observation"].iloc[-1]

            T_to_last = self._years_between(val_date, last_obs)
            barrier_call = {i + 1: float(row["Niveau de rappel"]) for i, row in grp.iterrows()}
            avg_coupon = float(grp["Coupon"].mean())

            ac = Autocall(
                spot=self.spot, strike=self.spot, coupon=avg_coupon,
                barrier_call=barrier_call, barrier_final=0.5,
                maturity=T_to_last, n_obs=max(int(len(grp) / max(T_to_last, 1)), 1),
                r=self.r, sigma=0.2, heston=self.heston, vol_surface=self.vol_surface,
            )
            port.add(ac, quantity=1,
                     label=f"Autocall #{int(pid)} — {len(grp)} obs, T={T_to_last:.1f}Y")
        return port

    def build_structured_notes(self) -> Portfolio:
        """
        Notes structurées SSPA (réplication statique) :
        - 1100 Tracker        : ZCB + Call (participation 1)
        - 1130 Bonus          : ZCB + Call(0) − Put down-in (barrière 1)
        - 1220 Capped Capital : ZCB + Call(K=spot, capé à `Cap`)
        - 1320 Reverse Conv.  : ZCB + short Put (barrière 2 = strike, barrière 1 = KI)
        """
        port = Portfolio("Inventaire — Notes structurées")
        df = self.get_sheet("Notes structurées")
        S, r = self.spot, self.r

        for _, row in df.iterrows():
            T = self._years_between(row["Date valorisation"], row["Maturité"])
            qty = int(row["Quantité"])
            code = int(row["Code produit SSPA"])
            participation = row.get("Taux de participation")
            barrier1 = row.get("Barrière 1")
            cap = row.get("Cap")
            barrier2 = row.get("Barrière 2")
            tag = SSPA_LABELS.get(code, f"SSPA {code}")

            if code == 1100:
                # Tracker = ZCB(N) + (participation) · Call(K=0)
                # Approche simplifiée : prix ≈ S0 (réplique le sous-jacent)
                opt = Option(S, 1e-3, T, r, option_type="call", vol_surface=self.vol_surface)
                port.add(opt, quantity=qty * float(participation or 1.0),
                         label=f"{tag} — Tracker T={T:.1f}Y")
            elif code == 1130:
                # Bonus Certificate ≈ Call(K=0) − Put down-in à barrier1
                kib = BarrierOption(S, S, T, r, float(barrier1),
                                     barrier_type="down-and-in", option_type="put",
                                     heston=self.heston, vol_surface=self.vol_surface)
                port.add(kib, quantity=-qty,
                         label=f"{tag} — Bonus T={T:.1f}Y B={barrier1:.0f}")
            elif code == 1220:
                # Capital Protection capé = ZCB + Call Spread (S, cap)
                cs = CallSpread(S, S, float(cap), T, r, vol_surface=self.vol_surface)
                port.add(cs, quantity=qty,
                         label=f"{tag} — Capped K={S:.0f}/{cap:.0f}")
            elif code == 1320:
                # Reverse Convertible avec barrière = short Put down-in
                rc_put = BarrierOption(S, float(barrier2), T, r, float(barrier1),
                                        barrier_type="down-and-in", option_type="put",
                                        heston=self.heston, vol_surface=self.vol_surface)
                port.add(rc_put, quantity=-qty,
                         label=f"{tag} — RevConv K={barrier2:.0f} B={barrier1:.0f}")
        return port

    def build_all(self) -> Dict[str, Portfolio]:
        return {
            "Swaps": self.build_swaps(),
            "Options & Stratégies": self.build_options(),
            "Autocalls": self.build_autocalls(),
            "Notes structurées": self.build_structured_notes(),
        }
