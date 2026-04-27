from pricer.portfolio.portfolio import Portfolio
from pricer.products.rates import (
    ZeroCouponBond, CouponBond, FloatingRateBond, InterestRateSwap,
    CallableBond, PutableBond,
)
from pricer.products.equity import (
    Option, CallSpread, PutSpread, Butterfly,
)
from pricer.products.exotic import BarrierOption, Autocall
from pricer.models.rates.hull_white_model import HullWhiteModel
from pricer.models.volatility.heston_model import HestonModel


def build_portfolios(market_data: dict) -> dict:
    """4 portefeuilles de démo : Taux, Vanilles, Exotiques, Mixte."""
    rc = market_data["rate_curve"]
    vol = market_data["vol_surface"]
    S = market_data["spot"]
    r = market_data["r"]

    hw = HullWhiteModel(a=0.1, sigma=0.01, rate_curve=rc)
    heston = HestonModel(v0=0.04, kappa=1.5, theta=0.04, zeta=0.3, rho=-0.7)

    # ---------- Taux ----------
    p1 = Portfolio("Taux")
    p1.add(ZeroCouponBond(1000, 2.0, rc), 10, "ZCB 2Y")
    p1.add(ZeroCouponBond(1000, 5.0, rc), 5, "ZCB 5Y")
    p1.add(CouponBond(1000, 0.03, 5.0, 2, rc), 10, "Bond 5Y 3%")
    p1.add(CouponBond(1000, 0.045, 10.0, 1, rc), 5, "Bond 10Y 4.5%")
    p1.add(FloatingRateBond(1000, 3.0, 0.005, 4, rc), 8, "Floater 3Y +50bps")
    p1.add(InterestRateSwap(100_000, 0.035, 5.0, 2, True, rc), 1, "IRS 5Y payeur 3.5%")
    p1.add(CallableBond(1000, 0.04, 5.0, [2, 3, 4], 1000, 1, hw), 5, "Callable 5Y 4%")
    p1.add(PutableBond(1000, 0.025, 5.0, [2, 3, 4], 1000, 1, hw), 5, "Putable 5Y 2.5%")

    # ---------- Vanilles ----------
    p2 = Portfolio("Vanilles")
    p2.add(Option(S, S, 1.0, r, "call", vol_surface=vol), 100, "Call ATM 1Y")
    p2.add(Option(S, S * 1.1, 1.0, r, "call", vol_surface=vol), -100, "Call OTM +10% short")
    p2.add(Option(S, S * 0.9, 1.0, r, "put", vol_surface=vol), 100, "Put OTM −10%")
    p2.add(CallSpread(S, S, S * 1.1, 1.0, r, vol_surface=vol), 50, "CS ATM/+10%")
    p2.add(PutSpread(S, S * 0.85, S * 0.95, 0.5, r, vol_surface=vol), 50, "PS −15%/−5%")
    p2.add(Butterfly(S, S * 0.9, S, S * 1.1, 1.0, r, vol_surface=vol), 20, "Butterfly ±10%")

    # ---------- Exotiques ----------
    p3 = Portfolio("Exotiques")
    p3.add(BarrierOption(S, S, 1.0, r, S * 0.8, "down-and-out", "call",
                          heston=heston), 10, "Call DownOut 80%")
    p3.add(BarrierOption(S, S * 0.9, 1.0, r, S * 0.7, "down-and-in", "put",
                          heston=heston), 10, "Put DownIn 70%")
    p3.add(Autocall(S, S, 0.05, {i+1: 1.0 for i in range(5)}, 0.5, 5.0, 1,
                    r, heston=heston), 10, "Autocall 5Y 5%")
    p3.add(Autocall(S, S, 0.04, {i+1: 1.0 - 0.05*i for i in range(5)}, 0.6, 5.0, 1,
                    r, heston=heston), 10, "Autocall stepdown 5Y")

    # ---------- Mixte ----------
    p4 = Portfolio("Mixte")
    p4.add(CouponBond(1000, 0.04, 5.0, 2, rc), 20, "Bond 5Y 4%")
    p4.add(InterestRateSwap(50_000, 0.04, 5.0, 2, False, rc), 1, "IRS receveur (hedge)")
    p4.add(Option(S, S, 1.0, r, "call", vol_surface=vol), 50, "Call ATM 1Y")
    p4.add(CallSpread(S, S, S * 1.15, 2.0, r, vol_surface=vol), 30, "CS 2Y +15%")
    p4.add(Autocall(S, S, 0.06, {i+1: 1.0 for i in range(3)}, 0.5, 3.0, 1, r), 5,
           "Autocall 3Y 6%")

    return {"Taux": p1, "Vanilles": p2, "Exotiques": p3, "Mixte": p4}
