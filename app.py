import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(__file__))

from pricer.data import (
    build_market_data, available_tickers, available_dates, DEFAULT_INVENTORY_PATH,
)

from pricer.models.rates import RateCurve, VasicekModel, CIRModel, HullWhiteModel, MATURITY_MAP
from pricer.models.volatility import ImpliedVolSurface, HestonModel, bs_call, bs_put

from pricer.products import (
    ZeroCouponBond, CouponBond, FloatingRateBond, InterestRateSwap,
    CallableBond, PutableBond,
    Option, CallSpread, PutSpread, Butterfly, BarrierOption, Autocall,
    TrackerCertificate, BonusCertificate,
    CappedCapitalProtection, ReverseConvertible,
)

from pricer.portfolio import Portfolio, InventoryLoader, build_portfolios

# Palette de couleurs
class Palette:
    # Fonds
    PAGE_BG = "#FFFFFF"
    PLOT_BG = "#FFFFFF"
    PANEL_BG = "#F8F9FA"

    # Texte
    TEXT_MAIN = "#212529"
    TEXT_MUTED = "#6C757D"

    # Grilles et axes
    GRID = "#E9ECEF"
    AXIS = "#ADB5BD"
    ZERO_LINE = "#CED4DA"

    # Couleurs métier
    BLUE = "#2E86C1"  
    ORANGE = "#E67E22"  
    GREEN = "#27AE60"  
    PURPLE = "#8E44AD"  
    RED = "#C0392B" 
    YELLOW = "#F1C40F"
    TEAL = "#16A085" 

    # Lignes de référence
    SPOT_LINE = "#27AE60"
    STRIKE_LINE = "#34495E"

    # Séquence pour catégories
    CATEGORICAL = ["#2E86C1", "#E67E22", "#27AE60", "#8E44AD", "#E74C3C"]


COLORS = Palette()

# Configuration et thème
st.set_page_config(page_title="Pricer Multi-Produits", layout="wide")

st.markdown(f"""
<style>
    .explain-box {{
        background: {COLORS.PANEL_BG};
        border-left: 3px solid {COLORS.BLUE};
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: {COLORS.TEXT_MAIN};
        margin-bottom: 12px;
    }}
</style>
""", unsafe_allow_html=True)


def explain(text: str):
    st.markdown(f'<div class="explain-box">{text}</div>', unsafe_allow_html=True)


def chart_layout(**kwargs) -> dict:
    """Mise en forme uniforme des graphiques (thème clair)."""
    base = dict(
        template="plotly_white",
        paper_bgcolor=COLORS.PAGE_BG,
        plot_bgcolor=COLORS.PLOT_BG,
        font=dict(color=COLORS.TEXT_MAIN, size=13),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=COLORS.GRID,
            borderwidth=1,
            font=dict(color=COLORS.TEXT_MAIN, size=12),
        ),
        xaxis=dict(color=COLORS.TEXT_MUTED, gridcolor=COLORS.GRID, zerolinecolor=COLORS.ZERO_LINE),
        yaxis=dict(color=COLORS.TEXT_MUTED, gridcolor=COLORS.GRID, zerolinecolor=COLORS.ZERO_LINE),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    base.update(kwargs)
    return base


# Cache
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data(ticker, country, options_date):
    return build_market_data(ticker=ticker, country=country, options_date=options_date)


@st.cache_resource
def get_default_portfolios(ticker, country, options_date):
    md = build_market_data(ticker=ticker, country=country, options_date=options_date)
    return build_portfolios(md)


@st.cache_resource
def get_inventory_portfolios(ticker, country, options_date):
    md = build_market_data(ticker=ticker, country=country, options_date=options_date)
    heston = HestonModel()
    inv = InventoryLoader(
        rate_curve=md["rate_curve"], vol_surface=md["vol_surface"],
        spot=md["spot"], r=md["r"], heston=heston,
    )
    return inv.build_all(), inv

# Sidebar
with st.sidebar:
    st.markdown("## Paramètres globaux")
    ticker = st.selectbox("Sous-jacent", available_tickers(), index=0)
    country = st.selectbox("Pays (taux)", ["United States", "France", "Germany"], index=0)
    options_date = st.selectbox("Date des options", available_dates(ticker=ticker), index=0)

    st.divider()
    rate_model_choice = st.radio("Modèle de taux stochastique", ["Hull-White", "CIR", "Vasicek"])
    vol_model_choice = st.radio("Modèle de volatilité", ["Surface implicite", "Heston"])

    st.divider()
    st.markdown("**Monte Carlo**")
    n_sims = st.slider("Simulations", 1000, 50000, 10000, step=1000)
    mc_seed = st.number_input("Seed", value=42, step=1)

    st.divider()
    st.caption("Master 272 - Pricing de produits structurés (2026) - LIESSE SERICOLA JEHANNIN")

# Chargement des données de marché
with st.spinner("Chargement des données de marché..."):
    try:
        mkt = get_market_data(ticker, country, options_date)
        rc = mkt["rate_curve"]
        vol_surface = mkt["vol_surface"]
        S, r = mkt["spot"], mkt["r"]
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        st.stop()

# Onglets
tabs = st.tabs(["Marché", "Pricing", "Portefeuilles démo",
                "Inventaire", "Risques", "Calibration"])

# 1. MARCHÉ
with tabs[0]:
    st.title("Données de marché")
    c1, c2, c3 = st.columns(3)
    c1.metric("Spot", f"{S:.2f}")
    c2.metric("Taux 3M", f"{rc.zero_rate(0.25)*100:.3f}%")
    c3.metric("Vol ATM 1Y", f"{vol_surface.get_atm_vol(1.0)*100:.1f}%")

    # Courbe de taux
    st.markdown("### Courbe zéro-coupon (NSS)")
    explain("Calibration <b>Nelson-Siegel-Svensson</b> sur les taux de marché. "
            "La pente reflète les anticipations macroéconomiques.")
    cd = rc.get_curve_data()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cd["maturity"], y=cd["rate"], mode="lines",
        name="Courbe NSS",
        line=dict(color=COLORS.BLUE, width=2.5),
    ))
    mats_obs = sorted(MATURITY_MAP.values())
    rates_obs = [rc.zero_rate(m) * 100 for m in mats_obs]
    fig.add_trace(go.Scatter(
        x=mats_obs, y=rates_obs, mode="markers",
        name="Points de marché",
        marker=dict(color=COLORS.ORANGE, size=7,
                    line=dict(color=COLORS.TEXT_MAIN, width=0.5)),
    ))
    fig.update_layout(**chart_layout(
        xaxis_title="Maturité (années)", yaxis_title="Taux (%)", height=350,
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Surface de volatilité
    st.markdown("### Surface de volatilité implicite")
    explain("Volatilités obtenues par <b>inversion BSM Newton-Raphson</b>. "
            "Convention : puts pour K&lt;S, calls pour K&gt;S.")
    try:
        vol_df = vol_surface.get_surface_dataframe()
        if vol_df["maturity"].nunique() >= 2:
            pivot = (vol_df.pivot_table(index="maturity", columns="moneyness",
                                        values="iv", aggfunc="mean")
                            .interpolate(axis=1).interpolate(axis=0))
            fig_s = go.Figure(data=[go.Surface(
                x=pivot.columns.values, y=pivot.index.values, z=pivot.values,
                colorscale="Viridis",
                colorbar=dict(title="Vol (%)", tickfont=dict(color=COLORS.TEXT_MAIN)),
            )])
            fig_s.update_layout(**chart_layout(
                scene=dict(
                    xaxis=dict(title="Moneyness", backgroundcolor=COLORS.PANEL_BG,
                               gridcolor=COLORS.GRID),
                    yaxis=dict(title="Maturité (années)", backgroundcolor=COLORS.PANEL_BG,
                               gridcolor=COLORS.GRID),
                    zaxis=dict(title="Vol (%)", backgroundcolor=COLORS.PANEL_BG,
                               gridcolor=COLORS.GRID),
                    bgcolor=COLORS.PAGE_BG,
                    camera=dict(eye=dict(x=1.8, y=-1.5, z=0.8)),
                ),
                height=480, title=f"Surface implicite - {ticker}",
            ))
            st.plotly_chart(fig_s, use_container_width=True)

        # Smile
        st.markdown("### Smile par maturité")
        fig_sm = go.Figure()
        for i, T in enumerate(sorted(vol_df["maturity"].unique())):
            sub = vol_df[vol_df["maturity"] == T].sort_values("moneyness")
            fig_sm.add_trace(go.Scatter(
                x=sub["moneyness"], y=sub["iv"], mode="lines+markers",
                name=f"{int(T*365)} jours",
                line=dict(color=COLORS.CATEGORICAL[i % len(COLORS.CATEGORICAL)], width=2),
            ))
        fig_sm.add_vline(x=1.0, line_dash="dash", line_color=COLORS.AXIS,
                          annotation_text="ATM")
        fig_sm.update_layout(**chart_layout(
            xaxis_title="Moneyness (K/S)", yaxis_title="Volatilité implicite (%)",
            height=400,
        ))
        st.plotly_chart(fig_sm, use_container_width=True)
    except Exception as e:
        st.warning(f"Affichage de la surface impossible : {e}")

# 2. PRICING INTERACTIF
with tabs[1]:
    st.title("Pricer interactif")
    explain("Sélectionnez un type de produit, ajustez les paramètres et obtenez "
            "instantanément prix et greeks.")

    product_type = st.selectbox("Type de produit", [
        "Option Européenne", "Call Spread", "Put Spread", "Butterfly",
        "Option à Barrière", "Autocall",
        "Tracker Certificate (1100)", "Bonus Certificate (1130)",
        "Capped Capital Protection (1220)", "Reverse Convertible (1320)",
        "Obligation Zéro-Coupon", "Obligation à Coupons", "Swap de Taux",
    ])
    st.divider()

    # Option Européenne
    if product_type == "Option Européenne":
        c1, c2, c3 = st.columns(3)
        K = c1.number_input("Strike", value=float(round(S)), step=1.0)
        side = c1.radio("Type", ["call", "put"])
        T = c2.number_input("Maturité (années)", value=1.0, step=0.25, min_value=0.01)
        q = c2.number_input("Dividende q", value=0.0, step=0.001, format="%.3f")
        use_surf = c3.checkbox("Utiliser la surface implicite", value=True)
        sig_m = c3.number_input("Volatilité manuelle", value=0.20, step=0.01, format="%.2f")

        heston = HestonModel() if vol_model_choice == "Heston" else None
        opt = Option(S, K, T, r, side, sig_m, q,
                     vol_surface=vol_surface if use_surf else None, heston=heston)
        g = opt.greeks()

        cols = st.columns(6)
        for col, (label, val) in zip(cols, [
            ("Prix", f"{g['price']:.4f}"), ("Delta", f"{g['delta']:.4f}"),
            ("Gamma", f"{g['gamma']:.6f}"), ("Vega", f"{g['vega']:.4f}"),
            ("Theta/jour", f"{g['theta']:.4f}"), ("Rho", f"{g['rho']:.4f}"),
        ]):
            col.metric(label, val)

        K_range = np.linspace(S * 0.5, S * 1.5, 200)
        payoffs = (np.maximum(K_range - K, 0) if side == "call"
                   else np.maximum(K - K_range, 0))
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=K_range, y=payoffs, name="Payoff brut",
                                    line=dict(color=COLORS.AXIS, dash="dash", width=1.5)))
        fig_p.add_trace(go.Scatter(x=K_range, y=payoffs - g["price"], name="P&L net",
                                    line=dict(color=COLORS.TEAL, width=2.5)))
        fig_p.add_vline(x=K, line_dash="dot", line_color=COLORS.STRIKE_LINE,
                          annotation_text="Strike")
        fig_p.add_vline(x=S, line_color=COLORS.SPOT_LINE, line_dash="dash",
                          annotation_text="Spot")
        fig_p.add_hline(y=0, line_color=COLORS.AXIS, opacity=0.6)
        fig_p.update_layout(**chart_layout(
            xaxis_title="Sous-jacent à maturité", yaxis_title="P&L", height=350,
        ))
        st.plotly_chart(fig_p, use_container_width=True)

    # Call Spread / Put Spread
    elif product_type in ("Call Spread", "Put Spread"):
        c1, c2 = st.columns(2)
        K1 = c1.number_input("Strike bas (K1)", value=float(round(S * 0.95, -1)), step=1.0)
        K2 = c2.number_input("Strike haut (K2)", value=float(round(S * 1.1, -1)), step=1.0)
        T = st.number_input("Maturité (années)", value=1.0, step=0.25)
        prod = (CallSpread(S, K1, K2, T, r, vol_surface=vol_surface)
                if product_type == "Call Spread"
                else PutSpread(S, K1, K2, T, r, vol_surface=vol_surface))
        g = prod.greeks()
        cols = st.columns(5)
        for col, (label, val) in zip(cols, [
            ("Prix", f"{g['price']:.4f}"), ("Delta", f"{g['delta']:.4f}"),
            ("Gamma", f"{g['gamma']:.6f}"), ("Vega", f"{g['vega']:.4f}"),
            ("Theta/jour", f"{g['theta']:.4f}"),
        ]):
            col.metric(label, val)

        K_range = np.linspace(S * 0.5, S * 1.5, 300)
        payoffs = (np.clip(K_range - K1, 0, K2 - K1) if product_type == "Call Spread"
                   else np.clip(K2 - K_range, 0, K2 - K1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=K_range, y=payoffs, name="Payoff brut",
                                  line=dict(color=COLORS.AXIS, dash="dash", width=1.5)))
        fig.add_trace(go.Scatter(x=K_range, y=payoffs - g["price"], name="P&L net",
                                  line=dict(color=COLORS.TEAL, width=2.5)))
        fig.add_hline(y=0, line_color=COLORS.AXIS, opacity=0.6)
        fig.add_vline(x=S, line_color=COLORS.SPOT_LINE, line_dash="dash",
                       annotation_text="Spot")
        fig.update_layout(**chart_layout(
            xaxis_title="Sous-jacent à maturité", yaxis_title="P&L", height=300,
        ))
        st.plotly_chart(fig, use_container_width=True)

    # Butterfly
    elif product_type == "Butterfly":
        c1, c2, c3 = st.columns(3)
        K1 = c1.number_input("Strike bas (K1)", value=float(round(S * 0.9, -1)))
        K2 = c2.number_input("Strike central (K2)", value=float(round(S, -1)))
        K3 = c3.number_input("Strike haut (K3)", value=float(round(S * 1.1, -1)))
        T = st.number_input("Maturité (années)", value=1.0, step=0.25)
        prod = Butterfly(S, K1, K2, K3, T, r, vol_surface=vol_surface)
        g = prod.greeks()
        cols = st.columns(4)
        for col, (label, val) in zip(cols, [
            ("Prix", f"{g['price']:.4f}"), ("Delta", f"{g['delta']:.4f}"),
            ("Gamma", f"{g['gamma']:.6f}"), ("Vega", f"{g['vega']:.4f}"),
        ]):
            col.metric(label, val)

    #  Option à Barrière 
    elif product_type == "Option à Barrière":
        c1, c2, c3 = st.columns(3)
        K = c1.number_input("Strike", value=float(round(S)))
        side = c1.radio("Type", ["call", "put"])
        T = c2.number_input("Maturité (années)", value=1.0, step=0.25)
        bt = c2.selectbox("Type de barrière",
                            ["down-and-out", "up-and-out", "down-and-in", "up-and-in"])
        H_default = S * 0.8 if "down" in bt else S * 1.2
        H = c3.number_input("Niveau de barrière (H)", value=float(round(H_default)))
        rebate = c3.number_input("Rebate", value=0.0)

        heston = HestonModel() if vol_model_choice == "Heston" else None
        sig = vol_surface.get_vol(K, T)
        with st.spinner("Pricing Monte Carlo en cours..."):
            prod = BarrierOption(S, K, T, r, H, bt, side, rebate, sig, 0.0,
                                  heston, vol_surface if not heston else None)
            p = prod.price(n_simulations=n_sims, seed=mc_seed)
            van = Option(S, K, T, r, side, sigma=sig, vol_surface=vol_surface).price()
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix avec barrière", f"{p:.4f}")
        c2.metric("Vanille équivalent", f"{van:.4f}")
        c3.metric("Décote due à la barrière", f"{van - p:.4f}")

    # Autocall
    elif product_type == "Autocall":
        c1, c2 = st.columns(2)
        T = c1.number_input("Maturité (années)", value=5.0, step=1.0)
        coupon = c1.number_input("Coupon (%)", value=5.0, step=0.5)
        bf = c1.slider("Barrière finale (% du strike)", 30, 90, 50) / 100
        n_obs = c2.selectbox("Observations par an", [1, 2, 4], index=0)
        sd = c2.checkbox("Structure step-down", value=False)
        sd_pct = c2.slider("Baisse annuelle de la barrière (%)", 0, 15, 5) if sd else 0

        total_obs = int(T * n_obs)
        bcall = {i + 1: max(1.0 - sd_pct/100 * ((i+1)/n_obs - 1), 0.5) if sd else 1.0
                 for i in range(total_obs)}
        heston = HestonModel() if vol_model_choice == "Heston" else None
        sig = vol_surface.get_atm_vol(T)
        with st.spinner("Pricing Monte Carlo de l'autocall..."):
            prod = Autocall(S, S, coupon/100, bcall, bf, T, n_obs, r, sig, 0.0, heston)
            price = prod.price(n_simulations=n_sims, seed=mc_seed)
            opt_c = prod.optimal_coupon(target_price=1.0, n_simulations=n_sims)
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix (% du nominal)", f"{price*100:.2f}%")
        c2.metric("Coupon at-the-par", f"{opt_c*100:.2f}%")
        c3.metric("Barrière finale", f"{bf*100:.0f}%")

    # Obligation Zéro-Coupon 
    elif product_type == "Obligation Zéro-Coupon":
        c1, c2 = st.columns(2)
        N = c1.number_input("Nominal", value=1000.0)
        T = c2.number_input("Maturité (années)", value=5.0)
        prod = ZeroCouponBond(N, T, rc)
        g = prod.greeks()
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix", f"{g['price']:.2f}")
        c2.metric("Taux zéro", f"{g['zero_rate']*100:.3f}%")
        c3.metric("DV01", f"{g['dv01']:.4f}")

    #  Obligation à Coupons 
    elif product_type == "Obligation à Coupons":
        c1, c2, c3 = st.columns(3)
        N = c1.number_input("Nominal", value=1000.0)
        cr = c2.number_input("Coupon (%)", value=4.0) / 100
        T = c3.number_input("Maturité (années)", value=5.0)
        freq = st.selectbox("Fréquence de coupon", [1, 2, 4, 12], index=1)
        prod = CouponBond(N, cr, T, freq, rc)
        g = prod.greeks()
        cols = st.columns(4)
        for col, (label, val) in zip(cols, [
            ("Prix", f"{g['price']:.2f}"), ("YTM", f"{g['ytm']*100:.3f}%"),
            ("Duration", f"{g['duration']:.2f}"), ("DV01", f"{g['dv01']:.4f}"),
        ]):
            col.metric(label, val)

    # Swap de Taux 
    elif product_type == "Swap de Taux":
        c1, c2, c3 = st.columns(3)
        N = c1.number_input("Notionnel", value=1_000_000.0, format="%.0f")
        fr = c2.number_input("Taux fixe (%)", value=3.5) / 100
        T = c3.number_input("Maturité (années)", value=5.0)
        c1, c2 = st.columns(2)
        freq = c1.selectbox("Fréquence de paiement", [1, 2, 4], index=1)
        pay_fixed = c2.radio("Sens", ["Payeur fixe", "Receveur fixe"]) == "Payeur fixe"
        prod = InterestRateSwap(N, fr, T, freq, pay_fixed, rc)
        g = prod.greeks()
        c1, c2, c3 = st.columns(3)
        c1.metric("Valeur", f"{g['price']:,.0f}")
        c2.metric("Taux pari (par rate)", f"{g['par_rate']*100:.3f}%")
        c3.metric("DV01", f"{g['dv01']:,.0f}")

    #  Tracker Certificate 
    elif product_type == "Tracker Certificate (1100)":
        explain("<b>SSPA 1100</b> - réplication linéaire du sous-jacent. "
                "Composition : participation x Call(K proche de zéro).")
        c1, c2 = st.columns(2)
        T = c1.number_input("Maturité (années)", value=2.0, step=0.5, key="t_trk")
        part = c2.slider("Taux de participation", 0.5, 2.0, 1.0, step=0.05)
        prod = TrackerCertificate(S, T, r, participation=part,
                                   rate_curve=rc, vol_surface=vol_surface)
        g = prod.greeks()
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix", f"{g['price']:.2f}")
        c2.metric("Delta", f"{g['delta']:.3f}")
        c3.metric("Vega", f"{g['vega']:.3f}")

    # Bonus Certificate 
    elif product_type == "Bonus Certificate (1130)":
        explain("<b>SSPA 1130</b> - niveau bonus garanti à maturité tant que la "
                "barrière n'est pas touchée. Composition : Call(K=0) + Put(K=bonus) - Put_KI.")
        c1, c2, c3 = st.columns(3)
        T = c1.number_input("Maturité (années)", value=2.0, step=0.5, key="t_bon")
        bn = c2.number_input("Niveau bonus", value=float(round(S * 1.1)))
        bb = c3.number_input("Barrière", value=float(round(S * 0.7)))
        heston = HestonModel() if vol_model_choice == "Heston" else None
        prod = BonusCertificate(S, T, r, barrier=bb, bonus_level=bn,
                                 vol_surface=vol_surface, heston=heston)
        with st.spinner("Pricing Monte Carlo..."):
            p = prod.price(n_simulations=n_sims, seed=mc_seed)
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix", f"{p:.2f}")
        c2.metric("Niveau bonus", f"{bn:.0f}")
        c3.metric("Barrière", f"{bb:.0f}")

    # Capped Capital Protection 
    elif product_type == "Capped Capital Protection (1220)":
        explain("<b>SSPA 1220</b> - capital garanti à maturité par le ZCB, participation "
                "à la hausse plafonnée par le cap. "
                "Composition : ZCB + participation x CallSpread(strike, cap).")
        c1, c2, c3 = st.columns(3)
        T = c1.number_input("Maturité (années)", value=3.0, step=0.5, key="t_ccp")
        K = c2.number_input("Strike", value=float(round(S)))
        cap = c3.number_input("Cap", value=float(round(S * 1.2)))
        c1, c2 = st.columns(2)
        nominal = c1.number_input("Nominal", value=100.0)
        part = c2.slider("Taux de participation", 0.1, 2.0, 1.0, step=0.1, key="p_ccp")
        prod = CappedCapitalProtection(S, T, r, strike=K, cap=cap,
                                        participation=part, nominal=nominal,
                                        rate_curve=rc, vol_surface=vol_surface)
        g = prod.greeks()
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix", f"{g['price']:.2f}")
        c2.metric("Delta", f"{g['delta']:.3f}")
        c3.metric("Capital protégé", f"{nominal:.0f}")

    # Reverse Convertible 
    elif product_type == "Reverse Convertible (1320)":
        explain("<b>SSPA 1320</b> - coupon élevé, capital exposé à la baisse via un "
                "short put barrière. Composition : CouponBond - Put_KI.")
        c1, c2, c3 = st.columns(3)
        T = c1.number_input("Maturité (années)", value=2.0, step=0.5, key="t_rc")
        K = c2.number_input("Strike (Barrière 2)", value=float(round(S)))
        bb = c3.number_input("Barrière (Barrière 1)", value=float(round(S * 0.7)))
        c1, c2 = st.columns(2)
        cr = c1.number_input("Coupon (%)", value=8.0, step=0.5) / 100
        nominal = c2.number_input("Nominal", value=100.0, key="n_rc")
        heston = HestonModel() if vol_model_choice == "Heston" else None
        prod = ReverseConvertible(S, T, r, strike=K, barrier=bb,
                                    coupon_rate=cr, frequency=2, nominal=nominal,
                                    rate_curve=rc, vol_surface=vol_surface, heston=heston)
        with st.spinner("Pricing Monte Carlo..."):
            p = prod.price(n_simulations=n_sims, seed=mc_seed)
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix", f"{p:.2f}")
        c2.metric("Coupon", f"{cr*100:.1f}%")
        c3.metric("Barrière", f"{bb:.0f}")

# 3. PORTEFEUILLES DE DÉMONSTRATION
with tabs[2]:
    st.title("Portefeuilles de démonstration")
    explain("Quatre portefeuilles préconstruits illustrent les capacités du pricer : "
            "<b>Taux</b>, <b>Vanilles</b>, <b>Exotiques</b>, <b>Mixte</b>.")

    with st.spinner("Calcul des portefeuilles..."):
        portfolios = get_default_portfolios(ticker, country, options_date)

    sel = st.selectbox("Portefeuille", list(portfolios.keys()))
    port = portfolios[sel]
    summary = port.summary()
    report = port.risk_report()

    cols = st.columns(5)
    cols[0].metric("Valeur totale", f"{summary['total_value']:,.2f}")
    cols[1].metric("Delta", f"{summary['total_delta']:.4f}")
    cols[2].metric("Gamma", f"{summary['total_gamma']:.6f}")
    cols[3].metric("Vega", f"{summary['total_vega']:.4f}")
    cols[4].metric("DV01", f"{summary['total_dv01']:.4f}")

    st.divider()
    st.markdown("#### Détail des positions")
    cols = ["label", "type", "quantity", "unit_price", "total_value"]
    for greek in ("delta", "gamma", "vega", "theta", "dv01"):
        if greek in report.columns:
            cols.append(greek)
    st.dataframe(report[cols].style.format({
        "unit_price": "{:.4f}", "total_value": "{:,.2f}",
        "delta": "{:.4f}", "gamma": "{:.6f}", "vega": "{:.4f}",
        "theta": "{:.4f}", "dv01": "{:.4f}",
    }), use_container_width=True)

    fig = px.bar(report, x="label", y="total_value", color="type",
                 height=350, color_discrete_sequence=COLORS.CATEGORICAL)
    fig.update_layout(**chart_layout(
        xaxis_title="Position", yaxis_title="Valeur", height=350,
    ))
    fig.update_xaxes(tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

# 4. INVENTAIRE
with tabs[3]:
    st.title("Inventaire de portefeuille")
    explain("Chargement du fichier <code>Inventaire.xlsx</code> (4 feuilles : Swap, Options, "
            "Autocall, Notes structurées). Chaque ligne est convertie en produit du pricer "
            "et valorisée sur la base des données de marché courantes.")

    with st.spinner("Construction des portefeuilles depuis l'inventaire..."):
        try:
            inv_ports, inv_loader = get_inventory_portfolios(ticker, country, options_date)
            inv_ok = True
        except Exception as e:
            st.error(f"Erreur de chargement de l'inventaire : {e}")
            inv_ok = False

    if inv_ok:
        rows = []
        for n, p in inv_ports.items():
            s = p.summary()
            rows.append({"Feuille": n, "Positions": s["n_positions"],
                         "Valeur": s["total_value"], "Delta": s["total_delta"],
                         "Vega": s["total_vega"], "DV01": s["total_dv01"]})
        st.markdown("### Synthèse par feuille")
        st.dataframe(pd.DataFrame(rows).style.format({
            "Valeur": "{:,.2f}", "Delta": "{:.4f}",
            "Vega": "{:.4f}", "DV01": "{:.4f}",
        }), use_container_width=True)

        st.markdown("### Données brutes Excel")
        sheet_choice = st.selectbox("Feuille", list(inv_ports.keys()), key="sheet_raw")
        raw_map = {"Swaps": "Swap", "Options & Stratégies": "Options",
                   "Autocalls": "Autocall", "Notes structurées": "Notes structurées"}
        raw = inv_loader.get_sheet(raw_map[sheet_choice])
        st.dataframe(raw, use_container_width=True)

        st.markdown("### Positions valorisées")
        port_choice = st.selectbox("Portefeuille", list(inv_ports.keys()), key="sheet_pos")
        port = inv_ports[port_choice]
        report = port.risk_report()
        if not report.empty:
            cols = ["label", "type", "quantity", "unit_price", "total_value"]
            for greek in ("delta", "gamma", "vega", "dv01"):
                if greek in report.columns:
                    cols.append(greek)
            st.dataframe(report[cols].style.format({
                "unit_price": "{:.4f}", "total_value": "{:,.2f}",
                "delta": "{:.4f}", "gamma": "{:.6f}",
                "vega": "{:.4f}", "dv01": "{:.4f}",
            }), use_container_width=True)

            fig_inv = px.bar(report, x="label", y="total_value", color="type",
                              height=350,
                              color_discrete_sequence=COLORS.CATEGORICAL,
                              title=f"Valeur par position - {port_choice}")
            fig_inv.update_layout(**chart_layout(
                xaxis_title="Position", yaxis_title="Valeur", height=350,
            ))
            fig_inv.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_inv, use_container_width=True)

# 5. RISQUES
with tabs[4]:
    st.title("Analyse des risques")
    explain("Décomposition de l'exposition par <b>pilier de maturité</b> et "
            "<b>bucket de moneyness</b>. Attribution de P&L au premier ordre : "
            "ΔV ≈ θ·Δt + Δ·ΔS + ½·Γ·ΔS² + ν·Δσ + ρ·Δr.")

    source = st.radio("Source", ["Portefeuilles démo", "Inventaire"], horizontal=True)
    if source == "Portefeuilles démo":
        ports_for_risk = portfolios
    else:
        if inv_ok:
            ports_for_risk = inv_ports
        else:
            ports_for_risk = portfolios
    sel_r = st.selectbox("Portefeuille à analyser", list(ports_for_risk.keys()), key="risk_sel")
    pr = ports_for_risk[sel_r]

    st.markdown("### Risque par pilier de maturité")
    bdf = pr.bucketed_risk()
    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(
        x=bdf["maturity_bucket"], y=bdf["dv01"], name="DV01",
        marker=dict(color=COLORS.BLUE),
    ))
    fig_b.add_trace(go.Bar(
        x=bdf["maturity_bucket"], y=bdf["delta"], name="Delta",
        marker=dict(color=COLORS.ORANGE), yaxis="y2",
    ))
    fig_b.update_layout(**chart_layout(
        xaxis_title="Pilier de maturité",
        yaxis=dict(title="DV01", color=COLORS.TEXT_MUTED, gridcolor=COLORS.GRID),
        yaxis2=dict(title="Delta", overlaying="y", side="right",
                    color=COLORS.TEXT_MUTED, gridcolor=COLORS.GRID),
        barmode="group", height=320,
    ))
    st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("### Risque par bucket de moneyness (K/S)")
    sdf = pr.bucketed_strike_risk()
    fig_s = go.Figure()
    for col, color, name in [
        ("delta", COLORS.ORANGE, "Delta"),
        ("gamma", COLORS.GREEN,  "Gamma"),
        ("vega",  COLORS.PURPLE, "Vega"),
    ]:
        fig_s.add_trace(go.Bar(
            x=sdf["strike_bucket"], y=sdf[col], name=name,
            marker=dict(color=color),
        ))
    fig_s.update_layout(**chart_layout(
        xaxis_title="Bucket de moneyness", yaxis_title="Sensibilité",
        barmode="group", height=320,
    ))
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("### Attribution de P&L")
    explain("Saisissez un mouvement de marché : la décomposition par grec apparaît ci-dessous.")
    c1, c2, c3, c4 = st.columns(4)
    ds = c1.number_input("Variation du spot (ΔS)", value=1.0, step=0.5)
    dr = c2.number_input("Variation des taux (Δr en bp)", value=10.0, step=1.0) / 10000
    dvol = c3.number_input("Variation de la vol (Δσ en %)", value=1.0, step=0.5) / 100
    dt = c4.number_input("Écoulement du temps (jours)", value=1.0, step=1.0) / 365

    pnl = pr.pnl_attribution(ds=ds, dr=dr, dsigma=dvol, dt=dt)
    contribs = [
        ("Theta", pnl["theta_pnl"], COLORS.AXIS),
        ("Delta", pnl["delta_pnl"], COLORS.ORANGE),
        ("Gamma", pnl["gamma_pnl"], COLORS.GREEN),
        ("Vega", pnl["vega_pnl"], COLORS.PURPLE),
        ("Rho", pnl["rho_pnl"], COLORS.RED),
        ("DV01", pnl["dv01_pnl"], COLORS.BLUE),
    ]
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(
        x=[c[0] for c in contribs], y=[c[1] for c in contribs],
        marker=dict(color=[c[2] for c in contribs]),
        text=[f"{c[1]:+,.2f}" for c in contribs],
        textposition="outside",
    ))
    fig_p.update_layout(**chart_layout(
        height=320, yaxis_title="Contribution au P&L",
    ))
    st.plotly_chart(fig_p, use_container_width=True)
    st.metric("P&L total", f"{pnl['total_pnl']:,.2f}")

# 6. CALIBRATION
with tabs[5]:
    st.title("Calibration des modèles")
    explain("Calibration des modèles stochastiques sur les données historiques (taux) "
            "et la surface implicite (Heston).")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Modèles de taux")
        if st.button("Calibrer Vasicek", width='stretch'):
            from pricer.data import load_rate_curves
            df = load_rate_curves(country=country)
            three_m = df[df["maturity"] == "3M"].sort_values("date")
            v = VasicekModel().calibrate(three_m["rate"].values / 100)
            st.success(f"a = {v.a:.4f}, k = {v.k*100:.3f}%, σ = {v.sigma*100:.3f}%")

        if st.button("Calibrer CIR", width='stretch'):
            from pricer.data import load_rate_curves
            df = load_rate_curves(country=country)
            three_m = df[df["maturity"] == "3M"].sort_values("date")
            with st.spinner("Calibration CIR par différentielle évolutive..."):
                cir = CIRModel().calibrate(three_m["rate"].values / 100)
            feller_ok = 2 * cir.a * cir.k > cir.sigma ** 2
            st.success(f"a = {cir.a:.4f}, k = {cir.k*100:.3f}%, σ = {cir.sigma*100:.3f}% "
                       f"(condition de Feller : {'respectée' if feller_ok else 'violée'})")

    with c2:
        st.markdown("### Modèle Heston")
        if st.button("Calibrer Heston", width='stretch'):
            with st.spinner("Calibration Heston (DE puis L-BFGS-B)..."):
                h = HestonModel().calibrate(vol_surface, r=r)
            st.success(f"v0 = {h.v0:.4f}, κ = {h.kappa:.3f}, θ = {h.theta:.4f}, "
                       f"ζ = {h.zeta:.3f}, ρ = {h.rho:.3f}")

            df_iv = vol_surface.get_surface_dataframe()
            sample = df_iv.sample(min(30, len(df_iv)), random_state=42)
            sample["iv_heston"] = sample.apply(
                lambda x: h.implied_vol_from_heston(S, x["strike"], x["maturity"], r) * 100,
                axis=1,
            )
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=sample["iv"], y=sample["iv_heston"], mode="markers",
                marker=dict(color=COLORS.BLUE, size=8),
                name="Points calibrés",
            ))
            lo, hi = sample["iv"].min(), sample["iv"].max()
            fig_h.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines",
                line=dict(color=COLORS.AXIS, dash="dash"),
                name="Diagonale (fit parfait)",
            ))
            fig_h.update_layout(**chart_layout(
                xaxis_title="Volatilité de marché (%)",
                yaxis_title="Volatilité Heston (%)",
                height=300,
            ))
            st.plotly_chart(fig_h, use_container_width=True)
