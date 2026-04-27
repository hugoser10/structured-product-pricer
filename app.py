"""
Pricer Multi-Produits — Application Streamlit
Onglets : Marché · Pricing · Portefeuilles démo · Inventaire (xlsx) · Risques · Calibration
"""

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
)
from pricer.portfolio import Portfolio, InventoryLoader, build_portfolios


# ---------------------------------------------------------------------------
# Configuration et thème
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Pricer Multi-Produits", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .explain-box { background: #0f172a; border-left: 3px solid #38bdf8;
                   padding: 10px 14px; border-radius: 4px; font-size: 0.88rem;
                   color: #94a3b8; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)


def explain(text: str):
    st.markdown(f'<div class="explain-box">{text}</div>', unsafe_allow_html=True)


def chart_layout(**kwargs) -> dict:
    base = dict(
        template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0", size=13),
        legend=dict(bgcolor="rgba(15,23,42,0.85)", bordercolor="#334155",
                    borderwidth=1, font=dict(color="#e2e8f0", size=12)),
        xaxis=dict(color="#94a3b8", gridcolor="#1e293b", zerolinecolor="#334155"),
        yaxis=dict(color="#94a3b8", gridcolor="#1e293b", zerolinecolor="#334155"),
        margin=dict(l=50, r=20, t=40, b=50),
    )
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Paramètres globaux")
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
    st.caption("Master 272 · Pricing de produits structurés (2026)")


# ---------------------------------------------------------------------------
# Chargement des données de marché
# ---------------------------------------------------------------------------
with st.spinner("Chargement des données de marché…"):
    try:
        mkt = get_market_data(ticker, country, options_date)
        rc = mkt["rate_curve"]
        vol_surface = mkt["vol_surface"]
        S, r = mkt["spot"], mkt["r"]
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        st.stop()


# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------
tabs = st.tabs(["📊 Marché", "🧮 Pricing", "💼 Portefeuilles démo",
                "📂 Inventaire", "⚠️ Risques", "🔧 Calibration"])


# ===========================================================================
# 1. MARCHÉ
# ===========================================================================
with tabs[0]:
    st.title("📊 Données de marché")
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
    fig.add_trace(go.Scatter(x=cd["maturity"], y=cd["rate"], mode="lines",
                             name="NSS", line=dict(color="#38bdf8", width=2.5)))
    mats_obs = sorted(MATURITY_MAP.values())
    rates_obs = [rc.zero_rate(m) * 100 for m in mats_obs]
    fig.add_trace(go.Scatter(x=mats_obs, y=rates_obs, mode="markers",
                             name="Marché", marker=dict(color="white", size=6)))
    fig.update_layout(**chart_layout(xaxis_title="Maturité (ans)", yaxis_title="Taux (%)",
                                      height=350))
    st.plotly_chart(fig, use_container_width=True)

    # Surface de vol
    st.markdown("### Surface de volatilité implicite")
    explain("Volatilités obtenues par <b>inversion BSM Newton-Raphson</b>. "
            "Convention : puts pour K&lt;S, calls pour K&gt;S.")
    try:
        vol_df = vol_surface.get_surface_dataframe()
        if vol_df["maturity"].nunique() >= 2:
            pivot = vol_df.pivot_table(index="maturity", columns="moneyness",
                                        values="iv", aggfunc="mean").interpolate(axis=1).interpolate(axis=0)
            fig_s = go.Figure(data=[go.Surface(
                x=pivot.columns.values, y=pivot.index.values, z=pivot.values,
                colorscale="Plasma",
                colorbar=dict(title="Vol (%)", tickfont=dict(color="#e2e8f0")),
            )])
            fig_s.update_layout(**chart_layout(
                scene=dict(xaxis=dict(title="Moneyness"),
                           yaxis=dict(title="Maturité"),
                           zaxis=dict(title="Vol (%)"),
                           bgcolor="#0d1526",
                           camera=dict(eye=dict(x=1.8, y=-1.5, z=0.8))),
                height=480, title=f"Surface — {ticker}",
            ))
            st.plotly_chart(fig_s, use_container_width=True)

        # Smile
        st.markdown("### Smile par maturité")
        fig_sm = go.Figure()
        colors = ["#38bdf8", "#f59e0b", "#22c55e", "#e879f9", "#fb923c"]
        for i, T in enumerate(sorted(vol_df["maturity"].unique())):
            sub = vol_df[vol_df["maturity"] == T].sort_values("moneyness")
            fig_sm.add_trace(go.Scatter(
                x=sub["moneyness"], y=sub["iv"], mode="lines+markers",
                name=f"DTE={int(T*365)}j",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig_sm.add_vline(x=1.0, line_dash="dash", line_color="#64748b")
        fig_sm.update_layout(**chart_layout(xaxis_title="Moneyness", yaxis_title="Vol (%)",
                                              height=400))
        st.plotly_chart(fig_sm, use_container_width=True)
    except Exception as e:
        st.warning(f"Affichage surface : {e}")


# ===========================================================================
# 2. PRICING INTERACTIF
# ===========================================================================
with tabs[1]:
    st.title("🧮 Pricer interactif")
    explain("Sélectionnez un type de produit, ajustez les paramètres et obtenez "
            "instantanément prix + greeks.")

    product_type = st.selectbox("Type de produit", [
        "Option Européenne", "Call Spread", "Put Spread", "Butterfly",
        "Option à Barrière", "Autocall",
        "Obligation Zéro-Coupon", "Obligation à Coupons", "Swap de Taux",
    ])
    st.divider()

    if product_type == "Option Européenne":
        c1, c2, c3 = st.columns(3)
        K = c1.number_input("Strike", value=float(round(S)), step=1.0)
        side = c1.radio("Type", ["call", "put"])
        T = c2.number_input("Maturité (ans)", value=1.0, step=0.25, min_value=0.01)
        q = c2.number_input("Dividende q", value=0.0, step=0.001, format="%.3f")
        use_surf = c3.checkbox("Surface implicite", value=True)
        sig_m = c3.number_input("Vol manuelle", value=0.20, step=0.01, format="%.2f")

        heston = HestonModel() if vol_model_choice == "Heston" else None
        opt = Option(S, K, T, r, side, sig_m, q,
                     vol_surface=vol_surface if use_surf else None, heston=heston)
        g = opt.greeks()

        cols = st.columns(6)
        for col, (n, v) in zip(cols, [
            ("Prix", f"{g['price']:.4f}"), ("Delta", f"{g['delta']:.4f}"),
            ("Gamma", f"{g['gamma']:.6f}"), ("Vega", f"{g['vega']:.4f}"),
            ("Theta/j", f"{g['theta']:.4f}"), ("Rho", f"{g['rho']:.4f}"),
        ]):
            col.metric(n, v)

        # Payoff
        K_range = np.linspace(S * 0.5, S * 1.5, 200)
        payoffs = (np.maximum(K_range - K, 0) if side == "call"
                   else np.maximum(K - K_range, 0))
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=K_range, y=payoffs, name="Payoff",
                                    line=dict(color="#94a3b8", dash="dash")))
        fig_p.add_trace(go.Scatter(x=K_range, y=payoffs - g["price"], name="P&L net",
                                    line=dict(color="#22d3ee", width=2)))
        fig_p.add_vline(x=K, line_dash="dot", annotation_text="K")
        fig_p.add_vline(x=S, line_color="green", line_dash="dash", annotation_text="Spot")
        fig_p.add_hline(y=0, line_color="#475569", opacity=0.6)
        fig_p.update_layout(**chart_layout(xaxis_title="S à maturité", yaxis_title="P&L",
                                            height=350))
        st.plotly_chart(fig_p, use_container_width=True)

    elif product_type in ("Call Spread", "Put Spread"):
        c1, c2 = st.columns(2)
        K1 = c1.number_input("K1 (bas)", value=float(round(S * 0.95, -1)), step=1.0)
        K2 = c2.number_input("K2 (haut)", value=float(round(S * 1.1, -1)), step=1.0)
        T = st.number_input("Maturité (ans)", value=1.0, step=0.25)
        prod = (CallSpread(S, K1, K2, T, r, vol_surface=vol_surface)
                if product_type == "Call Spread"
                else PutSpread(S, K1, K2, T, r, vol_surface=vol_surface))
        g = prod.greeks()
        cols = st.columns(5)
        for col, (n, v) in zip(cols, [
            ("Prix", f"{g['price']:.4f}"), ("Delta", f"{g['delta']:.4f}"),
            ("Gamma", f"{g['gamma']:.6f}"), ("Vega", f"{g['vega']:.4f}"),
            ("Theta/j", f"{g['theta']:.4f}"),
        ]):
            col.metric(n, v)
        K_range = np.linspace(S * 0.5, S * 1.5, 300)
        payoffs = (np.clip(K_range - K1, 0, K2 - K1) if product_type == "Call Spread"
                   else np.clip(K2 - K_range, 0, K2 - K1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=K_range, y=payoffs, name="Payoff",
                                  line=dict(dash="dash", color="#94a3b8")))
        fig.add_trace(go.Scatter(x=K_range, y=payoffs - g["price"], name="P&L net",
                                  line=dict(color="#22d3ee", width=2)))
        fig.add_hline(y=0, line_color="#475569", opacity=0.6)
        fig.add_vline(x=S, line_color="green", line_dash="dash", annotation_text="Spot")
        fig.update_layout(**chart_layout(height=300))
        st.plotly_chart(fig, use_container_width=True)

    elif product_type == "Butterfly":
        c1, c2, c3 = st.columns(3)
        K1 = c1.number_input("K1 (bas)", value=float(round(S * 0.9, -1)))
        K2 = c2.number_input("K2 (centre)", value=float(round(S, -1)))
        K3 = c3.number_input("K3 (haut)", value=float(round(S * 1.1, -1)))
        T = st.number_input("Maturité", value=1.0, step=0.25)
        prod = Butterfly(S, K1, K2, K3, T, r, vol_surface=vol_surface)
        g = prod.greeks()
        cols = st.columns(4)
        for col, (n, v) in zip(cols, [
            ("Prix", f"{g['price']:.4f}"), ("Delta", f"{g['delta']:.4f}"),
            ("Gamma", f"{g['gamma']:.6f}"), ("Vega", f"{g['vega']:.4f}"),
        ]):
            col.metric(n, v)

    elif product_type == "Option à Barrière":
        c1, c2, c3 = st.columns(3)
        K = c1.number_input("Strike", value=float(round(S)))
        side = c1.radio("Type", ["call", "put"])
        T = c2.number_input("Maturité", value=1.0, step=0.25)
        bt = c2.selectbox("Barrière", ["down-and-out", "up-and-out", "down-and-in", "up-and-in"])
        H_default = S * 0.8 if "down" in bt else S * 1.2
        H = c3.number_input("Niveau H", value=float(round(H_default)))
        rebate = c3.number_input("Rebate", value=0.0)

        heston = HestonModel() if vol_model_choice == "Heston" else None
        sig = vol_surface.get_vol(K, T)
        with st.spinner("Pricing Monte Carlo…"):
            prod = BarrierOption(S, K, T, r, H, bt, side, rebate, sig, 0.0,
                                  heston, vol_surface if not heston else None)
            p = prod.price(n_simulations=n_sims, seed=mc_seed)
            van = Option(S, K, T, r, side, sigma=sig, vol_surface=vol_surface).price()
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix barrière", f"{p:.4f}")
        c2.metric("Vanille équivalent", f"{van:.4f}")
        c3.metric("Décote barrière", f"{van - p:.4f}")

    elif product_type == "Autocall":
        c1, c2 = st.columns(2)
        T = c1.number_input("Maturité (ans)", value=5.0, step=1.0)
        coupon = c1.number_input("Coupon (%)", value=5.0, step=0.5)
        bf = c1.slider("Barrière finale (%)", 30, 90, 50) / 100
        n_obs = c2.selectbox("Observations/an", [1, 2, 4], index=0)
        sd = c2.checkbox("Step-down", value=False)
        sd_pct = c2.slider("Baisse annuelle (%)", 0, 15, 5) if sd else 0

        total_obs = int(T * n_obs)
        bcall = {i + 1: max(1.0 - sd_pct/100 * ((i+1)/n_obs - 1), 0.5) if sd else 1.0
                 for i in range(total_obs)}
        heston = HestonModel() if vol_model_choice == "Heston" else None
        sig = vol_surface.get_atm_vol(T)
        with st.spinner("Monte Carlo autocall…"):
            prod = Autocall(S, S, coupon/100, bcall, bf, T, n_obs, r, sig, 0.0, heston)
            price = prod.price(n_simulations=n_sims, seed=mc_seed)
            opt_c = prod.optimal_coupon(target_price=1.0, n_simulations=n_sims)
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix (% nominal)", f"{price*100:.2f}%")
        c2.metric("Coupon optimal", f"{opt_c*100:.2f}%")
        c3.metric("Barrière finale", f"{bf*100:.0f}%")

    elif product_type == "Obligation Zéro-Coupon":
        c1, c2 = st.columns(2)
        N = c1.number_input("Nominal", value=1000.0)
        T = c2.number_input("Maturité", value=5.0)
        prod = ZeroCouponBond(N, T, rc)
        g = prod.greeks()
        c1, c2, c3 = st.columns(3)
        c1.metric("Prix", f"{g['price']:.2f}")
        c2.metric("Taux zéro", f"{g['zero_rate']*100:.3f}%")
        c3.metric("DV01", f"{g['dv01']:.4f}")

    elif product_type == "Obligation à Coupons":
        c1, c2, c3 = st.columns(3)
        N = c1.number_input("Nominal", value=1000.0)
        cr = c2.number_input("Coupon (%)", value=4.0) / 100
        T = c3.number_input("Maturité", value=5.0)
        freq = st.selectbox("Fréquence", [1, 2, 4, 12], index=1)
        prod = CouponBond(N, cr, T, freq, rc)
        g = prod.greeks()
        cols = st.columns(4)
        for col, (n, v) in zip(cols, [
            ("Prix", f"{g['price']:.2f}"), ("YTM", f"{g['ytm']*100:.3f}%"),
            ("Duration", f"{g['duration']:.2f}"), ("DV01", f"{g['dv01']:.4f}"),
        ]):
            col.metric(n, v)

    elif product_type == "Swap de Taux":
        c1, c2, c3 = st.columns(3)
        N = c1.number_input("Notionnel", value=1_000_000.0, format="%.0f")
        fr = c2.number_input("Taux fixe (%)", value=3.5) / 100
        T = c3.number_input("Maturité", value=5.0)
        c1, c2 = st.columns(2)
        freq = c1.selectbox("Fréquence", [1, 2, 4], index=1)
        pay_fixed = c2.radio("Sens", ["Payeur fixe", "Receveur fixe"]) == "Payeur fixe"
        prod = InterestRateSwap(N, fr, T, freq, pay_fixed, rc)
        g = prod.greeks()
        c1, c2, c3 = st.columns(3)
        c1.metric("Valeur", f"{g['price']:,.0f}")
        c2.metric("Par rate", f"{g['par_rate']*100:.3f}%")
        c3.metric("DV01", f"{g['dv01']:,.0f}")


# ===========================================================================
# 3. PORTEFEUILLES DE DÉMO
# ===========================================================================
with tabs[2]:
    st.title("💼 Portefeuilles de démonstration")
    explain("Quatre portefeuilles préconstruits : <b>Taux</b>, <b>Vanilles</b>, "
            "<b>Exotiques</b>, <b>Mixte</b>.")

    with st.spinner("Calcul…"):
        portfolios = get_default_portfolios(ticker, country, options_date)

    sel = st.selectbox("Portefeuille", list(portfolios.keys()))
    port = portfolios[sel]
    summary = port.summary()
    report = port.risk_report()

    cols = st.columns(5)
    cols[0].metric("Valeur", f"{summary['total_value']:,.2f}")
    cols[1].metric("Delta", f"{summary['total_delta']:.4f}")
    cols[2].metric("Gamma", f"{summary['total_gamma']:.6f}")
    cols[3].metric("Vega", f"{summary['total_vega']:.4f}")
    cols[4].metric("DV01", f"{summary['total_dv01']:.4f}")

    st.divider()
    st.markdown("#### Positions")
    cols = ["label", "type", "quantity", "unit_price", "total_value"]
    for g in ("delta", "gamma", "vega", "theta", "dv01"):
        if g in report.columns:
            cols.append(g)
    st.dataframe(report[cols].style.format({
        "unit_price": "{:.4f}", "total_value": "{:,.2f}",
        "delta": "{:.4f}", "gamma": "{:.6f}", "vega": "{:.4f}",
        "theta": "{:.4f}", "dv01": "{:.4f}",
    }), use_container_width=True)

    fig = px.bar(report, x="label", y="total_value", color="type",
                 height=300, color_discrete_sequence=["#38bdf8", "#f59e0b", "#22c55e",
                                                       "#e879f9", "#fb923c"])
    fig.update_layout(**chart_layout(height=300))
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# 4. INVENTAIRE.XLSX
# ===========================================================================
with tabs[3]:
    st.title("📂 Inventaire de portefeuille")
    explain("Chargement du fichier <code>Inventaire.xlsx</code> (4 feuilles : Swap, Options, "
            "Autocall, Notes structurées). Chaque ligne est convertie en produit du pricer "
            "et valorisée sur la base des données de marché courantes.")

    with st.spinner("Construction des portefeuilles d'inventaire…"):
        try:
            inv_ports, inv_loader = get_inventory_portfolios(ticker, country, options_date)
            inv_ok = True
        except Exception as e:
            st.error(f"Erreur inventaire : {e}")
            inv_ok = False

    if inv_ok:
        # Vue d'ensemble
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

        # Données brutes par feuille
        st.markdown("### Données brutes Excel")
        sheet_choice = st.selectbox("Feuille", list(inv_ports.keys()), key="sheet_raw")
        raw_map = {"Swaps": "Swap", "Options & Stratégies": "Options",
                   "Autocalls": "Autocall", "Notes structurées": "Notes structurées"}
        raw = inv_loader.get_sheet(raw_map[sheet_choice])
        st.dataframe(raw, use_container_width=True)

        # Détail des positions valorisées
        st.markdown("### Positions valorisées")
        port_choice = st.selectbox("Portefeuille", list(inv_ports.keys()), key="sheet_pos")
        port = inv_ports[port_choice]
        report = port.risk_report()
        if not report.empty:
            cols = ["label", "type", "quantity", "unit_price", "total_value"]
            for g in ("delta", "gamma", "vega", "dv01"):
                if g in report.columns:
                    cols.append(g)
            st.dataframe(report[cols].style.format({
                "unit_price": "{:.4f}", "total_value": "{:,.2f}",
                "delta": "{:.4f}", "gamma": "{:.6f}",
                "vega": "{:.4f}", "dv01": "{:.4f}",
            }), use_container_width=True)

            fig_inv = px.bar(report, x="label", y="total_value", color="type",
                              height=350, title=f"Valeur par position — {port_choice}")
            fig_inv.update_layout(**chart_layout(height=350))
            fig_inv.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_inv, use_container_width=True)


# ===========================================================================
# 5. RISQUES
# ===========================================================================
with tabs[4]:
    st.title("⚠️ Analyse des risques")
    explain("Décomposition de l'exposition par <b>pilier de maturité × strike</b> "
            "et attribution de P&L : ΔV ≈ θ·Δt + Δ·ΔS + ½·Γ·ΔS² + ν·Δσ + ρ·Δr.")

    source = st.radio("Source", ["Portefeuilles démo", "Inventaire"], horizontal=True)
    if source == "Portefeuilles démo":
        ports_for_risk = portfolios
    else:
        if inv_ok:
            ports_for_risk = inv_ports
        else:
            ports_for_risk = portfolios
    sel_r = st.selectbox("Portefeuille analysé", list(ports_for_risk.keys()), key="risk_sel")
    pr = ports_for_risk[sel_r]

    st.markdown("### Risque par pilier de maturité")
    bdf = pr.bucketed_risk()
    fig_b = go.Figure()
    fig_b.add_trace(go.Bar(x=bdf["maturity_bucket"], y=bdf["dv01"], name="DV01",
                            marker=dict(color="#38bdf8")))
    fig_b.add_trace(go.Bar(x=bdf["maturity_bucket"], y=bdf["delta"], name="Delta",
                            marker=dict(color="#f59e0b"), yaxis="y2"))
    fig_b.update_layout(**chart_layout(
        xaxis_title="Maturité", yaxis=dict(title="DV01"),
        yaxis2=dict(title="Delta", overlaying="y", side="right",
                    color="#94a3b8", gridcolor="#1e293b"),
        barmode="group", height=320,
    ))
    st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("### Risque par strike (moneyness K/S)")
    sdf = pr.bucketed_strike_risk()
    fig_s = go.Figure()
    for col, color in [("delta", "#f59e0b"), ("gamma", "#22c55e"), ("vega", "#e879f9")]:
        fig_s.add_trace(go.Bar(x=sdf["strike_bucket"], y=sdf[col], name=col.capitalize(),
                                marker=dict(color=color)))
    fig_s.update_layout(**chart_layout(barmode="group", height=320,
                                         xaxis_title="Bucket moneyness"))
    st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("### Attribution P&L")
    explain("Simulation d'un mouvement combiné (spot, taux, vol) — décomposition par grec.")
    c1, c2, c3, c4 = st.columns(4)
    ds = c1.number_input("ΔS", value=1.0, step=0.5)
    dr = c2.number_input("Δr (en bp)", value=10.0, step=1.0) / 10000
    dvol = c3.number_input("Δσ (%)", value=1.0, step=0.5) / 100
    dt = c4.number_input("Δt (jours)", value=1.0, step=1.0) / 365

    pnl = pr.pnl_attribution(ds=ds, dr=dr, dsigma=dvol, dt=dt)
    fig_p = go.Figure()
    contribs = [("theta", pnl["theta_pnl"]), ("delta", pnl["delta_pnl"]),
                ("gamma", pnl["gamma_pnl"]), ("vega", pnl["vega_pnl"]),
                ("rho", pnl["rho_pnl"])]
    fig_p.add_trace(go.Bar(x=[c[0] for c in contribs], y=[c[1] for c in contribs],
                            marker=dict(color=["#94a3b8", "#f59e0b", "#22c55e",
                                                "#e879f9", "#fb923c"])))
    fig_p.update_layout(**chart_layout(height=300, yaxis_title="P&L"))
    st.plotly_chart(fig_p, use_container_width=True)
    st.metric("P&L total", f"{pnl['total_pnl']:,.2f}")


# ===========================================================================
# 6. CALIBRATION
# ===========================================================================
with tabs[5]:
    st.title("🔧 Calibration des modèles")
    explain("Calibration des modèles stochastiques sur les données historiques (taux) "
            "et la surface implicite (Heston).")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Modèles de taux")
        if st.button("Calibrer Vasicek", use_container_width=True):
            from pricer.data import load_rate_curves
            df = load_rate_curves(country=country)
            three_m = df[df["maturity"] == "3M"].sort_values("date")
            v = VasicekModel().calibrate(three_m["rate"].values / 100)
            st.success(f"a={v.a:.4f}, k={v.k*100:.3f}%, σ={v.sigma*100:.3f}%")
        if st.button("Calibrer CIR", use_container_width=True):
            from pricer.data import load_rate_curves
            df = load_rate_curves(country=country)
            three_m = df[df["maturity"] == "3M"].sort_values("date")
            with st.spinner("Calibration CIR (différentielle évolutive)…"):
                c = CIRModel().calibrate(three_m["rate"].values / 100)
            st.success(f"a={c.a:.4f}, k={c.k*100:.3f}%, σ={c.sigma*100:.3f}% "
                       f"(Feller={2*c.a*c.k > c.sigma**2})")

    with c2:
        st.markdown("### Modèle Heston")
        if st.button("Calibrer Heston", use_container_width=True):
            with st.spinner("Calibration Heston (DE puis L-BFGS-B)…"):
                h = HestonModel().calibrate(vol_surface, r=r)
            st.success(f"v0={h.v0:.4f}, κ={h.kappa:.3f}, θ={h.theta:.4f}, "
                       f"ζ={h.zeta:.3f}, ρ={h.rho:.3f}")
            # Comparaison vol marché vs Heston
            df_iv = vol_surface.get_surface_dataframe()
            sample = df_iv.sample(min(30, len(df_iv)), random_state=42)
            sample["iv_heston"] = sample.apply(
                lambda x: h.implied_vol_from_heston(S, x["strike"], x["maturity"], r) * 100,
                axis=1)
            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(x=sample["iv"], y=sample["iv_heston"],
                                        mode="markers", marker=dict(color="#38bdf8")))
            fig_h.add_trace(go.Scatter(x=[10, 80], y=[10, 80], mode="lines",
                                        line=dict(color="#94a3b8", dash="dash"),
                                        name="y=x"))
            fig_h.update_layout(**chart_layout(xaxis_title="Vol marché (%)",
                                                yaxis_title="Vol Heston (%)",
                                                height=300))
            st.plotly_chart(fig_h, use_container_width=True)
