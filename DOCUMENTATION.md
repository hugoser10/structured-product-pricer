# Pricer Multi-Produits — Documentation

## Architecture (un fichier par classe)

```
pricer/
├── models/
│   ├── rates/
│   │   ├── rate_curve.py             # RateCurve (NSS)
│   │   ├── stochastic_rate_model.py  # ABC
│   │   ├── vasicek_model.py          # OLS
│   │   ├── cir_model.py              # MLE (Feller)
│   │   └── hull_white_model.py       # fit courbe initiale
│   └── volatility/
│       ├── black_scholes.py          # bsm_price/greeks, implied_vol_newton
│       ├── implied_vol_surface.py    # surface (K,T) bicubique
│       └── heston_model.py           # Carr-Madan FFT + MC
├── products/
│   ├── base/                         # 7 classes abstraites
│   ├── rates/                        # 11 classes
│   ├── equity/                       # 9 classes
│   └── exotic/                       # 3 classes
├── portfolio/
│   ├── portfolio.py                  # Portfolio
│   ├── inventory_loader.py           # parse Inventaire.xlsx
│   └── default_portfolios.py         # 4 ptfs synthétiques
├── data/
│   ├── loader.py                     # build_market_data()
│   ├── rate_curves.parquet           # courbes historiques
│   ├── options.csv                   # panel d'options
│   └── Inventaire.xlsx               # 4 feuilles (Swap, Options, Autocall, Notes)
app.py                                # Streamlit (6 onglets)
requirements.txt
```

## Hiérarchie des produits

```
Product (ABC)
├── CompositeProduct (ABC)
│   ├── RateProduct       — CouponBond, FloatingRateBond, IRS, Cap, Floor,
│   │                       CallableBond, PutableBond
│   └── EquityProduct     — CallSpread, PutSpread, Butterfly,
│                           Straddle, Strangle, Strip, Strap
├── AtomicRateProduct     — ZeroCouponBond, Caplet, Swaption, BondOption
├── AtomicEquityProduct   — Option (BSM/Heston), DigitalOption
└── PathDependentProduct  — OneTouchOption, BarrierOption, Autocall
```

## Modèles

| Type | Modèle | Méthode |
|---|---|---|
| Taux statique | Nelson-Siegel-Svensson | minimisation OLS |
| Taux stochastique | Vasicek | OLS sur discrétisation |
| Taux stochastique | CIR | MLE (DE) avec contrainte Feller |
| Taux stochastique | Hull-White 1F | θ(t) = a·f(0,t) + ∂f/∂t + σ²/(2a)·(1−e^{−2at}) |
| Volatilité statique | Surface implicite | Newton-Raphson + spline bicubique |
| Volatilité stoch. | Heston | Carr-Madan FFT + MC Euler tronqué |

## Lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Onglets de l'application

1. **Marché** — courbe NSS + surface de vol 3D + smiles
2. **Pricing** — pricer interactif pour 9 types de produits
3. **Portefeuilles démo** — 4 portefeuilles synthétiques
4. **Inventaire** — chargement de `Inventaire.xlsx` et valorisation
5. **Risques** — bucketing par maturité × strike + attribution P&L
6. **Calibration** — Vasicek / CIR / Heston

## Inventaire.xlsx — feuilles supportées

- **Swap** : nominal, maturité, taux fixe, fréquence → `InterestRateSwap` ou `FloatingRateBond`
- **Options** : Call/Put/CallSpread/PutSpread/Butterfly + barrière OUT/IN éventuelle
- **Autocall** : observations + niveaux de rappel + coupons → `Autocall`
- **Notes structurées** : codes SSPA 1100/1130/1220/1320 → réplication statique
