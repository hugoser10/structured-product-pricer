import os
import pandas as pd

# Define directory
DATA_DIR = os.path.join(os.path.dirname(__file__))
PATH_OPTIONS = os.path.join(DATA_DIR, 'options.csv')
PATH_RATE_CURVES = os.path.join(DATA_DIR, 'rate_curves.csv')

# Rate curves
def load_rate_curves(path: str = PATH_RATE_CURVES, country: str = "United States") -> pd.DataFrame:
    """
    Loads rate curves csv and returns a DataFrame for the chosen country
    DataFrame: country, maturity, date, rate (%)
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    if country:
        df: pd.DataFrame = df[df["country"] == country]
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index).date
        df.sort_index(inplace=True)
    return df

def get_latest_curve_date(path: str = PATH_RATE_CURVES, country: str = "United States") -> str:
    """Returns the most recent date for which the chosen country has data"""
    df = load_rate_curves(path, country)
    return str(df.index.max())

# Options
def load_options(path: str = PATH_OPTIONS, ticker: str = "AAPL", date: str | None = None) -> pd.DataFrame:
    """
    Loads option data for a given ticker and a given date (or the most recent date)
    DataFrame: ticker, date, side, strike, dte, mid, underlyingPrice, iv, delta, etc
    """
    df = pd.read_csv(path, sep=";")
    df = df[df["ticker"] == ticker]
    if date:
        df = df[df["date"] == date]
    else:
        df = df[df["date"] == df["date"].max()]
    return df.reset_index(drop=True)

def available_tickers(path: str = PATH_OPTIONS) -> list:
    """Returns a list of the available tickers (options)"""
    df = pd.read_csv(path, sep=";", usecols=["ticker"])
    return sorted(df["ticker"].unique().tolist())

def available_dates(path: str = PATH_OPTIONS, ticker: str = "AAPL") -> list:
    """Returns a list of the available dates (options)"""
    df = pd.read_csv(path, sep=";", usecols=["ticker", "date"])
    return sorted(df[df["ticker"] == ticker]["date"].unique().tolist(), reverse=True)

# Market Data
def build_market_data(ticker: str = "AAPL", country: str = "United States",
                      options_date: str | None = None, rates_date: str | None = None,
                      options_path: str = PATH_OPTIONS, rates_path: str = PATH_RATE_CURVES) -> dict:
    """
    Builds a dictionary with a rate curve, an implied vol surface for given ticker and country
    Dict: {
        "rate_curve": RateCurve calibrée,
        "vol_surface": ImpliedVolSurface calibrée,
        "spot": prix spot du sous-jacent,
        "r": taux court terme (proxy depuis la courbe),
        "ticker": ticker,
        "options_date": date du panel d'options,
        }
    """
    from pricer.models.rates.rate_model import RateCurve
    from pricer.models.vol_model import ImpliedVolSurface

    # Build rate curve
    rc = RateCurve(country=country, date=rates_date, data_path=rates_path)

    # Get zero rate from rate curve
    r_short = rc.zero_rate(0.25)

    # Build vol surface from ticker's options
    vol_surface = ImpliedVolSurface(
        ticker=ticker, date=options_date,
        data_path=options_path, r=r_short
    )

    return {
        "rate_curve": rc,
        "vol_surface": vol_surface,
        "spot": vol_surface.spot,
        "r": r_short,
        "ticker": ticker,
        "options_date": options_date or available_dates(options_path, ticker)[0],
    }