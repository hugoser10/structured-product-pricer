import os

from pricer.data.loader import (
    build_market_data,
    load_rate_curves,
    load_options,
    available_tickers,
    available_dates,
    get_latest_curve_date,
    PATH_OPTIONS,
    PATH_RATE_CURVES,
)

# Chemin absolu vers l'inventaire (utilisé par app.py et InventoryLoader)
DEFAULT_INVENTORY_PATH = os.path.join(os.path.dirname(__file__), "Inventaire.xlsx")

__all__ = [
    "build_market_data",
    "load_rate_curves",
    "load_options",
    "available_tickers",
    "available_dates",
    "get_latest_curve_date",
    "PATH_OPTIONS",
    "PATH_RATE_CURVES",
    "DEFAULT_INVENTORY_PATH",
]
