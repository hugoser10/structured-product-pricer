import numpy as np
from scipy.stats import norm
from typing import Optional, Dict


def bs_call(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma, q=0.0):
    return bs_call(S, K, T, r, sigma, q) - S * np.exp(-q * T) + K * np.exp(-r * T)


def bs_vega(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0:
        return 1e-10
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1)


def bsm_d1_d2(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bsm_price(S, K, T, r, sigma, option_type, q=0.0):
    if T <= 0:
        return float(max(S - K, 0) if option_type == "call" else max(K - S, 0))
    d1, d2 = bsm_d1_d2(S, K, T, r, sigma, q)
    eq, dk = np.exp(-q * T), np.exp(-r * T)
    if option_type == "call":
        return float(S * eq * norm.cdf(d1) - K * dk * norm.cdf(d2))
    return float(K * dk * norm.cdf(-d2) - S * eq * norm.cdf(-d1))


def bsm_greeks(S, K, T, r, sigma, option_type, q=0.0) -> Dict[str, float]:
    if T <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = bsm_d1_d2(S, K, T, r, sigma, q)
    sq, eq, dk = np.sqrt(T), np.exp(-q * T), np.exp(-r * T)
    gamma = eq * norm.pdf(d1) / (S * sigma * sq)
    vega = S * eq * norm.pdf(d1) * sq / 100
    if option_type == "call":
        delta = eq * norm.cdf(d1)
        theta = (-S * eq * norm.pdf(d1) * sigma / (2 * sq)
                 - r * K * dk * norm.cdf(d2) + q * S * eq * norm.cdf(d1)) / 365
        rho = K * T * dk * norm.cdf(d2) / 100
    else:
        delta = -eq * norm.cdf(-d1)
        theta = (-S * eq * norm.pdf(d1) * sigma / (2 * sq)
                 + r * K * dk * norm.cdf(-d2) - q * S * eq * norm.cdf(-d1)) / 365
        rho = -K * T * dk * norm.cdf(-d2) / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol_newton(market_price, S, K, T, r, side="call", q=0.0,
                       tol=1e-6, max_iter=100) -> Optional[float]:
    """Newton-Raphson : sigma_{n+1} = sigma_n - (BS(sigma_n) - market) / vega."""
    if T <= 0 or market_price <= 0:
        return None
    fwd = S * np.exp((r - q) * T)
    intrinsic = (max(fwd - K, 0) if side == "call" else max(K - fwd, 0)) * np.exp(-r * T)
    if market_price < intrinsic - 1e-6:
        return None
    sigma = 0.3
    for _ in range(max_iter):
        price = bs_call(S, K, T, r, sigma, q) if side == "call" else bs_put(S, K, T, r, sigma, q)
        vega = bs_vega(S, K, T, r, sigma, q)
        if abs(vega) < 1e-10:
            sigma = np.random.uniform(0.05, 0.8)
            continue
        sigma_new = np.clip(sigma - (price - market_price) / vega, 1e-4, 5.0)
        if abs(sigma_new - sigma) < tol:
            return sigma_new
        sigma = sigma_new
    return None