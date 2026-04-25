# fichier avec les outils necessaires aux calculs

### BSM
def _bsm_d1_d2(S, K, T, r, sigma, q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)

def _bsm_price(S, K, T, r, sigma, option_type, q=0.0):
    if T <= 0:
        return float(max(S - K, 0) if option_type == "call" else max(K - S, 0))
    d1, d2 = _bsm_d1_d2(S, K, T, r, sigma, q)
    eq, dk = np.exp(-q * T), np.exp(-r * T)
    if option_type == "call":
        return float(S * eq * norm.cdf(d1) - K * dk * norm.cdf(d2))
    return float(K * dk * norm.cdf(-d2) - S * eq * norm.cdf(-d1))

def _bsm_greeks(S, K, T, r, sigma, option_type, q=0.0) -> Dict[str, float]:
    if T <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _bsm_d1_d2(S, K, T, r, sigma, q)
    sq, eq, dk = np.sqrt(T), np.exp(-q * T), np.exp(-r * T)
    gamma = eq * norm.pdf(d1) / (S * sigma * sq)
    vega  = S * eq * norm.pdf(d1) * sq / 100
    if option_type == "call":
        delta = eq * norm.cdf(d1)
        theta = (-S * eq * norm.pdf(d1) * sigma / (2 * sq)
                 - r * K * dk * norm.cdf(d2)
                 + q * S * eq * norm.cdf(d1)) / 365
        rho = K * T * dk * norm.cdf(d2) / 100
    else:
        delta = -eq * norm.cdf(-d1)
        theta = (-S * eq * norm.pdf(d1) * sigma / (2 * sq)
                 + r * K * dk * norm.cdf(-d2)
                 - q * S * eq * norm.cdf(-d1)) / 365
        rho = -K * T * dk * norm.cdf(-d2) / 100
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
