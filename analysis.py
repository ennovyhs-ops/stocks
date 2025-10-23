import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from math import log, sqrt, exp

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

# --- Indicators & realized vol ---
def returns_from_close(close):
    return np.log(close / close.shift(1)).dropna()

def ewma_vol(close, span_days=60):
    r = returns_from_close(close)
    if r.empty:
        return 0.25
    lam = 2 / (span_days + 1)
    ewma_var = (r ** 2).ewm(alpha=lam, adjust=False).mean()
    return float(np.sqrt(ewma_var.iloc[-1] * 252))

def fit_garch11_annualized(close, horizon_days=1):
    if not ARCH_AVAILABLE:
        return None
    r = returns_from_close(close) * 100.0
    if r.empty:
        return None
    try:
        am = arch_model(r, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", last_obs=r.index[-1])
        f = res.forecast(horizon=horizon_days, reindex=False)
        var_forecast = f.variance.values[-1, -1] / (100.0 ** 2)
        return float(sqrt(var_forecast * 252))
    except Exception:
        logger.exception("GARCH fit failed")
        return None

# --- Black-Scholes & IV inversion ---
def bs_d1(S, K, r, q, sigma, T):
    return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * sqrt(max(T, 1e-12)) + 1e-12)

def call_price_bs(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma*sqrt(T)
    return S*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def put_price_bs(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma*sqrt(T)
    return K*exp(-r*T)*norm.cdf(-d2) - S*exp(-q*T)*norm.cdf(-d1)

def vega_bs(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = bs_d1(S, K, r, q, sigma, T)
    return S * exp(-q*T) * norm.pdf(d1) * sqrt(T)

def implied_vol_from_price(mkt_price, S, K, r, q, T, is_call=True, tol=1e-8, maxiter=100):
    if mkt_price <= 0 or S <= 0 or K <= 0:
        return 0.0
    sigma = 0.20
    for i in range(maxiter):
        price = call_price_bs(S, K, r, q, sigma, T) if is_call else put_price_bs(S, K, r, q, sigma, T)
        diff = price - mkt_price
        if abs(diff) < tol:
            logger.debug("IV Newton converged iter=%d sigma=%.6f", i, sigma)
            return max(sigma, 1e-6)
        v = vega_bs(S, K, r, q, sigma, T)
        if v < 1e-8:
            logger.debug("Vega too small; fallback to Brent")
            break
        sigma = sigma - diff / v
        if sigma <= 1e-8 or sigma > 5.0:
            logger.debug("Newton produced invalid sigma; fallback to Brent")
            break
    # Brent fallback
    def f(s):
        return (call_price_bs(S, K, r, q, s, T) if is_call else put_price_bs(S, K, r, q, s, T)) - mkt_price
    try:
        low, high = 1e-8, 5.0
        iv = brentq(f, low, high, maxiter=200)
        logger.debug("IV Brent converged sigma=%.6f", iv)
        return iv
    except Exception:
        logger.exception("Implied vol solver failed; returning last sigma guess")
        return max(sigma, 1e-6)
