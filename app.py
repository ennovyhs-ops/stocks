#!/usr/bin/env python3
# app.py - Daily Option Strategy Analyzer without Matplotlib
# Defaults: 9988.HK, 0005.HK, 1810.HK against ^HSI
# Charts are generated as lightweight inline SVG sparklines so Matplotlib is not required.

import io
import json
import os
import threading
import time
import logging
from datetime import datetime, date
from math import log, sqrt, exp

from urllib.parse import quote

from flask import Flask, request, jsonify, render_template, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, least_squares

# Optional extras
try:
    from diskcache import Cache
    CACHE_AVAILABLE = True
except Exception:
    Cache = None
    CACHE_AVAILABLE = False

try:
    from sklearn.isotonic import IsotonicRegression
    ISOTONIC_AVAILABLE = True
except Exception:
    ISOTONIC_AVAILABLE = False

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

try:
    import py_ssvi
    SSVI_AVAILABLE = True
except Exception:
    py_ssvi = None
    SSVI_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dos_analyzer")

# Flask app
app = Flask(__name__, template_folder="templates")

# Config & constants
DEFAULT_CONFIG_PATH = "config.json"
ALLOWED_TRADE_POLICIES = ["prefer_spread", "allow_naked"]
DEFAULT_STOCKS = ["9988.HK", "0005.HK", "1810.HK"]
STRIKE_STEP = 0.01
R = 0.02
Q = 0.0
TIMEFRAMES = {"next_day": 1, "2_weeks": 10, "3_months": 65, "6_months": 130}
DELTA_MAP = {1: (0.50, 0.20), 10: (0.50, 0.25), 65: (0.48, 0.22), 130: (0.42, 0.20)}
PREFER_OTM = {1: False, 10: True, 65: True, 130: True}
TRADE_POLICY = "prefer_spread"

OPTION_MIN_VOLUME = 1
OPTION_MIN_OI = 1
MAX_BID_ASK_SPREAD_PCT = 0.30
MIN_STRIKES_PER_SLICE = 6

CACHE_DIR = "cache_dir"
CACHE_TTL_SECONDS = 30 * 60
RECALIBRATE_INTERVAL_SECONDS = 24 * 3600
CALIBRATION_ALERT_THRESHOLD = 0.10

# Initialize cache
cache = Cache(CACHE_DIR) if CACHE_AVAILABLE else None
if not CACHE_AVAILABLE:
    logger.info("diskcache not available; caching disabled")

# Helpers
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def today_str():
    return date.today().isoformat()

def round_strike(x):
    return float(np.round(np.round(x / STRIKE_STEP) * STRIKE_STEP, 2))

# Indicators and realized vol
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

# Black-Scholes and IV inversion
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
            logger.debug("Vega too small; falling back to Brent")
            break
        sigma = sigma - diff / v
        if sigma <= 1e-8 or sigma > 5.0:
            logger.debug("Newton produced invalid sigma; falling back to Brent")
            break
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

# SVI helpers
def svi_total_variance(params, x):
    a, b, rho, m, sigma = params
    return a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))

def svi_residuals(params, x, w_market):
    return svi_total_variance(params, x) - w_market

def fit_svi_slice(strikes, ivs, F, T):
    if T <= 0 or len(strikes) < MIN_STRIKES_PER_SLICE:
        return None
    x = np.log(np.array(strikes) / float(F))
    w_market = (np.array(ivs) ** 2) * T
    a0 = max(1e-8, float(np.mean(w_market) * 0.8))
    p0 = [a0, 0.1, 0.0, float(np.median(x)), 0.2]
    lower = [0.0, 1e-8, -0.999, float(np.min(x))*2, 1e-8]
    upper = [float(np.max(w_market))*5 + 1.0, 5.0, 0.999, float(np.max(x))*2, 5.0]
    try:
        res = least_squares(svi_residuals, p0, bounds=(lower, upper), args=(x, w_market), xtol=1e-8, ftol=1e-8, max_nfev=5000)
        if res.success:
            return res.x.tolist()
        logger.debug("SVI fit not successful: %s", res.message)
        return None
    except Exception:
        logger.exception("SVI fit exception")
        return None

def svi_iv_from_params(params, K, F, T):
    a, b, rho, m, sigma = params
    x = log(K / F)
    w = a + b * (rho * (x - m) + sqrt((x - m)**2 + sigma**2))
    return sqrt(max(w / max(T, 1e-12), 0.0))

# Cross-expiry calibration
def enforce_atm_monotonic_isotonic(svi_surface):
    items = []
    for exp_iso, params in svi_surface.items():
        if not params:
            continue
        T = expiry_to_T(exp_iso)
        w_atm = svi_total_variance(params, 0.0)
        items.append((exp_iso, T, float(w_atm)))
    if not items:
        return svi_surface
    items.sort(key=lambda x: x[1])
    Ts = np.array([x[1] for x in items])
    ws = np.array([x[2] for x in items])
    if ISOTONIC_AVAILABLE:
        try:
            ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
            w_iso = ir.fit_transform(Ts, ws)
            for (exp_iso, _, _), w_target in zip(items, w_iso):
                params = svi_surface.get(exp_iso)
                if not params:
                    continue
                a,b,rho,m,sig = params
                atm_contrib = b * (rho * (-m) + sqrt(m**2 + sig**2))
                params[0] = max(0.0, float(w_target) - atm_contrib)
                svi_surface[exp_iso] = params
            return svi_surface
        except Exception:
            logger.exception("Isotonic enforcement failed; falling back")
    prev = -1.0
    for exp_iso, T, w in items:
        params = svi_surface.get(exp_iso)
        if not params:
            continue
        if w < prev:
            bump = prev - w + 1e-8
            params[0] = params[0] + bump
            svi_surface[exp_iso] = params
            w = w + bump
        prev = w
    return svi_surface

def calibrate_surface_ssvi(parsed_option_rows, spot):
    if SSVI_AVAILABLE and py_ssvi is not None:
        try:
            surface_input = {}
            for exp_iso, rows in parsed_option_rows.items():
                strikes = []
                ivs = []
                T = expiry_to_T(exp_iso)
                for r in rows:
                    k = r['strike']
                    iv = r.get('iv_call') if r.get('iv_call') is not None else r.get('iv_put')
                    if iv is not None and iv > 0:
                        strikes.append(k); ivs.append(iv)
                if len(strikes) >= MIN_STRIKES_PER_SLICE:
                    surface_input[exp_iso] = {"T": T, "strikes": strikes, "ivs": ivs}
            if not surface_input:
                raise RuntimeError("No valid slices for SSVI")
            if hasattr(py_ssvi, "calibrate_surface"):
                ssvi_surface = py_ssvi.calibrate_surface(surface_input, spot=spot)
                logger.info("SSVI calibrated via py_ssvi.calibrate_surface")
                return ssvi_surface
            if hasattr(py_ssvi, "ssvi") and hasattr(py_ssvi.ssvi, "calibrate_surface"):
                ssvi_surface = py_ssvi.ssvi.calibrate_surface(surface_input, spot=spot)
                logger.info("SSVI calibrated via py_ssvi.ssvi.calibrate_surface")
                return ssvi_surface
            logger.info("py_ssvi present but no calibrator API; falling back")
        except Exception:
            logger.exception("SSVI calibration failed; falling back")
    svi_surface = {}
    for exp_iso, rows in parsed_option_rows.items():
        strikes = []
        ivs = []
        for r in rows:
            k = r['strike']
            iv = r.get('iv_call') if r.get('iv_call') is not None else r.get('iv_put')
            if iv is not None and iv > 0:
                strikes.append(k); ivs.append(iv)
        if len(strikes) < MIN_STRIKES_PER_SLICE:
            svi_surface[exp_iso] = None
            continue
        params = fit_svi_slice(strikes, ivs, spot, expiry_to_T(exp_iso))
        svi_surface[exp_iso] = params
    svi_surface = enforce_atm_monotonic_isotonic(svi_surface)
    return svi_surface

# Option chain parsing with quality filters
def parse_option_chain_with_quality(tk, spot):
    out = {}
    try:
        expiries = getattr(tk, "options", []) or []
    except Exception:
        return {}
    for exp in expiries[:8]:
        try:
            chain = tk.option_chain(exp)
            calls = chain.calls
            puts = chain.puts
            strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
            rows = []
            for K in strikes:
                call_mid = None; put_mid = None
                crow = calls[calls['strike'] == K]
                prow = puts[puts['strike'] == K]
                if not crow.empty:
                    try:
                        cbid = float(crow['bid'].iloc[0]); cask = float(crow['ask'].iloc[0])
                        if cbid > 0 and cask > 0:
                            mid = (cbid + cask) / 2.0
                            spread_pct = (cask - cbid) / max(1e-8, mid)
                            vol = int(crow.get('volume', [0]).iloc[0]) if 'volume' in crow else 0
                            oi = int(crow.get('openInterest', [0]).iloc[0]) if 'openInterest' in crow else 0
                            if spread_pct <= MAX_BID_ASK_SPREAD_PCT and (vol >= OPTION_MIN_VOLUME or oi >= OPTION_MIN_OI):
                                call_mid = mid
                        if call_mid is None:
                            last = float(crow.get('lastPrice', np.nan).iloc[0]) if 'lastPrice' in crow else None
                            if last and last > 0:
                                call_mid = last
                    except Exception:
                        pass
                if not prow.empty:
                    try:
                        pbid = float(prow['bid'].iloc[0]); pask = float(prow['ask'].iloc[0])
                        if pbid > 0 and pask > 0:
                            mid = (pbid + pask) / 2.0
                            spread_pct = (pask - pbid) / max(1e-8, mid)
                            vol = int(prow.get('volume', [0]).iloc[0]) if 'volume' in prow else 0
                            oi = int(prow.get('openInterest', [0]).iloc[0]) if 'openInterest' in prow else 0
                            if spread_pct <= MAX_BID_ASK_SPREAD_PCT and (vol >= OPTION_MIN_VOLUME or oi >= OPTION_MIN_OI):
                                put_mid = mid
                        if put_mid is None:
                            last = float(prow.get('lastPrice', np.nan).iloc[0]) if 'lastPrice' in prow else None
                            if last and last > 0:
                                put_mid = last
                    except Exception:
                        pass
                rows.append({"strike": float(K), "call_mid": call_mid, "put_mid": put_mid})
            out[datetime.fromisoformat(exp).isoformat()] = rows
        except Exception:
            logger.exception("Failed parsing option chain for %s", exp)
            continue
    return out

# Caching helpers
def cache_key_option_chain(ticker):
    return f"optchain:{ticker}:{today_str()}"

def cache_key_svi_surface(ticker):
    return f"svi_surf:{ticker}:{today_str()}"

def get_option_chain_cached(ticker, spot, force_refresh=False):
    if not CACHE_AVAILABLE:
        return parse_option_chain_with_quality(yf.Ticker(ticker), spot)
    key = cache_key_option_chain(ticker)
    if force_refresh:
        cache.delete(key)
    data = cache.get(key)
    if data is not None:
        logger.debug("Option chain cache hit %s", ticker)
        return data
    tk = yf.Ticker(ticker)
    parsed = parse_option_chain_with_quality(tk, spot)
    cache.set(key, parsed, expire=CACHE_TTL_SECONDS)
    return parsed

def get_svi_surface_cached(ticker, spot, force_refresh=False):
    if not CACHE_AVAILABLE:
        parsed = parse_option_chain_with_quality(yf.Ticker(ticker), spot)
        return calibrate_surface_ssvi(parsed, spot)
    key = cache_key_svi_surface(ticker)
    if force_refresh:
        cache.delete(key)
    data = cache.get(key)
    if data is not None:
        logger.debug("SVI surface cache hit %s", ticker)
        return data
    parsed = get_option_chain_cached(ticker, spot)
    svi = calibrate_surface_ssvi(parsed, spot) if parsed else {}
    cache.set(key, svi, expire=CACHE_TTL_SECONDS)
    return svi

# Lightweight SVG chart generator to replace Matplotlib
def generate_svg_chart(df, ticker, width=420, height=260, padding=6):
    # df expected to have Date and Close columns sorted ascending
    close = np.array(df['Close'].astype(float))
    if close.size == 0:
        return None
    w = width - 2 * padding
    h = height - 2 * padding
    xs = np.linspace(padding, padding + w, num=close.size)
    mn, mx = float(np.min(close)), float(np.max(close))
    if mx - mn < 1e-8:
        ys = np.full_like(xs, padding + h / 2.0)
    else:
        ys = padding + (mx - close) / (mx - mn) * h
    # Build path string
    pts = ["{:.2f},{:.2f}".format(x, y) for x, y in zip(xs, ys)]
    path = "M " + " L ".join(pts)
    # Small sparkline stroke color
    stroke = "#1f77b4"
    # Title and subtitle
    title = f"{ticker} Close"
    # Build simple SVG
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect width="100%" height="100%" fill="#ffffff"/>
      <g>
        <path d="{path}" fill="none" stroke="{stroke}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
      </g>
      <text x="{padding+2}" y="{padding+12}" font-family="Arial" font-size="12" fill="#222">{title}</text>
      <text x="{width-padding-120}" y="{padding+12}" font-family="Arial" font-size="11" fill="#666">latest {float(close[-1]):.2f}</text>
    </svg>'''
    return "data:image/svg+xml;utf8," + quote(svg)

def plot_base64(df, ticker):
    # Returns an image data URI (SVG) or None
    try:
        svg_uri = generate_svg_chart(df, ticker)
        return svg_uri
    except Exception:
        logger.exception("SVG chart generation failed")
        return None

# Blended sigma
def blended_sigma(close_series, iv_surface=None, S=None, K=None, expiry=None, weights=None):
    if weights is None:
        weights = {"implied": 0.6, "ewma": 0.25, "garch": 0.15}
    ewma = ewma_vol(close_series, span_days=60)
    garch = fit_garch11_annualized(close_series, horizon_days=1) if ARCH_AVAILABLE else None
    implied = None
    if iv_surface and expiry and S and K:
        params = iv_surface.get(expiry)
        if params:
            T = expiry_to_T(expiry)
            try:
                implied = svi_iv_from_params(params, K, S, T)
            except Exception:
                implied = None
    components = []
    vals = []
    if implied is not None:
        components.append("implied"); vals.append(float(implied))
    if ewma is not None:
        components.append("ewma"); vals.append(float(ewma))
    if garch is not None:
        components.append("garch"); vals.append(float(garch))
    total_w = sum(weights.get(k, 0) for k in components)
    if total_w <= 0 or not vals:
        return max(0.05, float(ewma or garch or 0.25))
    blended = sum(vals[i] * (weights.get(components[i], 0) / total_w) for i in range(len(vals)))
    return float(max(1e-4, blended))

# Delta helpers and strike inversion
def call_delta(S, K, r, q, sigma, T):
    if T <=0 or sigma <=0: return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, r, q, sigma, T)
    return np.exp(-q*T) * norm.cdf(d1)

def put_delta(S, K, r, q, sigma, T):
    if T <=0 or sigma <=0: return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, r, q, sigma, T)
    return np.exp(-q*T) * (norm.cdf(d1) - 1.0)

def strike_for_delta(S, target_delta, is_call, r, q, sigma, T, prefer_otm=False):
    abs_target = abs(target_delta)
    def f(K):
        if sigma <= 1e-6 or T <= 1e-12:
            return (1.0 if (is_call and S > K) else 0.0) - abs_target
        return (call_delta(S, K, r, q, sigma, T) - abs_target) if is_call else (abs(put_delta(S, K, r, q, sigma, T)) - abs_target)
    if prefer_otm:
        if is_call:
            low = max(S * 1.0001, S + STRIKE_STEP)
            high = S * 3.0
        else:
            low = 0.01
            high = max(S * 0.9999, S - STRIKE_STEP)
    else:
        low = max(0.01, S * 0.2)
        high = S * 3.0
    try:
        K = brentq(f, low, high, maxiter=300)
        return round_strike(K)
    except Exception:
        logger.debug("strike_for_delta brent failed S=%.4f target=%.3f T=%.4f", S, target_delta, T)
        return round_strike(S)

# Scoring and recommendation
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def compute_rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)

def compute_macd(close):
    fast = ema(close, 12)
    slow = ema(close, 26)
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def vwma(price, vol, window):
    pv = (price * vol).rolling(window=window, min_periods=1).sum()
    v = vol.rolling(window=window, min_periods=1).sum().replace(0, np.nan)
    return pv / v

def score_and_components(df, index_df, days):
    window = df.tail(days)
    if window.empty:
        return 0.0, {}
    latest = window.iloc[-1]
    comps = {}
    comps['rsi'] = 1.0 if latest['RSI'] <= 30 else (-1.0 if latest['RSI'] >= 70 else (50 - latest['RSI']) / 50.0)
    mdiff = latest['MACD'] - latest['MACD_sign']
    comps['macd'] = 1.0 if mdiff > 0 else -1.0
    comps['vwma'] = 1.0 if latest['Close'] > latest['VWMA50'] else -1.0
    comps['bb'] = 1.0 if latest['Close'] > latest['BB_upper'] else (-1.0 if latest['Close'] < latest['BB_lower'] else np.tanh((latest['Close'] - latest['BB_mid']) / (latest['BB_std'] + 1e-9) / 2.0))
    comps['vol'] = 1.0 if latest['Volume'] > latest['Vol20'] else 0.6
    comps['vwap'] = np.tanh((latest['Close'] - latest.get('VWAP20', latest['Close'])) / (latest.get('VWAP20', latest['Close']) + 1e-9) * 10.0)
    rsr = (df['Close'] / index_df['Close']).dropna()
    slope = 0.0
    if len(rsr) >= 3:
        y = rsr.tail(min(len(rsr), days)).values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
    comps['rsr'] = np.tanh(slope * 100.0)
    weights = {'rsi': 0.14, 'macd': 0.16, 'vwma': 0.26, 'bb': 0.18, 'vol': 0.10, 'vwap': 0.06, 'rsr': 0.10}
    total = sum(weights[k] * comps.get(k, 0.0) for k in weights)
    wsum = sum(abs(v) for v in weights.values())
    composite = float(np.clip(total / wsum, -1.0, 1.0))
    return composite, comps

def recommend_with_policy(S, close_series, composite, days, iv_surface=None, policy="prefer_spread"):
    T = max(1, days) / 365.0
    expiry = next(iter(iv_surface.keys()), None) if iv_surface else None
    sigma_blend = blended_sigma(close_series, iv_surface, S=S, K=S, expiry=expiry)
    nearest = min(DELTA_MAP.keys(), key=lambda k: abs(k - days))
    buy_delta, sell_delta = DELTA_MAP[nearest]
    prefer_otm = PREFER_OTM.get(nearest, True)
    if composite >= 0.6:
        side = "LONG_CALL"; is_call = True; target_delta = buy_delta
    elif composite <= -0.6:
        side = "LONG_PUT"; is_call = False; target_delta = -buy_delta
    elif composite > 0.2:
        side = "BUY_CALL_SPREAD"; is_call = True; target_delta = buy_delta
    elif composite < -0.2:
        side = "BUY_PUT_SPREAD"; is_call = False; target_delta = -buy_delta
    else:
        side = "NEUTRAL_SELL_CREDIT"; is_call = composite > 0; target_delta = sell_delta if composite > 0 else -sell_delta
    abs_target = abs(target_delta)
    strike_target = strike_for_delta(S, abs_target, is_call, R, Q, sigma_blend, T, prefer_otm=prefer_otm)
    atm = round_strike(S)
    conservative = round_strike((strike_target + atm) / 2.0)
    final = {"side": side, "is_call": bool(is_call), "target_delta": float(np.sign(target_delta) * abs_target),
             "strike_target": strike_target, "strike_atm": atm, "strike_conservative": conservative,
             "sigma_used": round(sigma_blend, 4), "T_years": round(T, 4)}
    if is_call:
        long_price = call_price_bs(S, conservative, R, Q, sigma_blend, T)
        short_price = call_price_bs(S, strike_target, R, Q, sigma_blend, T)
    else:
        long_price = put_price_bs(S, conservative, R, Q, sigma_blend, T)
        short_price = put_price_bs(S, strike_target, R, Q, sigma_blend, T)
    if policy == "prefer_spread" and final["side"] in ("LONG_CALL", "LONG_PUT"):
        spread_side = "BUY_CALL_SPREAD" if is_call else "BUY_PUT_SPREAD"
        long_leg = conservative
        short_leg = strike_target if ((is_call and strike_target > long_leg) or (not is_call and strike_target < long_leg)) else atm
        if is_call:
            long_p = call_price_bs(S, long_leg, R, Q, sigma_blend, T); short_p = call_price_bs(S, short_leg, R, Q, sigma_blend, T)
        else:
            long_p = put_price_bs(S, long_leg, R, Q, sigma_blend, T); short_p = put_price_bs(S, short_leg, R, Q, sigma_blend, T)
        spread_cost = round(long_p - short_p, 6)
        final.update({"side": spread_side, "long_strike": long_leg, "short_strike": short_leg, "long_premium": round(long_p, 6), "short_premium": round(short_p, 6), "spread_cost": spread_cost})
    else:
        final.update({"leg_strike": conservative, "leg_premium": round(long_price, 6), "note": "Trade now." if abs(composite) >= 0.3 else "Consider wait for better setup or premium."})
    return final

# Backtest / recalibration
def realised_forward_vol(close_series, days_forward=21):
    ret = returns_from_close(close_series)
    if len(ret) < days_forward:
        return None
    fwd = ret.tail(days_forward)
    return float(np.sqrt(np.sum(fwd**2) * (252.0 / days_forward)))

def recalibration_job():
    logger.info("Starting recalibration job")
    while True:
        try:
            cfg = load_config()
            stocks = cfg.get("stocks", DEFAULT_STOCKS)[:3]
            metrics = {}
            for t in stocks:
                try:
                    tk = yf.Ticker(t)
                    hist = tk.history(period="1y", interval="1d", auto_adjust=False)
                    if hist.empty:
                        metrics[t] = {"error": "no_history"}
                        continue
                    spot = float(hist['Close'].iloc[-1])
                    parsed = get_option_chain_cached(t, spot, force_refresh=True) if CACHE_AVAILABLE else parse_option_chain_with_quality(tk, spot)
                    svi = calibrate_surface_ssvi(parsed, spot) if parsed else {}
                    residuals = []
                    for exp_iso, rows in (parsed or {}).items():
                        params = svi.get(exp_iso)
                        if not params:
                            continue
                        T = expiry_to_T(exp_iso)
                        for r in rows:
                            k = r['strike']
                            iv_market = None
                            if r.get('call_mid'):
                                try:
                                    iv_market = implied_vol_from_price(r['call_mid'], spot, k, R, Q, T, is_call=True)
                                except Exception:
                                    iv_market = None
                            if r.get('put_mid') and iv_market is None:
                                try:
                                    iv_market = implied_vol_from_price(r['put_mid'], spot, k, R, Q, T, is_call=False)
                                except Exception:
                                    iv_market = None
                            if iv_market is None:
                                continue
                            try:
                                iv_model = svi_iv_from_params(params, k, spot, T)
                                residuals.append(iv_model - iv_market)
                            except Exception:
                                continue
                    if residuals:
                        rmse = float(np.sqrt(np.mean(np.array(residuals)**2)))
                        metrics[t] = {"rmse": rmse}
                        if rmse > CALIBRATION_ALERT_THRESHOLD:
                            logger.warning("Calibration alert %s rmse=%.4f", t, rmse)
                            with open("calibration_alert.log", "a") as af:
                                af.write(f"{now_iso()} ALERT {t} rmse={rmse}\n")
                    else:
                        metrics[t] = {"rmse": None}
                except Exception:
                    logger.exception("Recal failed for %s", t)
                    metrics[t] = {"error": "failed"}
            with open("recalibration_metrics.json", "w") as f:
                json.dump({"timestamp": now_iso(), "metrics": metrics}, f, indent=2)
            logger.info("Recalibration complete; sleeping %ds", RECALIBRATE_INTERVAL_SECONDS)
        except Exception:
            logger.exception("Recal top-level error")
        time.sleep(RECALIBRATE_INTERVAL_SECONDS)

def start_background_jobs():
    t = threading.Thread(target=recalibration_job, daemon=True, name="recalib-thread")
    t.start()
    logger.info("Background recalibration started")

# Config load/save
def load_config(path=DEFAULT_CONFIG_PATH):
    if not os.path.exists(path):
        sample = {"stocks": DEFAULT_STOCKS, "index": "^HSI", "strike_step": STRIKE_STEP, "trade_policy": TRADE_POLICY}
        with open(path, "w") as f:
            json.dump(sample, f, indent=2)
        return sample
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("stocks", DEFAULT_STOCKS)
    cfg.setdefault("index", "^HSI")
    cfg.setdefault("strike_step", STRIKE_STEP)
    cfg.setdefault("trade_policy", TRADE_POLICY)
    return cfg

def save_config(cfg, path=DEFAULT_CONFIG_PATH):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

# Clear cache endpoint used by UI Force Refresh
@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    if not CACHE_AVAILABLE:
        return jsonify({"cleared": False, "reason": "cache not available"}), 200
    try:
        cache.clear()
        logger.info("Cache cleared via /clear-cache")
        return jsonify({"cleared": True}), 200
    except Exception:
        logger.exception("Failed to clear cache")
        return jsonify({"cleared": False, "reason": "error"}), 500

# API endpoints
@app.route("/config", methods=["GET"])
def get_config_api():
    return jsonify(load_config())

@app.route("/config", methods=["POST"])
def post_config_api():
    body = request.get_json(force=True)
    stocks = body.get("stocks")
    index = body.get("index")
    strike_step = body.get("strike_step")
    trade_policy = body.get("trade_policy")
    if stocks is not None:
        if not isinstance(stocks, list):
            return jsonify({"error": "stocks must be a list"}), 400
        if len(stocks) > 5:
            return jsonify({"error": "maximum 5 stocks allowed"}), 400
    if trade_policy is not None and trade_policy not in ALLOWED_TRADE_POLICIES:
        return jsonify({"error": f"trade_policy must be one of: {', '.join(ALLOWED_TRADE_POLICIES)}"}), 400
    cfg = load_config()
    if stocks is not None: cfg["stocks"] = stocks[:5]
    if index is not None: cfg["index"] = index
    if strike_step is not None: cfg["strike_step"] = float(strike_step)
    if trade_policy is not None: cfg["trade_policy"] = trade_policy
    save_config(cfg)
    return jsonify(cfg)

@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True)
    stocks = body.get("stocks") or load_config().get("stocks", DEFAULT_STOCKS)
    stocks = stocks[:5]
    index = body.get("index") or load_config().get("index", "^HSI")
    if not stocks or not index:
        return jsonify({"error": "Provide stocks and index"}), 400
    period = "2y"
    results = {}
    idx_df = yf.Ticker(index).history(period=period, interval="1d", auto_adjust=False).reset_index()
    idx_df = idx_df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    for t in stocks:
        try:
            raw = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=False).reset_index()
            raw = raw.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
            if raw.empty:
                results[t] = {"error": "no_history"}
                continue
            df = raw.copy()
            df['VWMA20'] = vwma(df['Close'], df['Volume'], 20)
            df['VWMA50'] = vwma(df['Close'], df['Volume'], 50)
            df['BB_mid'] = df['Close'].rolling(20).mean()
            df['BB_std'] = df['Close'].rolling(20).std()
            df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
            df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
            df['RSI'] = compute_rsi(df['Close'], 14)
            macd, signal, hist = compute_macd(df['Close'])
            df['MACD'] = macd; df['MACD_sign'] = signal; df['MACD_hist'] = hist
            df['Vol20'] = df['Volume'].rolling(20).mean()
            tp = (df['High'] + df['Low'] + df['Close']) / 3.0
            pv = (tp * df['Volume']).rolling(20).sum()
            v = df['Volume'].rolling(20).sum().replace(0, np.nan)
            df['VWAP20'] = pv / v

            # latest intraday price freshness
            tk = yf.Ticker(t)
            latest_price = float(df['Close'].iloc[-1])
            try:
                intraday = tk.history(period="1d", interval="1m", auto_adjust=False)
                if not intraday.empty:
                    latest_price = float(intraday['Close'].iloc[-1])
            except Exception:
                pass

            parsed = get_option_chain_cached(t, latest_price) if CACHE_AVAILABLE else parse_option_chain_with_quality(tk, latest_price)
            iv_surface = get_svi_surface_cached(t, latest_price) if CACHE_AVAILABLE else calibrate_surface_ssvi(parsed, latest_price)

            summary = {}
            for name, days in TIMEFRAMES.items():
                comp, comps = score_and_components(df, idx_df, days)
                rec_pref = recommend_with_policy(latest_price, df['Close'], comp, days, iv_surface, policy="prefer_spread")
                rec_naked = recommend_with_policy(latest_price, df['Close'], comp, days, iv_surface, policy="allow_naked")
                summary[name] = {
                    "score": round(comp, 3),
                    "components": comps,
                    "recommendation": rec_pref,
                    "recommendation_by_policy": {
                        "prefer_spread": rec_pref,
                        "allow_naked": rec_naked
                    }
                }
            img = plot_base64(df.tail(130), t)
            results[t] = {"latest": float(latest_price), "summary": summary, "chart": img, "iv_surface_fitted": bool(iv_surface)}
        except Exception:
            logger.exception("Analysis failed for %s", t)
            results[t] = {"error": "analysis_failed"}
    return jsonify({"timestamp": now_iso(), "results": results})

# Root and UI template routes
@app.route("/config-ui", methods=["GET"])
def config_ui():
    return render_template("config_ui.html")

@app.route("/output-ui", methods=["GET"])
def output_ui():
    return render_template("output_ui.html")

@app.route("/", methods=["GET"])
def index_ui():
    return render_template_string("""
    <!doctype html>
    <html><head><meta charset="utf-8"><title>Analyzer</title></head>
    <body style="font-family:system-ui;padding:2rem">
      <h2>Daily Option Strategy Analyzer</h2>
      <ul>
        <li><a href="/config-ui">Config Editor</a></li>
        <li><a href="/output-ui">Output Dashboard</a></li>
      </ul>
    </body></html>
    """)

if __name__ == "__main__":
    logger.info("Starting app. DiskCache=%s Isotonic=%s SSVI=%s ARCH=%s", CACHE_AVAILABLE, ISOTONIC_AVAILABLE, SSVI_AVAILABLE, ARCH_AVAILABLE)
    start_background_jobs()
    app.run(host="0.0.0.0", port=5000, threaded=True)
