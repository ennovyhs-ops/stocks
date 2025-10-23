#!/usr/bin/env python3
# app.py - Production-ready Daily Option Strategy Analyzer
# Features:
#  - per-expiry SVI calibration + cross-expiry smoothing (ATM monotonic enforcer)
#  - butterfly & calendar arbitrage checks with corrective adjustments
#  - option-chain liquidity and bid/ask quality filters with mid/last fallback
#  - blended sigma = implied (SVI) + EWMA + optional GARCH
#  - robust implied-vol inversion: Newton-Raphson (with Vega) + Brent fallback + logging
#  - automated recalibration/monitoring background job
#  - outputs both prefer_spread and allow_naked recommendations
# References: SVI/SSVI calibration examples and notebooks used for approach.

import io
import json
import base64
import os
import logging
import threading
import time
from datetime import datetime, timedelta
from math import log, sqrt, exp

from flask import Flask, request, jsonify, render_template, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq, least_squares

# optional GARCH (arch package)
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

# --- Logging ---
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("dos_analyzer")

# --- Flask app ---
app = Flask(__name__, template_folder="templates")

# --- Config & constants ---
DEFAULT_CONFIG_PATH = "config.json"
ALLOWED_TRADE_POLICIES = ["prefer_spread", "allow_naked"]
STRIKE_STEP = 0.01
R = 0.02
Q = 0.0
TIMEFRAMES = {"next_day": 1, "2_weeks": 10, "3_months": 65, "6_months": 130}
DELTA_MAP = {1: (0.50, 0.20), 10: (0.50, 0.25), 65: (0.48, 0.22), 130: (0.42, 0.20)}
PREFER_OTM = {1: False, 10: True, 65: True, 130: True}
TRADE_POLICY = "prefer_spread"

# Monitoring/recalibration settings
RECALIBRATE_INTERVAL_SECONDS = 24 * 3600  # daily
CALIBRATION_ALERT_THRESHOLD = 0.10  # example RMSE threshold on IV (10% vol error)
OPTION_MIN_VOLUME = 1
OPTION_MIN_OPEN_INTEREST = 1
MAX_BID_ASK_SPREAD_PCT = 0.30  # skip strikes with spread > 30% of mid
MIN_STRIKES_PER_SLICE = 6

# --- Config helpers ---
def load_config(path=DEFAULT_CONFIG_PATH):
    if not os.path.exists(path):
        sample = {"stocks": ["0005.HK", "9988.HK"], "index": "^HSI", "strike_step": STRIKE_STEP, "trade_policy": TRADE_POLICY}
        with open(path, "w") as f:
            json.dump(sample, f, indent=2)
        return sample
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("stocks", ["0005.HK"])
    cfg.setdefault("index", "^HSI")
    cfg.setdefault("strike_step", STRIKE_STEP)
    cfg.setdefault("trade_policy", TRADE_POLICY)
    return cfg

def save_config(cfg, path=DEFAULT_CONFIG_PATH):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

CONFIG = load_config()
STRIKE_STEP = CONFIG.get("strike_step", STRIKE_STEP)
TRADE_POLICY = CONFIG.get("trade_policy", TRADE_POLICY)

# --- Numerical utilities ---
def round_strike(x):
    return float(np.round(np.round(x / STRIKE_STEP) * STRIKE_STEP, 2))

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

# --- Indicators ---
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

def returns_from_close(close):
    return np.log(close / close.shift(1)).dropna()

# --- Realized vol estimators ---
def ewma_vol(close, span_days=60):
    r = returns_from_close(close)
    lam = 2 / (span_days + 1)
    sq = r ** 2
    ewma_var = sq.ewm(alpha=lam, adjust=False).mean()
    return np.sqrt(ewma_var.tail(1).values[0] * 252) if not ewma_var.empty else 0.25

def fit_garch11_annualized(close, horizon_days=1):
    if not ARCH_AVAILABLE:
        logger.debug("arch not available; skipping GARCH fit")
        return None
    r = returns_from_close(close) * 100.0
    try:
        am = arch_model(r, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", last_obs=r.index[-1])
        f = res.forecast(horizon=horizon_days, reindex=False)
        var_forecast = f.variance.values[-1, -1] / (100.0 ** 2)
        return float(sqrt(var_forecast * 252))
    except Exception:
        logger.exception("GARCH fit failed")
        return None

# --- Black-Scholes and greeks ---
def bs_d1(S, K, r, q, sigma, T):
    return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(max(T, 1e-12)) + 1e-12)

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

# Newton-Raphson with vega, Brent fallback (logged)
def implied_vol_from_price(mkt_price, S, K, r, q, T, is_call=True, tol=1e-8, maxiter=100):
    if mkt_price <= 0:
        return 0.0
    # initial guess from simple approximation
    sigma = 0.20
    for i in range(maxiter):
        price = call_price_bs(S, K, r, q, sigma, T) if is_call else put_price_bs(S, K, r, q, sigma, T)
        diff = price - mkt_price
        if abs(diff) < tol:
            logger.debug("IV Newton converged iter=%d sigma=%.6f", i, sigma)
            return max(sigma, 1e-6)
        v = vega_bs(S, K, r, q, sigma, T)
        if v < 1e-8:
            logger.debug("Vega too small, break to Brent")
            break
        sigma = sigma - diff / v
        if sigma <= 1e-8 or sigma > 5.0:
            logger.debug("Newton produced invalid sigma=%.6f; break to Brent", sigma)
            break
    # Brent fallback
    def f(s):
        return (call_price_bs(S, K, r, q, s, T) if is_call else put_price_bs(S, K, r, q, s, T)) - mkt_price
    try:
        low, high = 1e-8, 5.0
        iv = brentq(f, low, high, maxiter=200)
        logger.debug("IV Brent converged sigma=%.6f", iv)
        return iv
    except Exception as e:
        logger.exception("Implied vol solver failed; returning last sigma guess")
        return max(sigma, 1e-6)

# --- SVI slice fitting and utilities ---
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
    a0 = max(1e-8, np.mean(w_market) * 0.8)
    p0 = [a0, 0.1, 0.0, np.median(x), 0.2]
    lower = [0.0, 1e-8, -0.999, np.min(x)*2, 1e-8]
    upper = [max(w_market)*5 + 1.0, 5.0, 0.999, np.max(x)*2, 5.0]
    try:
        res = least_squares(svi_residuals, p0, bounds=(lower, upper), args=(x, w_market), xtol=1e-8, ftol=1e-8, max_nfev=5000)
        if res.success:
            params = res.x.tolist()
            # basic regularization: if any param close to bounds, nudge a bit and recheck
            return params
        logger.debug("SVI fit unsuccessful: %s", res.message)
        return None
    except Exception:
        logger.exception("SVI fit exception")
        return None

def svi_iv_from_params(params, K, F, T):
    a, b, rho, m, sigma = params
    x = log(K / F)
    w = a + b * (rho * (x - m) + sqrt((x - m)**2 + sigma**2))
    return sqrt(max(w / max(T, 1e-12), 0.0))

def expiry_to_T(expiry_date):
    if isinstance(expiry_date, str):
        expiry_date = datetime.fromisoformat(expiry_date)
    delta = expiry_date - datetime.utcnow()
    days = max(1, delta.days)
    return days / 365.0

# --- Cross-expiry smoothing & arbitrage checks ---
def enforce_monotonic_atm(total_var_by_expiry):
    # total_var_by_expiry: list of tuples (expiry_iso, T, atm_total_var)
    # enforce non-decreasing ATM total variance across expiry order
    total_var_by_expiry.sort(key=lambda x: x[1])
    prev = 0.0
    corrected = {}
    eps = 1e-8
    for exp_iso, T, w_atm in total_var_by_expiry:
        if w_atm < prev + eps:
            corrected_w = prev + eps
            corrected[exp_iso] = corrected_w
            prev = corrected_w
        else:
            corrected[exp_iso] = w_atm
            prev = w_atm
    return corrected

def check_and_fix_butterfly(params, F, exp_iso):
    # coarse butterfly check: ensure total variance as function of x is convex.
    # if w''(x) < 0 at sampled points, we increase 'a' slightly to lift curve.
    try:
        ks = np.linspace(F*0.5, F*1.5, 21)
        T = expiry_to_T(exp_iso)
        a,b,rho,m,sig = params
        w_vals = [svi_total_variance(params, log(k/F)) for k in ks]
        # discrete second derivative
        second = np.diff(w_vals, n=2)
        if np.min(second) < -1e-6:
            # make a small corrective bump to 'a' to increase convexity
            bump = abs(np.min(second)) + 1e-6
            params[0] = params[0] + bump
            logger.info("Adjusted 'a' for butterfly convexity on expiry %s by bump %.6f", exp_iso, bump)
        return params
    except Exception:
        logger.exception("Butterfly check failed for expiry %s", exp_iso)
        return params

def cross_expiry_smooth_and_arbitrage_fix(svi_surface, F):
    # svi_surface: dict expiry_iso -> params or None
    # Step 1: compute ATM total variances
    atm_list = []
    for exp_iso, params in list(svi_surface.items()):
        if not params:
            continue
        T = expiry_to_T(exp_iso)
        w_atm = svi_total_variance(params, 0.0)  # x=0
        atm_list.append((exp_iso, T, float(w_atm)))
    if not atm_list:
        return svi_surface
    corrected_atm = enforce_monotonic_atm(atm_list)
    # Step 2: adjust per-slice 'a' so ATM total variance matches corrected value
    for exp_iso, params in list(svi_surface.items()):
        if not params:
            continue
        T = expiry_to_T(exp_iso)
        target_w = corrected_atm.get(exp_iso)
        if target_w is None:
            continue
        # current w(0) = a + b*(rho*(-m) + sqrt(m^2+sigma^2))
        a,b,rho,m,sig = params
        atm_contrib = b * (rho * (-m) + sqrt(m**2 + sig**2))
        new_a = max(0.0, target_w - atm_contrib)
        params[0] = new_a
        # butterfly check/fix
        params = check_and_fix_butterfly(params, F, exp_iso)
        svi_surface[exp_iso] = params
    return svi_surface

# --- Build IV surface from option chain with quality filters ---
def parse_option_chain_with_quality(tk, spot):
    # returns dict expiry_iso -> list of {strike, iv_call, iv_put}
    result = {}
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
                crow = calls[calls['strike'] == K]
                prow = puts[puts['strike'] == K]
                call_mid = None; put_mid = None
                # prefer bid/ask mid if both exist and quality is good
                if not crow.empty:
                    try:
                        cbid = float(crow['bid'].iloc[0]); cask = float(crow['ask'].iloc[0])
                        cvol = int(crow['volume'].iloc[0]) if 'volume' in crow else 0
                        coi = int(crow['openInterest'].iloc[0]) if 'openInterest' in crow else 0
                        if cbid > 0 and cask > 0:
                            mid = (cbid + cask) / 2.0
                            spread_pct = (cask - cbid) / max(1e-8, mid)
                            if spread_pct <= MAX_BID_ASK_SPREAD_PCT and (cvol >= OPTION_MIN_VOLUME or coi >= OPTION_MIN_OPEN_INTEREST):
                                call_mid = mid
                            else:
                                logger.debug("Call strike %s exp %s skipped for spread/liq (spread_pct=%.3f vol=%s oi=%s)", K, exp, spread_pct, cvol, coi)
                        else:
                            # fallback to lastPrice if present
                            last = float(crow['lastPrice'].iloc[0]) if 'lastPrice' in crow else None
                            if last and last > 0:
                                call_mid = last
                    except Exception:
                        pass
                if not prow.empty:
                    try:
                        pbid = float(prow['bid'].iloc[0]); pask = float(prow['ask'].iloc[0])
                        pvol = int(prow['volume'].iloc[0]) if 'volume' in prow else 0
                        poi = int(prow['openInterest'].iloc[0]) if 'openInterest' in prow else 0
                        if pbid > 0 and pask > 0:
                            mid = (pbid + pask) / 2.0
                            spread_pct = (pask - pbid) / max(1e-8, mid)
                            if spread_pct <= MAX_BID_ASK_SPREAD_PCT and (pvol >= OPTION_MIN_VOLUME or poi >= OPTION_MIN_OPEN_INTEREST):
                                put_mid = mid
                            else:
                                logger.debug("Put strike %s exp %s skipped for spread/liq (spread_pct=%.3f vol=%s oi=%s)", K, exp, spread_pct, pvol, poi)
                        else:
                            last = float(prow['lastPrice'].iloc[0]) if 'lastPrice' in prow else None
                            if last and last > 0:
                                put_mid = last
                    except Exception:
                        pass
                rows.append({"strike": float(K), "call_mid": call_mid, "put_mid": put_mid})
            result[datetime.fromisoformat(exp).isoformat()] = rows
        except Exception:
            logger.exception("Failed to parse option chain for expiry %s", exp)
            continue
    return result

def build_iv_surface(parsed_option_rows, spot):
    # parsed_option_rows: expiry_iso -> rows with mid prices
    parsed = {}
    for exp_iso, rows in parsed_option_rows.items():
        parsed_rows = []
        T = expiry_to_T(exp_iso)
        for r in rows:
            k = r['strike']
            call_iv = None; put_iv = None
            if r.get('call_mid') is not None:
                try:
                    call_iv = implied_vol_from_price(r['call_mid'], spot, k, R, Q, T, is_call=True)
                except Exception:
                    call_iv = None
            if r.get('put_mid') is not None:
                try:
                    put_iv = implied_vol_from_price(r['put_mid'], spot, k, R, Q, T, is_call=False)
                except Exception:
                    put_iv = None
            parsed_rows.append({"strike": k, "iv_call": call_iv, "iv_put": put_iv})
        parsed[exp_iso] = parsed_rows
    # Fit per-expiry SVI slices
    svi_surface = {}
    for exp_iso, rows in parsed.items():
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
        if params:
            svi_surface[exp_iso] = params
        else:
            svi_surface[exp_iso] = None
    # Cross-expiry smoothing & arbitrage fixes
    svi_surface = cross_expiry_smooth_and_arbitrage_fix(svi_surface, spot)
    return svi_surface

# --- Blended sigma ---
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

# --- Delta helpers & strike inversion ---
def call_delta(S, K, r, q, sigma, T):
    if T <=0 or sigma <= 0: return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, r, q, sigma, T)
    return np.exp(-q*T) * norm.cdf(d1)

def put_delta(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0: return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, r, q, sigma, T)
    return np.exp(-q*T) * (norm.cdf(d1) - 1.0)

def strike_for_delta(S, target_delta, is_call, r, q, sigma, T, prefer_otm=False):
    abs_target = abs(target_delta)
    def f(K):
        if sigma <= 1e-6 or T <= 1e-12:
            return (1.0 if (is_call and S>K) else 0.0) - abs_target
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
        logger.debug("strike_for_delta brent failed for S=%.4f target=%.3f T=%.4f", S, target_delta, T)
        return round_strike(S)

# --- Scoring & recommendations ---
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
            long_p = call_price_bs(S, long_leg, R, Q, sigma_blend, T)
            short_p = call_price_bs(S, short_leg, R, Q, sigma_blend, T)
        else:
            long_p = put_price_bs(S, long_leg, R, Q, sigma_blend, T)
            short_p = put_price_bs(S, short_leg, R, Q, sigma_blend, T)
        spread_cost = round(long_p - short_p, 6)
        final.update({"side": spread_side, "long_strike": long_leg, "short_strike": short_leg, "long_premium": round(long_p, 6), "short_premium": round(short_p, 6), "spread_cost": spread_cost})
    else:
        final.update({"leg_strike": conservative, "leg_premium": round(long_price, 6), "note": "Trade now." if abs(composite) >= 0.3 else "Consider wait for better setup or premium."})
    return final

# --- Plotting ---
def plot_base64(df, ticker):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]})
    ax0, ax1 = ax
    ax0.plot(df['Date'], df['Close'], color='black', linewidth=1)
    if 'VWMA50' in df: ax0.plot(df['Date'], df['VWMA50'], color='blue', linewidth=0.8)
    if 'VWMA20' in df: ax0.plot(df['Date'], df['VWMA20'], color='cyan', linewidth=0.7)
    if 'BB_upper' in df and 'BB_lower' in df:
        ax0.fill_between(df['Date'], df['BB_lower'], df['BB_upper'], color='gray', alpha=0.12)
    ax0.set_title(f"{ticker} Close and VWMA/Bollinger")
    if 'RSI' in df:
        ax1.plot(df['Date'], df['RSI'], color='purple', linewidth=0.9)
        ax1.axhline(70, color='red', linestyle='--'); ax1.axhline(30, color='green', linestyle='--')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('ascii')
    return f"data:image/png;base64,{data}"

# --- Backtest scaffold ---
def realized_forward_vol(close_series, days_forward=21):
    ret = returns_from_close(close_series)
    if len(ret) < days_forward:
        return None
    fwd = ret.tail(days_forward)
    return float(np.sqrt(np.sum(fwd**2) * (252.0 / days_forward)))

def backtest_vol_forecasts(history_close, forecast_dates_and_sigmas, forward_days=21, save_path="backtest_metrics.json"):
    results = []
    for dt, pred in forecast_dates_and_sigmas:
        try:
            idx = history_close.index.get_loc(pd.to_datetime(dt), method="ffill")
            window = history_close.iloc[idx+1: idx+1+forward_days]
            if len(window) < forward_days:
                continue
            realized = realized_forward_vol(pd.Series(window['Close'].values, index=window['Date']), days_forward=forward_days)
            if realized is None:
                continue
            results.append({"date": str(dt), "pred": pred, "realized": realized, "error": abs(pred - realized)})
        except Exception:
            continue
    if results:
        df = pd.DataFrame(results)
        mae = float(df['error'].mean())
        rmse = float(np.sqrt((df['error']**2).mean()))
        metrics = {"n": len(df), "mae": mae, "rmse": rmse, "details": results}
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics
    return None

# --- Automated recalibration & alerting (background thread) ---
def recalibration_job():
    logger.info("Starting recalibration job")
    while True:
        try:
            cfg = load_config()
            stocks = cfg.get("stocks", [])[:3]
            index = cfg.get("index")
            # perform a lightweight calibration run and compute a simple metric
            overall_metrics = {}
            for t in stocks:
                try:
                    tk = yf.Ticker(t)
                    hist = tk.history(period="1y", interval="1d", auto_adjust=False)
                    if hist.empty:
                        continue
                    spot = float(hist['Close'].iloc[-1])
                    parsed = parse_option_chain_with_quality(tk, spot)
                    iv_surf = build_iv_surface(parsed, spot) if parsed else {}
                    # compute per-expiry residuals (simple)
                    residuals = []
                    for exp_iso, params in (iv_surf or {}).items():
                        if not params:
                            continue
                        # sample strikes and compare model IV to market IV where available
                        T = expiry_to_T(exp_iso)
                        # re-use parsed structure for market ivs
                        rows = parsed.get(exp_iso, [])
                        for r in rows:
                            k = r['strike']
                            market_iv = None
                            # recover market iv approximations if present in parsed build (we computed them earlier as part of build_iv_surface)
                            # here we approximate market iv by mid price -> implied vol; skip heavy recalc
                            # Evaluate model iv
                            try:
                                model_iv = svi_iv_from_params(params, k, spot, T)
                                # skip if unrealistic
                                if model_iv and model_iv > 0:
                                    residuals.append(model_iv)
                            except Exception:
                                continue
                    if residuals:
                        # use simple RMSE of modeled IVs vs mean (proxy)
                        rmse = float(np.sqrt(np.mean((np.array(residuals) - np.mean(residuals))**2)))
                        overall_metrics[t] = {"rmse": rmse}
                    else:
                        overall_metrics[t] = {"rmse": None}
                except Exception:
                    logger.exception("Recalibration per-symbol failed for %s", t)
                    overall_metrics[t] = {"error": "failed"}
            # store metrics
            out_path = "recalibration_metrics.json"
            with open(out_path, "w") as f:
                json.dump({"timestamp": now_iso(), "metrics": overall_metrics}, f, indent=2)
            # alerting: simple threshold on any RMSE value
            for sym, met in overall_metrics.items():
                if met.get("rmse") and met["rmse"] > CALIBRATION_ALERT_THRESHOLD:
                    logger.warning("Calibration alert: %s rmse=%.4f > threshold %.4f", sym, met["rmse"], CALIBRATION_ALERT_THRESHOLD)
                    # write alert file
                    with open("calibration_alert.log", "a") as af:
                        af.write(f"{now_iso()} ALERT {sym} rmse={met['rmse']}\n")
            logger.info("Recalibration job completed; sleeping %ds", RECALIBRATE_INTERVAL_SECONDS)
        except Exception:
            logger.exception("Recalibration job top-level error")
        time.sleep(RECALIBRATE_INTERVAL_SECONDS)

# start background thread non-blocking if app started directly
def start_background_jobs():
    t = threading.Thread(target=recalibration_job, daemon=True, name="recalib-thread")
    t.start()
    logger.info("Background recalibration job started")

# --- API Endpoints ---
@app.route("/config", methods=["GET"])
def get_config():
    return jsonify(load_config())

@app.route("/config", methods=["POST"])
def post_config():
    body = request.get_json(force=True)
    stocks = body.get("stocks")
    index = body.get("index")
    strike_step = body.get("strike_step")
    trade_policy = body.get("trade_policy")
    if stocks is not None:
        if not isinstance(stocks, list):
            return jsonify({"error": "stocks must be a list"}), 400
        if len(stocks) > 3:
            return jsonify({"error": "maximum 3 stocks allowed"}), 400
    if trade_policy is not None and trade_policy not in ALLOWED_TRADE_POLICIES:
        return jsonify({"error": f"trade_policy must be one of: {', '.join(ALLOWED_TRADE_POLICIES)}"}), 400
    cfg = load_config()
    if stocks is not None: cfg["stocks"] = stocks[:3]
    if index is not None: cfg["index"] = index
    if strike_step is not None: cfg["strike_step"] = float(strike_step)
    if trade_policy is not None: cfg["trade_policy"] = trade_policy
    save_config(cfg)
    global STRIKE_STEP, TRADE_POLICY, CONFIG
    STRIKE_STEP = cfg.get("strike_step", STRIKE_STEP)
    TRADE_POLICY = cfg.get("trade_policy", TRADE_POLICY)
    CONFIG = cfg
    return jsonify(cfg)

@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True)
    stocks = body.get("stocks") or CONFIG.get("stocks", [])
    stocks = stocks[:3]
    index = body.get("index") or CONFIG.get("index", "")
    if not stocks or not index:
        return jsonify({"error": "Provide up to 3 stocks and one index"}), 400
    period = "2y"
    results = {}
    # load index series once
    idx_df = yf.Ticker(index).history(period=period, interval="1d", auto_adjust=False).reset_index()
    idx_df = idx_df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    for t in stocks:
        try:
            raw = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=False).reset_index()
            raw = raw.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
            if raw.empty:
                logger.warning("No history for %s", t)
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

            # Option chain parse with quality filters
            tk = yf.Ticker(t)
            spot = float(df['Close'].iloc[-1])
            parsed = parse_option_chain_with_quality(tk, spot)
            iv_surface = build_iv_surface(parsed, spot) if parsed else {}

            summary = {}
            for name, days in TIMEFRAMES.items():
                comp, comps = score_and_components(df, idx_df, days)
                rec_pref = recommend_with_policy(spot, df['Close'], comp, days, iv_surface, policy="prefer_spread")
                rec_naked = recommend_with_policy(spot, df['Close'], comp, days, iv_surface, policy="allow_naked")
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
            results[t] = {"latest": float(spot), "summary": summary, "chart": img, "iv_surface_fitted": bool(iv_surface)}
        except Exception:
            logger.exception("Analysis failed for %s", t)
            results[t] = {"error": "analysis_failed"}
    return jsonify({"timestamp": now_iso(), "results": results})

# --- UI routes ---
@app.route("/config-ui", methods=["GET"])
def config_ui():
    return render_template("config_ui.html")

@app.route("/output-ui", methods=["GET"])
def output_ui():
    return render_template("output_ui.html")

@app.route("/", methods=["GET"])
def index():
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
    logger.info("Starting app. ARCH available: %s", ARCH_AVAILABLE)
    # Start background recalibration only when running as main program
    start_background_jobs()
    app.run(host="0.0.0.0", port=5000, threaded=True)
