# api/index.py
# Serverless Flask entry for Vercel
# - No background threads
# - Optional Redis cache via REDIS_URL
# - Lightweight SVG charting (no Matplotlib)
# - On-demand recalibration endpoint instead of background job

import os
import json
import time
import logging
from datetime import datetime, date
from math import log, sqrt, exp
from urllib.parse import quote

from flask import Flask, request, jsonify, render_template
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, least_squares

# Optional Redis cache
REDIS_URL = os.environ.get("REDIS_URL")
try:
    if REDIS_URL:
        import redis
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    else:
        redis_client = None
except Exception:
    redis_client = None

# In-memory fallback cache (process-local, ephemeral)
_mem_cache = {}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("serverless-dos")

app = Flask(__name__, template_folder="templates")

# Basic configuration
DEFAULT_STOCKS = ["9988.HK", "0005.HK", "1810.HK"]
DEFAULT_INDEX = "^HSI"
STRIKE_STEP = 0.01
R = 0.02
Q = 0.0

TIMEFRAMES = {"next_day": 1, "2_weeks": 10, "3_months": 65, "6_months": 130}
DELTA_MAP = {1: (0.50, 0.20), 10: (0.50, 0.25), 65: (0.48, 0.22), 130: (0.42, 0.20)}
PREFER_OTM = {1: False, 10: True, 65: True, 130: True}

# Calibration / SVI settings
MIN_STRIKES_PER_SLICE = 6

# Utility
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def today_str():
    return date.today().isoformat()

def cache_get(key):
    if redis_client:
        try:
            v = redis_client.get(key)
            return json.loads(v) if v else None
        except Exception:
            logger.exception("Redis get failed")
            return None
    # in-memory
    entry = _mem_cache.get(key)
    if not entry:
        return None
    val, expiry = entry
    if expiry and time.time() > expiry:
        _mem_cache.pop(key, None)
        return None
    return val

def cache_set(key, value, ttl=None):
    if redis_client:
        try:
            redis_client.set(key, json.dumps(value), ex=ttl)
            return
        except Exception:
            logger.exception("Redis set failed")
            pass
    expiry = time.time() + ttl if ttl else None
    _mem_cache[key] = (value, expiry)

def cache_delete_prefix(prefix):
    if redis_client:
        try:
            keys = redis_client.keys(prefix + "*")
            if keys:
                redis_client.delete(*keys)
            return
        except Exception:
            logger.exception("Redis delete prefix failed")
            pass
    for k in list(_mem_cache.keys()):
        if k.startswith(prefix):
            _mem_cache.pop(k, None)

# Minimal numeric helpers (SVI, BS, IV)
def bs_d1(S, K, r, q, sigma, T):
    return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * sqrt(max(T, 1e-12)) + 1e-12)

def call_price_bs(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0: return max(S-K, 0.0)
    d1 = bs_d1(S,K,r,q,sigma,T); d2 = d1 - sigma*sqrt(T)
    return S*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def put_price_bs(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0: return max(K-S, 0.0)
    d1 = bs_d1(S,K,r,q,sigma,T); d2 = d1 - sigma*sqrt(T)
    return K*exp(-r*T)*norm.cdf(-d2) - S*exp(-q*T)*norm.cdf(-d1)

def vega_bs(S,K,r,q,sigma,T):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = bs_d1(S,K,r,q,sigma,T)
    return S * exp(-q*T) * norm.pdf(d1) * sqrt(T)

def implied_vol_from_price(mkt_price, S, K, r, q, T, is_call=True, tol=1e-8, maxiter=100):
    if mkt_price <= 0 or S <= 0 or K <= 0: return 0.0
    sigma = 0.20
    for i in range(maxiter):
        price = call_price_bs(S,K,r,q,sigma,T) if is_call else put_price_bs(S,K,r,q,sigma,T)
        diff = price - mkt_price
        if abs(diff) < tol:
            return max(sigma, 1e-6)
        v = vega_bs(S,K,r,q,sigma,T)
        if v < 1e-8: break
        sigma = sigma - diff / v
        if sigma <= 1e-8 or sigma > 5.0: break
    def f(s): return (call_price_bs(S,K,r,q,s,T) if is_call else put_price_bs(S,K,r,q,s,T)) - mkt_price
    try:
        iv = brentq(f, 1e-8, 5.0, maxiter=200)
        return iv
    except Exception:
        return max(sigma, 1e-6)

# SVI helpers
def svi_total_variance(params, x):
    a,b,rho,m,sig = params
    return a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sig**2))

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
        if res.success: return res.x.tolist()
    except Exception:
        logger.exception("SVI slice fit failed")
    return None

def svi_iv_from_params(params, K, F, T):
    a,b,rho,m,sig = params
    x = log(K / F)
    w = a + b * (rho * (x - m) + sqrt((x - m)**2 + sig**2))
    return sqrt(max(w / max(T, 1e-12), 0.0))

def expiry_to_T(expiry_date):
    if isinstance(expiry_date, str):
        expiry_date = datetime.fromisoformat(expiry_date)
    delta = expiry_date - datetime.utcnow()
    days = max(1, delta.days)
    return days / 365.0

# Option parsing
def parse_option_chain_with_quality(tk, spot):
    out = {}
    try:
        expiries = getattr(tk, "options", []) or []
    except Exception:
        return {}
    for exp in expiries[:8]:
        try:
            chain = tk.option_chain(exp)
            calls = chain.calls; puts = chain.puts
            strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
            rows = []
            for K in strikes:
                call_mid = None; put_mid = None
                crow = calls[calls['strike'] == K]; prow = puts[puts['strike'] == K]
                if not crow.empty:
                    try:
                        cbid = float(crow['bid'].iloc[0]); cask = float(crow['ask'].iloc[0])
                        if cbid > 0 and cask > 0:
                            mid = (cbid + cask) / 2.0
                            spread_pct = (cask - cbid) / max(1e-8, mid)
                            vol = int(crow.get('volume', [0]).iloc[0]) if 'volume' in crow else 0
                            oi = int(crow.get('openInterest', [0]).iloc[0]) if 'openInterest' in crow else 0
                            if spread_pct <= 0.30 and (vol >= 1 or oi >= 1):
                                call_mid = mid
                        if call_mid is None:
                            last = float(crow.get('lastPrice', np.nan).iloc[0]) if 'lastPrice' in crow else None
                            if last and last > 0: call_mid = last
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
                            if spread_pct <= 0.30 and (vol >= 1 or oi >= 1):
                                put_mid = mid
                        if put_mid is None:
                            last = float(prow.get('lastPrice', np.nan).iloc[0]) if 'lastPrice' in prow else None
                            if last and last > 0: put_mid = last
                    except Exception:
                        pass
                rows.append({"strike": float(K), "call_mid": call_mid, "put_mid": put_mid})
            out[datetime.fromisoformat(exp).isoformat()] = rows
        except Exception:
            logger.exception("Failed parsing option chain %s", exp)
            continue
    return out

# Surface calibration (per-request)
def enforce_atm_monotonic_isotonic(svi_surface):
    items = []
    for exp_iso, params in svi_surface.items():
        if not params: continue
        T = expiry_to_T(exp_iso)
        w_atm = svi_total_variance(params, 0.0)
        items.append((exp_iso, T, float(w_atm)))
    if not items: return svi_surface
    items.sort(key=lambda x: x[1])
    Ts = np.array([x[1] for x in items]); ws = np.array([x[2] for x in items])
    try:
        # best effort isotonic if sklearn present
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
        w_iso = ir.fit_transform(Ts, ws)
        for (exp_iso, _, _), w_target in zip(items, w_iso):
            params = svi_surface.get(exp_iso)
            if not params: continue
            a,b,rho,m,sig = params
            atm_contrib = b * (rho * (-m) + sqrt(m**2 + sig**2))
            params[0] = max(0.0, float(w_target) - atm_contrib)
            svi_surface[exp_iso] = params
    except Exception:
        # fallback: simple non-decreasing lift
        prev = -1.0
        for exp_iso, T, w in items:
            params = svi_surface.get(exp_iso)
            if not params: continue
            if w < prev:
                bump = prev - w + 1e-8
                params[0] = params[0] + bump
                svi_surface[exp_iso] = params
                w = w + bump
            prev = w
    return svi_surface

def calibrate_surface(parsed_option_rows, spot):
    svi_surface = {}
    for exp_iso, rows in parsed_option_rows.items():
        strikes = []; ivs = []
        for r in rows:
            k = r['strike']
            iv = r.get('call_mid') and implied_vol_from_price(r['call_mid'], spot, k, R, Q, expiry_to_T(exp_iso), is_call=True) or (
                 r.get('put_mid') and implied_vol_from_price(r['put_mid'], spot, k, R, Q, expiry_to_T(exp_iso), is_call=False))
            if iv and iv > 0:
                strikes.append(k); ivs.append(iv)
        if len(strikes) < MIN_STRIKES_PER_SLICE:
            svi_surface[exp_iso] = None
            continue
        params = fit_svi_slice(strikes, ivs, spot, expiry_to_T(exp_iso))
        svi_surface[exp_iso] = params
    svi_surface = enforce_atm_monotonic_isotonic(svi_surface)
    return svi_surface

# Lightweight SVG chart
def generate_svg_chart(df, ticker, width=420, height=260, padding=6):
    close = np.array(df['Close'].astype(float))
    if close.size == 0: return None
    w = width - 2 * padding; h = height - 2 * padding
    xs = np.linspace(padding, padding + w, num=close.size)
    mn, mx = float(np.min(close)), float(np.max(close))
    if mx - mn < 1e-8:
        ys = np.full_like(xs, padding + h / 2.0)
    else:
        ys = padding + (mx - close) / (mx - mn) * h
    pts = ["{:.2f},{:.2f}".format(x, y) for x, y in zip(xs, ys)]
    path = "M " + " L ".join(pts)
    stroke = "#1f77b4"
    title = f"{ticker} Close"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <rect width="100%" height="100%" fill="#ffffff"/>
      <g>
        <path d="{path}" fill="none" stroke="{stroke}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
      </g>
      <text x="{padding+2}" y="{padding+12}" font-family="Arial" font-size="12" fill="#222">{title}</text>
      <text x="{width-padding-120}" y="{padding+12}" font-family="Arial" font-size="11" fill="#666">latest {float(close[-1]):.2f}</text>
    </svg>'''
    return "data:image/svg+xml;utf8," + quote(svg)

# Simple indicators used by recommendation logic
def returns_from_close(close):
    return np.log(close / close.shift(1)).dropna()

def ewma_vol(close, span_days=60):
    r = returns_from_close(close)
    if r.empty: return 0.25
    lam = 2 / (span_days + 1)
    ewma_var = (r ** 2).ewm(alpha=lam, adjust=False).mean()
    return float(np.sqrt(ewma_var.iloc[-1] * 252))

def round_strike(x):
    return float(np.round(np.round(x / STRIKE_STEP) * STRIKE_STEP, 2))

def vwma(price, vol, window):
    pv = (price * vol).rolling(window=window, min_periods=1).sum()
    v = vol.rolling(window=window, min_periods=1).sum().replace(0, np.nan)
    return pv / v

def ema(series, span): return series.ewm(span=span, adjust=False).mean()

def compute_rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean(); ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-12); rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)

def compute_macd(close):
    fast = ema(close, 12); slow = ema(close, 26)
    macd = fast - slow; signal = macd.ewm(span=9, adjust=False).mean(); hist = macd - signal
    return macd, signal, hist

def score_and_components(df, index_df, days):
    window = df.tail(days)
    if window.empty: return 0.0, {}
    latest = window.iloc[-1]; comps = {}
    comps['rsi'] = 1.0 if latest['RSI'] <= 30 else (-1.0 if latest['RSI'] >= 70 else (50 - latest['RSI']) / 50.0)
    mdiff = latest['MACD'] - latest['MACD_sign']; comps['macd'] = 1.0 if mdiff > 0 else -1.0
    comps['vwma'] = 1.0 if latest['Close'] > latest['VWMA50'] else -1.0
    comps['bb'] = 1.0 if latest['Close'] > latest['BB_upper'] else (-1.0 if latest['Close'] < latest['BB_lower'] else np.tanh((latest['Close'] - latest['BB_mid']) / (latest['BB_std'] + 1e-9) / 2.0))
    comps['vol'] = 1.0 if latest['Volume'] > latest['Vol20'] else 0.6
    comps['vwap'] = np.tanh((latest['Close'] - latest.get('VWAP20', latest['Close'])) / (latest.get('VWAP20', latest['Close']) + 1e-9) * 10.0)
    rsr = (df['Close'] / index_df['Close']).dropna()
    slope = 0.0
    if len(rsr) >= 3:
        y = rsr.tail(min(len(rsr), days)).values; x = np.arange(len(y)); slope = np.polyfit(x, y, 1)[0]
    comps['rsr'] = np.tanh(slope * 100.0)
    weights = {'rsi': 0.14, 'macd': 0.16, 'vwma': 0.26, 'bb': 0.18, 'vol': 0.10, 'vwap': 0.06, 'rsr': 0.10}
    total = sum(weights[k] * comps.get(k, 0.0) for k in weights); wsum = sum(abs(v) for v in weights.values())
    composite = float(np.clip(total / wsum, -1.0, 1.0))
    return composite, comps

def recommend_with_policy(S, close_series, composite, days, iv_surface=None, policy="prefer_spread"):
    T = max(1, days) / 365.0
    expiry = next(iter(iv_surface.keys()), None) if iv_surface else None
    sigma_blend = ewma_vol(close_series, span_days=60)
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
    strike_target = round_strike(S)
    atm = round_strike(S); conservative = round_strike((strike_target + atm) / 2.0)
    final = {"side": side, "is_call": bool(is_call), "target_delta": float(np.sign(target_delta) * abs_target),
             "strike_target": strike_target, "strike_atm": atm, "strike_conservative": conservative,
             "sigma_used": round(sigma_blend, 4), "T_years": round(T, 4)}
    long_price = call_price_bs(S, conservative, R, Q, sigma_blend, T) if is_call else put_price_bs(S, conservative, R, Q, sigma_blend, T)
    short_price = call_price_bs(S, strike_target, R, Q, sigma_blend, T) if is_call else put_price_bs(S, strike_target, R, Q, sigma_blend, T)
    if policy == "prefer_spread" and final["side"] in ("LONG_CALL", "LONG_PUT"):
        spread_cost = round(long_price - short_price, 6)
        final.update({"side": "BUY_CALL_SPREAD" if is_call else "BUY_PUT_SPREAD",
                      "long_strike": conservative, "short_strike": strike_target,
                      "long_premium": round(long_price, 6), "short_premium": round(short_price, 6), "spread_cost": spread_cost})
    else:
        final.update({"leg_strike": conservative, "leg_premium": round(long_price, 6)})
    return final

# Endpoints
@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True) or {}
    stocks = body.get("stocks") or DEFAULT_STOCKS
    idx_sym = body.get("index") or DEFAULT_INDEX
    stocks = stocks[:5]
    period = "2y"
    results = {}
    idx_df = yf.Ticker(idx_sym).history(period=period, interval="1d", auto_adjust=False).reset_index()
    idx_df = idx_df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    for t in stocks:
        try:
            raw = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=False).reset_index()
            raw = raw.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
            if raw.empty:
                results[t] = {"error": "no_history"}; continue
            df = raw.copy()
            df['VWMA20'] = vwma(df['Close'], df['Volume'], 20)
            df['VWMA50'] = vwma(df['Close'], df['Volume'], 50)
            df['BB_mid'] = df['Close'].rolling(20).mean(); df['BB_std'] = df['Close'].rolling(20).std()
            df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']; df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
            df['RSI'] = compute_rsi(df['Close'], 14); macd, sig, hist = compute_macd(df['Close'])
            df['MACD'] = macd; df['MACD_sign'] = sig; df['MACD_hist'] = hist
            df['Vol20'] = df['Volume'].rolling(20).mean()
            tp = (df['High'] + df['Low'] + df['Close']) / 3.0; pv = (tp * df['Volume']).rolling(20).sum()
            v = df['Volume'].rolling(20).sum().replace(0, np.nan); df['VWAP20'] = pv / v

            latest_price = float(df['Close'].iloc[-1])
            try:
                intraday = yf.Ticker(t).history(period="1d", interval="1m", auto_adjust=False)
                if not intraday.empty: latest_price = float(intraday['Close'].iloc[-1])
            except Exception:
                pass

            # Option chain and SVI per-request; cache short-lived
            opt_key = f"optchain:{t}:{today_str()}"
            parsed = cache_get(opt_key)
            if parsed is None:
                parsed = parse_option_chain_with_quality(yf.Ticker(t), latest_price)
                cache_set(opt_key, parsed, ttl=60*30)
            svi_key = f"svi_surf:{t}:{today_str()}"
            svi = cache_get(svi_key)
            if svi is None:
                svi = calibrate_surface(parsed, latest_price) if parsed else {}
                cache_set(svi_key, svi, ttl=60*30)

            summary = {}
            for name, days in TIMEFRAMES.items():
                comp, comps = score_and_components(df, idx_df, days)
                rec_pref = recommend_with_policy(latest_price, df['Close'], comp, days, svi, policy="prefer_spread")
                rec_naked = recommend_with_policy(latest_price, df['Close'], comp, days, svi, policy="allow_naked")
                summary[name] = {"score": round(comp, 3), "components": comps, "recommendation_by_policy": {"prefer_spread": rec_pref, "allow_naked": rec_naked}}
            img = generate_svg_chart(df.tail(130), t)
            results[t] = {"latest": float(latest_price), "summary": summary, "chart": img, "iv_surface_fitted": bool(svi)}
        except Exception:
            logger.exception("Analysis failed for %s", t)
            results[t] = {"error": "analysis_failed"}
    return jsonify({"timestamp": now_iso(), "results": results})

@app.route("/clear-cache", methods=["POST"])
def clear_cache():
    # Clears Redis keys for today's caches or in-memory
    prefix = request.json.get("prefix", "") if request.is_json else ""
    try:
        if redis_client:
            # careful on production; here we remove todays keys with common prefixes
            keys = redis_client.keys(f"{prefix}*")
            if keys: redis_client.delete(*keys)
        else:
            cache_delete_prefix(prefix)
        return jsonify({"cleared": True})
    except Exception:
        logger.exception("clear cache failed")
        return jsonify({"cleared": False}), 500

@app.route("/recalibrate", methods=["POST"])
def recalibrate_now():
    # Run a limited recalibration on-demand. Avoid long loops in serverless context.
    stocks = request.json.get("stocks") if request.is_json else DEFAULT_STOCKS
    stocks = stocks[:3]
    results = {}
    for t in stocks:
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period="1y", interval="1d", auto_adjust=False)
            if hist.empty:
                results[t] = {"error": "no_history"}
                continue
            spot = float(hist['Close'].iloc[-1])
            parsed = parse_option_chain_with_quality(tk, spot)
            svi = calibrate_surface(parsed, spot) if parsed else {}
            results[t] = {"fitted_expiries": list(svi.keys())}
        except Exception:
            logger.exception("recal failed for %s", t)
            results[t] = {"error": "failed"}
    return jsonify({"timestamp": now_iso(), "metrics": results})

@app.route("/", methods=["GET"])
def root_ui():
    return render_template("output_ui.html")

# For local dev convenience
if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)), debug=True)
