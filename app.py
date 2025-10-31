
#!/usr/bin/env python3
# app.py - Daily Option Strategy Analyzer using Alpha Vantage API

import io
import json
import os
import threading
import time
import logging
from datetime import datetime, timedelta
from urllib.parse import quote
from flask import Flask, request, jsonify, render_template, render_template_string
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, least_squares

# Import the new data fetching functions
from get_price import get_historical_prices_alpha_vantage, get_latest_price_alpha_vantage, get_option_chain_alpha_vantage

# Optional extras
try:
    from diskcache import Cache
    CACHE_AVAILABLE = True
except ImportError:
    Cache = None
    CACHE_AVAILABLE = False

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dos_analyzer")

# Flask app
app = Flask(__name__, template_folder="templates")

# --- Constants and Configuration ---
DEFAULT_CONFIG_PATH = "config.json"
DEFAULT_STOCK = "AAPL"
DEFAULT_INDEX = "^GSPC"
STRIKE_STEP = 0.01
R = 0.02  # Risk-free rate
Q = 0.0  # Dividend yield
TIMEFRAMES = {"next_day": 1, "2_weeks": 10, "3_months": 65, "6_months": 130}
TRADE_POLICY = "prefer_spread"
CACHE_DIR = "cache_dir"
CACHE_TTL_SECONDS = 30 * 60

# --- Caching ---
cache = Cache(CACHE_DIR) if CACHE_AVAILABLE else None
if not CACHE_AVAILABLE:
    logger.info("diskcache not available; caching disabled")

def get_cached_or_fetch(key, fetch_func, *args, **kwargs):
    if not CACHE_AVAILABLE:
        return fetch_func(*args, **kwargs)
    
    force_refresh = kwargs.pop('force_refresh', False)
    if force_refresh:
        cache.delete(key)
    
    cached_data = cache.get(key)
    if cached_data is not None:
        logger.debug(f"Cache hit for {key}")
        return cached_data
    
    logger.debug(f"Cache miss for {key}, fetching data...")
    fresh_data = fetch_func(*args, **kwargs)
    if fresh_data is not None and not (isinstance(fresh_data, pd.DataFrame) and fresh_data.empty):
        cache.set(key, fresh_data, expire=CACHE_TTL_SECONDS)
    return fresh_data

# --- Financial Calculations ---
def bs_d1(S, K, T, r, q, sigma):
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def bs_d2(S, K, T, r, q, sigma):
    return bs_d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)

def call_price_bs(S, K, T, r, q, sigma):
    d1 = bs_d1(S, K, T, r, q, sigma)
    d2 = bs_d2(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def put_price_bs(S, K, T, r, q, sigma):
    d1 = bs_d1(S, K, T, r, q, sigma)
    d2 = bs_d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def call_delta_bs(S, K, T, r, q, sigma):
    d1 = bs_d1(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.cdf(d1)

def put_delta_bs(S, K, T, r, q, sigma):
    d1 = bs_d1(S, K, T, r, q, sigma)
    return np.exp(-q * T) * (norm.cdf(d1) - 1)

def implied_vol_call(S, K, T, r, q, price):
    try:
        return brentq(lambda sigma: call_price_bs(S, K, T, r, q, sigma) - price, 1e-6, 5.0)
    except (ValueError, RuntimeError):
        return np.nan

def implied_vol_put(S, K, T, r, q, price):
    try:
        return brentq(lambda sigma: put_price_bs(S, K, T, r, q, sigma) - price, 1e-6, 5.0)
    except (ValueError, RuntimeError):
        return np.nan

# --- Indicator Calculations ---
def returns_from_close(close):
    return np.log(close / close.shift(1)).dropna()

def ewma_vol(close, span_days=60):
    r = returns_from_close(close)
    if r.empty:
        return 0.25
    lam = 2 / (span_days + 1)
    ewma_var = (r ** 2).ewm(alpha=lam, adjust=False).mean()
    return float(np.sqrt(ewma_var.iloc[-1] * 252))

def vwma(price, vol, window):
    pv = (price * vol).rolling(window=window, min_periods=1).sum()
    v = vol.rolling(window=window, min_periods=1).sum().replace(0, np.nan)
    return pv / v

def compute_rsi(close, period=14):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    ma_down = down.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    rs = ma_up / (ma_down + 1e-12)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)

def compute_macd(close):
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def vwap(df, window=20):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * df['Volume']).rolling(window).sum() / df['Volume'].rolling(window).sum()

def compute_rsr(stock_close, index_close):
    rs = stock_close / index_close
    return rs.dropna()

# --- Scoring Logic ---
def score_rsi(rsi):
    if rsi <= 30: return 1.0
    if rsi >= 70: return -1.0
    return (50 - rsi) / 20.0

def score_macd(macd_hist):
    # Simple scoring based on histogram sign
    return 1.0 if macd_hist > 0 else -1.0

def score_vwma(close, vwma20, vwma50):
    if close > vwma20 and close > vwma50: return 1.0
    if close < vwma20 and close < vwma50: return -1.0
    return 0.0

def score_bollinger(close, upper, lower):
    if close > upper: return -1.0 # Reversion expectation
    if close < lower: return 1.0 # Reversion expectation
    return 0.0

def score_volume(volume, vol20):
    return 1.0 if volume > vol20 else 0.6 # Positive bias for volume confirmation

def score_vwap(close, vwap_val):
    return np.tanh((close - vwap_val) / vwap_val) if vwap_val else 0

def score_rsr_slope(rsr, days):
    if len(rsr) < days: return 0.0
    slope = np.polyfit(range(days), rsr.tail(days), 1)[0]
    return np.tanh(slope * 100) # Scale and squash


def score_and_components(df, index_df, days):
    weights = {
        "next_day": {"rsi": 0.14, "macd": 0.16, "vwma": 0.26, "bb": 0.18, "vol": 0.10, "vwap": 0.06, "rsr": 0.10},
        "2_weeks": {"rsi": 0.16, "macd": 0.20, "vwma": 0.24, "bb": 0.15, "vol": 0.08, "vwap": 0.07, "rsr": 0.10},
        "3_months": {"rsi": 0.12, "macd": 0.22, "vwma": 0.28, "bb": 0.12, "vol": 0.06, "vwap": 0.08, "rsr": 0.12},
        "6_months": {"rsi": 0.10, "macd": 0.20, "vwma": 0.30, "bb": 0.10, "vol": 0.05, "vwap": 0.10, "rsr": 0.15},
    }
    timeframe_key = next((k for k, v in TIMEFRAMES.items() if v == days), "next_day")
    w = weights[timeframe_key]

    # Calculate indicators for the given timeframe
    close = df['Close'].iloc[-1]
    rsi = compute_rsi(df['Close'], 14).iloc[-1]
    macd, signal, hist = compute_macd(df['Close'])
    vwma20 = vwma(df['Close'], df['Volume'], 20).iloc[-1]
    vwma50 = vwma(df['Close'], df['Volume'], 50).iloc[-1]
    bb_mid = df['Close'].rolling(20).mean().iloc[-1]
    bb_std = df['Close'].rolling(20).std().iloc[-1]
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    volume = df['Volume'].iloc[-1]
    vol20 = df['Volume'].rolling(20).mean().iloc[-1]
    vwap20 = vwap(df, 20).iloc[-1]
    rsr = compute_rsr(df['Close'], index_df['Close'])

    components = {
        "rsi": score_rsi(rsi),
        "macd": score_macd(hist.iloc[-1]),
        "vwma": score_vwma(close, vwma20, vwma50),
        "bb": score_bollinger(close, bb_upper, bb_lower),
        "vol": score_volume(volume, vol20),
        "vwap": score_vwap(close, vwap20),
        "rsr": score_rsr_slope(rsr, days)
    }

    total_score = sum(w[k] * components[k] for k in w)
    norm_score = np.clip(total_score / sum(w.values()), -1.0, 1.0)

    return norm_score, components


# --- Recommendation Logic ---
def find_strike_for_delta(S, T, r, q, sigma, target_delta, is_call):
    func = call_delta_bs if is_call else put_delta_bs
    
    # Objective function to minimize |actual_delta - target_delta|
    def objective(K):
        K = K[0]
        if K <= 0: return 1e12 # Penalize invalid strikes
        delta = func(S, K, T, r, q, sigma)
        return abs(delta - target_delta)

    # Start with an initial guess for K around S
    initial_guess = [S]
    result = least_squares(objective, initial_guess, bounds=(0, np.inf))

    if result.success:
        return result.x[0]
    else:
        # Fallback for difficult cases: coarse search
        strikes = np.linspace(S * 0.5, S * 1.5, 101)
        deltas = func(S, strikes, T, r, q, sigma)
        return strikes[np.argmin(np.abs(deltas - target_delta))]

def recommend_with_policy(S, close_series, composite, days, iv_surface=None, policy="prefer_spread"):
    T_years = days / 252.0
    sigma = ewma_vol(close_series, span_days=60)
    
    # Delta targets (example values, can be configured)
    delta_map = {1: 0.50, 10: 0.55, 65: 0.60, 130: 0.65}
    target_delta_abs = delta_map.get(days, 0.50)

    rec = {}
    if composite >= 0.6:
        rec['side'] = "LONG_CALL"
        rec['is_call'] = True
        rec['target_delta'] = target_delta_abs
    elif 0.2 < composite < 0.6:
        rec['side'] = "BUY_CALL_SPREAD"
        rec['is_call'] = True
        rec['target_delta'] = target_delta_abs
    elif -0.2 <= composite <= 0.2:
        rec['side'] = "NEUTRAL"
    elif -0.6 < composite < -0.2:
        rec['side'] = "BUY_PUT_SPREAD"
        rec['is_call'] = False
        rec['target_delta'] = -target_delta_abs
    elif composite <= -0.6:
        rec['side'] = "LONG_PUT"
        rec['is_call'] = False
        rec['target_delta'] = -target_delta_abs

    if 'target_delta' in rec:
        strike_target = find_strike_for_delta(S, T_years, R, Q, sigma, rec['target_delta'], rec['is_call'])
        rec['strike_target'] = round(strike_target / STRIKE_STEP) * STRIKE_STEP
        rec['strike_atm'] = round(S / STRIKE_STEP) * STRIKE_STEP
        rec['strike_conservative'] = round(((strike_target + S) / 2) / STRIKE_STEP) * STRIKE_STEP
        rec['sigma_used'] = sigma
        rec['T_years'] = T_years
        rec['note'] = "Trade now"
        
        # Handle spread policy
        if policy == "prefer_spread" and rec['side'] in ["LONG_CALL", "LONG_PUT"]:
            rec['side'] = "BUY_CALL_SPREAD" if rec['is_call'] else "BUY_PUT_SPREAD"

    return rec

# --- Flask Endpoints ---
@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True)
    stock = body.get("stock", "").upper() or load_config().get("stock", DEFAULT_STOCK)
    index_ticker = body.get("index", "").upper() or load_config().get("index", DEFAULT_INDEX)

    results = {}
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)

    # Fetch data
    df = get_cached_or_fetch(f"hist:{stock}", get_historical_prices_alpha_vantage, stock, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    index_df = get_cached_or_fetch(f"hist:{index_ticker}", get_historical_prices_alpha_vantage, index_ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if df is None or df.empty or index_df is None or index_df.empty:
        return jsonify({"error": "Could not fetch historical data"}), 500

    latest_price_data = get_cached_or_fetch(f"price:{stock}", get_latest_price_alpha_vantage, stock)
    latest_price = latest_price_data.get('latest_price') if latest_price_data else df['Close'].iloc[-1]

    analysis_results = {}
    for timeframe_name, days in TIMEFRAMES.items():
        composite_score, components = score_and_components(df, index_df, days)
        recommendation = recommend_with_policy(latest_price, df['Close'], composite_score, days)
        analysis_results[timeframe_name] = {
            "composite_score": composite_score,
            "components": components,
            "recommendation": recommendation
        }

    results[stock] = {
        "latest": float(latest_price),
        "analysis": analysis_results
    }

    return jsonify({"timestamp": datetime.utcnow().isoformat() + "Z", "results": results})


@app.route("/config", methods=["GET"])
def get_config_api():
    return jsonify(load_config())
    
@app.route('/config', methods=['POST'])
def save_config_api():
    new_config = request.json
    if not new_config or 'stock' not in new_config or 'index' not in new_config:
        return jsonify({'error': 'Invalid config format'}), 400
    with open(DEFAULT_CONFIG_PATH, 'w') as f:
        json.dump(new_config, f, indent=2)
    return jsonify({'message': 'Config saved'})

def load_config(path=DEFAULT_CONFIG_PATH):
    if not os.path.exists(path):
        sample = {"stock": DEFAULT_STOCK, "index": DEFAULT_INDEX, "strike_step": STRIKE_STEP, "trade_policy": TRADE_POLICY}
        with open(path, "w") as f:
            json.dump(sample, f, indent=2)
        return sample
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg

@app.route("/output-ui", methods=["GET"])
def output_ui():
    return render_template("output_ui.html")

@app.route("/config-ui")
def config_ui():
    return render_template("config_ui.html")

@app.route("/")
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
    logger.info(f"Starting app with Alpha Vantage data source. Cache={CACHE_AVAILABLE}")
    app.run(host="0.0.0.0", port=5000, threaded=True)
