#!/usr/bin/env python3
# app.py - Daily Option Strategy Analyzer (outputs both prefer_spread and allow_naked recommendations)

import io
import json
import base64
import os
import logging
from datetime import datetime
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

# optional GARCH dependency
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

app = Flask(__name__, template_folder="templates")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dos_analyzer")

# ---------- CONFIG ----------
DEFAULT_CONFIG_PATH = "config.json"
ALLOWED_TRADE_POLICIES = ["prefer_spread", "allow_naked"]

STRIKE_STEP = 0.01
R = 0.02
Q = 0.0
TIMEFRAMES = {"next_day": 1, "2_weeks": 10, "3_months": 65, "6_months": 130}
DELTA_MAP = {1: (0.50, 0.20), 10: (0.50, 0.25), 65: (0.48, 0.22), 130: (0.42, 0.20)}
PREFER_OTM = {1: False, 10: True, 65: True, 130: True}
TRADE_POLICY = "prefer_spread"

def load_config(path=DEFAULT_CONFIG_PATH):
    if not os.path.exists(path):
        sample = {"stocks": ["0005.HK"], "index": "^HSI", "strike_step": STRIKE_STEP, "trade_policy": TRADE_POLICY}
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

# ---------- HELPERS: INDICATORS & VOL ----------
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

def ewma_vol(close, span_days=60):
    r = returns_from_close(close)
    lam = 2 / (span_days + 1)
    sq = r ** 2
    ewma_var = sq.ewm(alpha=lam, adjust=False).mean()
    return np.sqrt(ewma_var.tail(1).values[0] * 252) if not ewma_var.empty else 0.25

def fit_garch11_annualized(close, horizon_days=1):
    if not ARCH_AVAILABLE:
        return None
    r = returns_from_close(close) * 100.0
    try:
        am = arch_model(r, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off", last_obs=r.index[-1])
        f = res.forecast(horizon=horizon_days, reindex=False)
        var_forecast = f.variance.values[-1, -1] / (100.0 ** 2)
        annualized = sqrt(var_forecast * 252)
        return float(annualized)
    except Exception:
        return None

# ---------- BLACK-SCHOLES & INVERSION ----------
def bs_d1(S, K, r, q, sigma, T):
    return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * sqrt(max(T, 1e-12)) + 1e-12)

def call_price_bs(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0: return max(S - K, 0.0)
    d1 = bs_d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma*sqrt(T)
    return S*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def put_price_bs(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0: return max(K - S, 0.0)
    d1 = bs_d1(S, K, r, q, sigma, T)
    d2 = d1 - sigma*sqrt(T)
    return K*exp(-r*T)*norm.cdf(-d2) - S*exp(-q*T)*norm.cdf(-d1)

def vega_bs(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0: return 0.0
    d1 = bs_d1(S, K, r, q, sigma, T)
    return S * exp(-q*T) * norm.pdf(d1) * sqrt(T)

def implied_vol_from_price(mkt_price, S, K, r, q, T, is_call=True, tol=1e-6, maxiter=100):
    if mkt_price <= 0:
        return 0.0
    sigma = 0.2 if S > 0 else 0.3
    for i in range(maxiter):
        price = call_price_bs(S, K, r, q, sigma, T) if is_call else put_price_bs(S, K, r, q, sigma, T)
        diff = price - mkt_price
        if abs(diff) < tol:
            return max(sigma, 1e-6)
        v = vega_bs(S, K, r, q, sigma, T)
        if v < 1e-8:
            break
        sigma = sigma - diff / v
        if sigma <= 0 or sigma > 5:
            break
    def f(s):
        return (call_price_bs(S, K, r, q, s, T) if is_call else put_price_bs(S, K, r, q, s, T)) - mkt_price
    try:
        low, high = 1e-6, 5.0
        return brentq(f, low, high, maxiter=200)
    except Exception:
        return max(sigma, 1e-6)

# ---------- SVI (per-expiry) ----------
def svi_total_variance(params, x):
    a, b, rho, m, sigma = params
    return a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))

def svi_residuals(params, x, w_market):
    w_model = svi_total_variance(params, x)
    return w_model - w_market

def fit_svi_slice(strikes, ivs, F, T):
    if T <= 0 or len(strikes) < 4:
        return None
    x = np.log(np.array(strikes) / float(F))
    w_market = (np.array(ivs) ** 2) * T
    a0 = np.maximum(0.0001, np.mean(w_market) * 0.8)
    b0 = 0.1
    rho0 = 0.0
    m0 = np.median(x)
    sigma0 = 0.2
    p0 = [a0, b0, rho0, m0, sigma0]
    lower = [0.0, 1e-8, -0.999, np.min(x)*2, 1e-8]
    upper = [np.max(w_market)*5 + 1.0, 5.0, 0.999, np.max(x)*2, 5.0]
    try:
        res = least_squares(svi_residuals, p0, bounds=(lower, upper), args=(x, w_market), xtol=1e-8, ftol=1e-8, max_nfev=2000)
        if res.success:
            return res.x.tolist()
        return None
    except Exception:
        return None

def svi_slice_iv_from_params(params, K, F, T):
    a, b, rho, m, sigma = params
    x = log(K / F)
    w = a + b * (rho * (x - m) + sqrt((x - m)**2 + sigma**2))
    iv = sqrt(max(w / max(T, 1e-12), 0.0))
    return iv

def expiry_to_T(expiry_date):
    if isinstance(expiry_date, str):
        expiry_date = datetime.fromisoformat(expiry_date)
    delta = expiry_date - datetime.utcnow()
    days = max(1, delta.days)
    return days / 365.0

def build_iv_surface_from_option_chain(option_chain_data, F):
    surface = {}
    for expiry, rows in option_chain_data.items():
        if not rows:
            surface[expiry] = None
            continue
        strikes = []
        ivs = []
        for r in rows:
            k = r.get("strike")
            iv_call = r.get("iv_call")
            iv_put = r.get("iv_put")
            if iv_call is not None:
                strikes.append(k); ivs.append(iv_call)
            elif iv_put is not None:
                strikes.append(k); ivs.append(iv_put)
        if len(strikes) < 4:
            surface[expiry] = None
            continue
        T = expiry_to_T(expiry)
        params = fit_svi_slice(strikes, ivs, F, T)
        surface[expiry] = params
    return surface

# ---------- BLENDED VOLATILITY ----------
def blended_sigma(close_series, option_surface=None, S=None, K=None, expiry=None, weights=None):
    if weights is None:
        weights = {"implied": 0.6, "ewma": 0.25, "garch": 0.15}
    ewma = ewma_vol(close_series, span_days=60)
    garch = fit_garch11_annualized(close_series, horizon_days=1) if ARCH_AVAILABLE else None
    implied = None
    if option_surface and expiry and S and K:
        params = option_surface.get(expiry)
        if params:
            T = expiry_to_T(expiry)
            implied = svi_slice_iv_from_params(params, K, F=max(S, 1.0), T=T)
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

# ---------- DELTA HELPERS ----------
def call_delta(S,K,r,q,sigma,T):
    if T<=0 or sigma<=0: return 1.0 if S>K else 0.0
    d1 = bs_d1(S,K,r,q,sigma,T)
    return np.exp(-q*T) * norm.cdf(d1)

def put_delta(S,K,r,q,sigma,T):
    if T<=0 or sigma<=0: return -1.0 if S<K else 0.0
    d1 = bs_d1(S,K,r,q,sigma,T)
    return np.exp(-q*T) * (norm.cdf(d1)-1.0)

def round_strike(x):
    return float(np.round(np.round(x / STRIKE_STEP) * STRIKE_STEP, 2))

def strike_for_delta_brent(S, target_delta, is_call, r, q, sigma, T, prefer_otm=False):
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
        K = brentq(f, low, high, maxiter=200)
        return round_strike(K)
    except Exception:
        return round_strike(S)

# ---------- SCORING ----------
def score_and_components(df, index_df, days):
    window = df.tail(days)
    if window.empty:
        return 0.0, {}
    latest = window.iloc[-1]
    comps = {}
    comps['rsi'] = 1.0 if latest['RSI']<=30 else (-1.0 if latest['RSI']>=70 else (50-latest['RSI'])/50.0)
    mdiff = latest['MACD'] - latest['MACD_sign']
    comps['macd'] = 1.0 if mdiff>0 else -1.0
    comps['vwma'] = 1.0 if latest['Close']>latest['VWMA50'] else -1.0
    comps['bb'] = 1.0 if latest['Close']>latest['BB_upper'] else (-1.0 if latest['Close']<latest['BB_lower'] else np.tanh((latest['Close']-latest['BB_mid'])/(latest['BB_std']+1e-9)/2.0))
    comps['vol'] = 1.0 if latest['Volume']>latest['Vol20'] else 0.6
    comps['vwap'] = np.tanh((latest['Close']-latest.get('VWAP20', latest['Close']))/(latest.get('VWAP20', latest['Close'])+1e-9)*10.0)
    rsr = (df['Close'] / index_df['Close']).dropna()
    slope = 0.0
    if len(rsr) >= 3:
        y = rsr.tail(min(len(rsr), days)).values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
    comps['rsr'] = np.tanh(slope * 100.0)
    weights = {'rsi':0.14,'macd':0.16,'vwma':0.26,'bb':0.18,'vol':0.10,'vwap':0.06,'rsr':0.10}
    total = sum(weights[k]*comps.get(k,0.0) for k in weights)
    wsum = sum(abs(v) for v in weights.values())
    composite = float(np.clip(total/wsum, -1.0, 1.0))
    return composite, comps

# ---------- RECOMMENDATION: compute per-policy ----------
def recommend_with_policy(S, close_series, composite, days, option_surface=None, policy="prefer_spread"):
    T = max(1, days)/365.0
    expiry = None
    if option_surface:
        expiry = next(iter(option_surface.keys()), None)
    sigma_blend = blended_sigma(close_series, option_surface, S=S, K=S, expiry=expiry)
    nearest = min(DELTA_MAP.keys(), key=lambda k: abs(k-days))
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
        side = "NEUTRAL_SELL_CREDIT"; is_call = composite>0; target_delta = sell_delta if composite>0 else -sell_delta
    abs_target = abs(target_delta)
    strike_target = strike_for_delta_brent(S, abs_target, is_call, R, Q, sigma_blend, T, prefer_otm=prefer_otm)
    atm = round_strike(S)
    conservative = round_strike((strike_target + atm)/2.0)
    result = {"side":side, "is_call":bool(is_call), "target_delta":float(np.sign(target_delta)*abs_target),
              "strike_target":strike_target, "strike_atm":atm, "strike_conservative":conservative,
              "sigma_used":round(sigma_blend,4), "T_years":round(T,4)}
    if is_call:
        long_price = call_price_bs(S, conservative, R, Q, sigma_blend, T)
        short_price = call_price_bs(S, strike_target, R, Q, sigma_blend, T)
    else:
        long_price = put_price_bs(S, conservative, R, Q, sigma_blend, T)
        short_price = put_price_bs(S, strike_target, R, Q, sigma_blend, T)
    if policy == "prefer_spread" and result["side"] in ("LONG_CALL","LONG_PUT"):
        spread_side = "BUY_CALL_SPREAD" if is_call else "BUY_PUT_SPREAD"
        long_leg = conservative
        short_leg = strike_target if ((is_call and strike_target>long_leg) or (not is_call and strike_target<long_leg)) else atm
        if is_call:
            long_p = call_price_bs(S, long_leg, R, Q, sigma_blend, T); short_p = call_price_bs(S, short_leg, R, Q, sigma_blend, T)
        else:
            long_p = put_price_bs(S, long_leg, R, Q, sigma_blend, T); short_p = put_price_bs(S, short_leg, R, Q, sigma_blend, T)
        spread_cost = round(long_p - short_p, 6)
        result.update({"side":spread_side, "long_strike":long_leg, "short_strike":short_leg, "long_premium":round(long_p,6), "short_premium":round(short_p,6), "spread_cost":spread_cost})
    else:
        result.update({"leg_strike":conservative, "leg_premium":round(long_price,6), "note": "Trade now." if abs(composite)>=0.3 else "Consider wait for better setup or premium."})
    return result

# ---------- PLOTTING ----------
def plot_base64(df, ticker):
    fig, ax = plt.subplots(2,1, figsize=(8,6), gridspec_kw={'height_ratios':[3,1]})
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

# ---------- API ENDPOINTS ----------
@app.route("/config", methods=["GET"])
def get_config():
    cfg = load_config()
    return jsonify(cfg)

@app.route("/config", methods=["POST"])
def post_config():
    body = request.get_json(force=True)
    stocks = body.get("stocks")
    index = body.get("index")
    strike_step = body.get("strike_step")
    trade_policy = body.get("trade_policy")
    if stocks is not None:
        if not isinstance(stocks, list): return jsonify({"error":"stocks must be a list"}), 400
        if len(stocks) > 3: return jsonify({"error":"maximum 3 stocks allowed"}), 400
    if trade_policy is not None and trade_policy not in ALLOWED_TRADE_POLICIES:
        return jsonify({"error":f"trade_policy must be one of: {', '.join(ALLOWED_TRADE_POLICIES)}"}), 400
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
        return jsonify({"error":"Provide up to 3 stocks and one index"}), 400
    period = "2y"
    results = {}
    idx_df = yf.Ticker(index).history(period=period, interval="1d", auto_adjust=False).reset_index()
    idx_df = idx_df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
    for t in stocks:
        raw = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=False).reset_index()
        raw = raw.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
        if raw.empty: continue
        df = raw.copy()
        df['VWMA20'] = vwma(df['Close'], df['Volume'], 20)
        df['VWMA50'] = vwma(df['Close'], df['Volume'], 50)
        df['BB_mid'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_upper'] = df['BB_mid'] + 2*df['BB_std']
        df['BB_lower'] = df['BB_mid'] - 2*df['BB_std']
        df['RSI'] = compute_rsi(df['Close'], 14)
        macd, signal, hist = compute_macd(df['Close'])
        df['MACD'] = macd; df['MACD_sign'] = signal; df['MACD_hist'] = hist
        df['Vol20'] = df['Volume'].rolling(20).mean()
        tp = (df['High'] + df['Low'] + df['Close'])/3.0
        pv = (tp * df['Volume']).rolling(20).sum()
        v = df['Volume'].rolling(20).sum().replace(0, np.nan)
        df['VWAP20'] = pv / v

        # Option chain -> IV surface (best-effort)
        option_surface = {}
        try:
            tk = yf.Ticker(t)
            expiries = getattr(tk, "options", []) or []
            F = float(df['Close'].iloc[-1])
            for exp in expiries[:6]:
                try:
                    chain = tk.option_chain(exp)
                    calls = chain.calls
                    puts = chain.puts
                    rows = []
                    strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
                    T = expiry_to_T(exp)
                    for K in strikes:
                        call_mid = None; put_mid = None
                        crow = calls[calls['strike']==K]
                        prow = puts[puts['strike']==K]
                        if not crow.empty:
                            call_bid = float(crow['bid'].iloc[0]); call_ask = float(crow['ask'].iloc[0])
                            if call_bid>0 and call_ask>0: call_mid = (call_bid+call_ask)/2.0
                        if not prow.empty:
                            put_bid = float(prow['bid'].iloc[0]); put_ask = float(prow['ask'].iloc[0])
                            if put_bid>0 and put_ask>0: put_mid = (put_bid+put_ask)/2.0
                        rows.append({"strike": float(K), "call_mid": call_mid, "put_mid": put_mid})
                    option_surface[datetime.fromisoformat(exp).isoformat()] = rows
                except Exception:
                    continue
        except Exception:
            option_surface = {}

        iv_surface = {}
        if option_surface:
            parsed = {}
            for exp, rows in option_surface.items():
                parsed_rows = []
                T = expiry_to_T(exp)
                S_now = float(df['Close'].iloc[-1])
                for r in rows:
                    k = r['strike']
                    call_iv = None; put_iv = None
                    if r.get('call_mid') is not None:
                        try:
                            call_iv = implied_vol_from_price(r['call_mid'], S_now, k, R, Q, T, is_call=True)
                        except Exception:
                            call_iv = None
                    if r.get('put_mid') is not None:
                        try:
                            put_iv = implied_vol_from_price(r['put_mid'], S_now, k, R, Q, T, is_call=False)
                        except Exception:
                            put_iv = None
                    parsed_rows.append({"strike":k, "iv_call": call_iv, "iv_put": put_iv})
                parsed[exp] = parsed_rows
            iv_surface = build_iv_surface_from_option_chain(parsed, float(df['Close'].iloc[-1]))

        summary = {}
        for name, days in TIMEFRAMES.items():
            comp, comps = score_and_components(df, idx_df, days)
            rec_pref = recommend_with_policy(float(df['Close'].iloc[-1]), df['Close'], comp, days, option_surface=iv_surface, policy="prefer_spread")
            rec_naked = recommend_with_policy(float(df['Close'].iloc[-1]), df['Close'], comp, days, option_surface=iv_surface, policy="allow_naked")
            # keep legacy "recommendation" for backward compatibility — choose prefer_spread as default
            summary[name] = {
                "score": round(comp,3),
                "components": comps,
                "recommendation": rec_pref,
                "recommendation_by_policy": {
                    "prefer_spread": rec_pref,
                    "allow_naked": rec_naked
                }
            }
        img = plot_base64(df.tail(130), t)
        results[t] = {"latest": float(df['Close'].iloc[-1]), "summary": summary, "chart": img}
    return jsonify({"timestamp": datetime.utcnow().isoformat()+"Z", "results":results})

# ---------- UI Routes ----------
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
    logger.info("Starting app (ARCH available: %s)", ARCH_AVAILABLE)
    app.run(host="0.0.0.0", port=5000)
