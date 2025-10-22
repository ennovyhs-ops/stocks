#!/usr/bin/env python3
# app.py - Config separated into config.json; /config GET and POST added.

import io
import json
import base64
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import log, sqrt, exp
from scipy.stats import norm
from scipy.optimize import brentq
import os

app = Flask(__name__)

# ---------- DEFAULTS and CONFIG ----------
DEFAULT_CONFIG_PATH = "config.json"
STRIKE_STEP = 0.01
R = 0.02
Q = 0.0
TIMEFRAMES = {"next_day":1, "2_weeks":10, "3_months":65, "6_months":130}

# adjustable runtime variables (may be overridden by config.json fields)
DELTA_MAP = {
    1:  (0.50, 0.20),
    10: (0.50, 0.25),
    65: (0.48, 0.22),
    130:(0.42, 0.20)
}
PREFER_OTM = {1: False, 10: True, 65: True, 130: True}
TRADE_POLICY = "prefer_spread"

def load_config(path=DEFAULT_CONFIG_PATH):
    if not os.path.exists(path):
        sample = {"stocks":["0005.HK"], "index":"^HSI", "strike_step":STRIKE_STEP, "trade_policy":TRADE_POLICY}
        with open(path, "w") as f:
            json.dump(sample, f, indent=2)
        return sample
    with open(path, "r") as f:
        cfg = json.load(f)
    # populate missing defaults
    cfg.setdefault("stocks", ["0005.HK"])
    cfg.setdefault("index", "^HSI")
    cfg.setdefault("strike_step", STRIKE_STEP)
    cfg.setdefault("trade_policy", TRADE_POLICY)
    return cfg

def save_config(cfg, path=DEFAULT_CONFIG_PATH):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

CONFIG = load_config()

# override globals from config where applicable
STRIKE_STEP = CONFIG.get("strike_step", STRIKE_STEP)
TRADE_POLICY = CONFIG.get("trade_policy", TRADE_POLICY)

# ---------- INDICATORS AND BS HELPERS ----------
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

def annual_vol(close):
    lr = np.log(close / close.shift(1)).dropna()
    return float(lr.std() * np.sqrt(252)) if not lr.empty else 0.25

def bs_d1(S, K, r, q, sigma, T):
    return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma * np.sqrt(T) + 1e-12)

def call_delta(S,K,r,q,sigma,T):
    if T<=0 or sigma<=0: return 1.0 if S>K else 0.0
    d1 = bs_d1(S,K,r,q,sigma,T)
    return np.exp(-q*T) * norm.cdf(d1)

def put_delta(S,K,r,q,sigma,T):
    if T<=0 or sigma<=0: return -1.0 if S<K else 0.0
    d1 = bs_d1(S,K,r,q,sigma,T)
    return np.exp(-q*T) * (norm.cdf(d1)-1.0)

def bs_price_call(S,K,r,q,sigma,T):
    if T<=0 or sigma<=0: return max(S-K, 0.0)
    d1 = bs_d1(S,K,r,q,sigma,T)
    d2 = d1 - sigma*sqrt(T)
    return S*exp(-q*T)*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def bs_price_put(S,K,r,q,sigma,T):
    if T<=0 or sigma<=0: return max(K-S, 0.0)
    d1 = bs_d1(S,K,r,q,sigma,T)
    d2 = d1 - sigma*sqrt(T)
    return K*exp(-r*T)*norm.cdf(-d2) - S*exp(-q*T)*norm.cdf(-d1)

def round_strike(x):
    return float(np.round(np.round(x / STRIKE_STEP) * STRIKE_STEP, 2))

def strike_for_delta(S, target_delta, is_call, r, q, sigma, T, prefer_otm=False):
    abs_target = abs(target_delta)
    def f(K):
        return (call_delta(S,K,r,q,sigma,T) - abs_target) if is_call else (abs(put_delta(S,K,r,q,sigma,T)) - abs_target)
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

# ---------- SCORING & RECOMMEND ----------
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

def recommend(S, close_series, composite, days):
    T = max(1, days)/365.0
    sigma = annual_vol(close_series)
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
    strike_target = strike_for_delta(S, abs_target, is_call, R, Q, sigma, T, prefer_otm=prefer_otm)
    atm = round_strike(S)
    conservative = round_strike((strike_target + atm)/2.0)
    final = {"side":side, "is_call":bool(is_call), "target_delta":float(np.sign(target_delta)*abs_target),
             "strike_target":strike_target, "strike_atm":atm, "strike_conservative":conservative,
             "sigma_used":round(sigma,4), "T_years":round(T,4)}
    if is_call:
        long_price = bs_price_call(S, conservative, R, Q, sigma, T)
        short_price = bs_price_call(S, strike_target, R, Q, sigma, T)
    else:
        long_price = bs_price_put(S, conservative, R, Q, sigma, T)
        short_price = bs_price_put(S, strike_target, R, Q, sigma, T)
    if TRADE_POLICY == "prefer_spread" and final["side"] in ("LONG_CALL","LONG_PUT"):
        if is_call:
            spread_side = "BUY_CALL_SPREAD"
        else:
            spread_side = "BUY_PUT_SPREAD"
        long_leg = conservative
        short_leg = strike_target if ((is_call and strike_target>long_leg) or (not is_call and strike_target<long_leg)) else atm
        if is_call:
            long_p = bs_price_call(S, long_leg, R, Q, sigma, T)
            short_p = bs_price_call(S, short_leg, R, Q, sigma, T)
        else:
            long_p = bs_price_put(S, long_leg, R, Q, sigma, T)
            short_p = bs_price_put(S, short_leg, R, Q, sigma, T)
        spread_cost = round(long_p - short_p, 6)
        final.update({"side":spread_side, "long_strike":long_leg, "short_strike":short_leg, "long_premium":round(long_p,6), "short_premium":round(short_p,6), "spread_cost":spread_cost})
    else:
        final.update({"leg_strike":conservative, "leg_premium":round(long_price,6), "note": "Trade now." if abs(composite)>=0.3 else "Consider wait for better setup or premium."})
    return final

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
    if stocks:
        if not isinstance(stocks, list):
            return jsonify({"error":"stocks must be a list"}), 400
        if len(stocks) > 3:
            return jsonify({"error":"maximum 3 stocks allowed"}), 400
    cfg = load_config()
    if stocks is not None:
        cfg["stocks"] = stocks[:3]
    if index is not None:
        cfg["index"] = index
    if strike_step is not None:
        cfg["strike_step"] = float(strike_step)
    if trade_policy is not None:
        cfg["trade_policy"] = trade_policy
    save_config(cfg)
    # apply to runtime
    global STRIKE_STEP, TRADE_POLICY, CONFIG
    STRIKE_STEP = cfg.get("strike_step", STRIKE_STEP)
    TRADE_POLICY = cfg.get("trade_policy", TRADE_POLICY)
    CONFIG = cfg
    return jsonify(cfg)

@app.route("/analyze", methods=["POST"])
def analyze():
    body = request.get_json(force=True)
    # prefer explicit request values; fallback to config
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
        if raw.empty:
            continue
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
        summary = {}
        for name, days in TIMEFRAMES.items():
            comp, comps = score_and_components(df, idx_df, days)
            rec = recommend(float(df['Close'].iloc[-1]), df['Close'], comp, days)
            summary[name] = {"score":round(comp,3), "components":comps, "recommendation":rec}
        img = plot_base64(df.tail(130), t)
        results[t] = {"latest": float(df['Close'].iloc[-1]), "summary": summary, "chart": img}
    return jsonify({"timestamp":datetime.utcnow().isoformat()+"Z", "results":results})

# ---------- SIMPLE UI ----------
INDEX_HTML = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Daily Option Strategy Analyzer</title></head>
<body style="font-family:system-ui,Arial;max-width:900px;margin:2rem auto;padding:1rem">
  <h1>Daily Option Strategy Analyzer</h1>
  <form id="f">
    <label>Stocks (comma separated, up to 3):</label><br>
    <input id="stocks" style="width:100%"><br><br>
    <label>Benchmark index:</label><br>
    <input id="index" style="width:100%"><br><br>
    <button type="submit">Analyze</button>
    <button id="load" type="button">Load defaults</button>
    <button id="savecfg" type="button">Save as defaults</button>
  </form>
  <div id="out" style="margin-top:1rem"></div>
  <script>
  async function loadDefaults(){
    const r = await fetch('/config');
    const j = await r.json();
    document.getElementById('stocks').value = j.stocks.join(', ');
    document.getElementById('index').value = j.index;
  }
  document.getElementById('load').onclick = loadDefaults;
  document.getElementById('savecfg').onclick = async ()=>{
    const stocks = document.getElementById('stocks').value.split(',').map(s=>s.trim()).filter(Boolean);
    const index = document.getElementById('index').value.trim();
    const r = await fetch('/config', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({stocks,index})});
    const res = await r.json();
    alert('Saved defaults');
  };
  document.getElementById('f').onsubmit = async (e)=>{
    e.preventDefault();
    const stocks = document.getElementById('stocks').value.split(',').map(s=>s.trim()).filter(Boolean);
    const index = document.getElementById('index').value.trim();
    document.getElementById('out').innerText = 'Running...';
    const r = await fetch('/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({stocks, index})});
    if(!r.ok){ document.getElementById('out').innerText = 'Error: ' + await r.text(); return; }
    const j = await r.json();
    const out = document.getElementById('out');
    out.innerHTML = '<div>Updated: '+new Date(j.timestamp).toLocaleString()+'</div>';
    for(const t of Object.keys(j.results)){
      const v = j.results[t];
      const div = document.createElement('div');
      div.innerHTML = `<h2>${t} — ${v.latest}</h2>
        <img src="${v.chart}" style="max-width:100%"><pre>${JSON.stringify(v.summary,null,2)}</pre>`;
      out.appendChild(div);
    }
  };
  // load defaults on open
  loadDefaults();
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
