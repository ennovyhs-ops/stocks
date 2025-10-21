#!/usr/bin/env python3
"""
auto_analysis.py
Fetch 5y daily data from Yahoo (yfinance), compute VWMA/VWAP/Bollinger/RSI/MACD,
score Next Day / 2 Weeks / 3 Months / 6 Months, recommend option strikes,
save CSVs, PNG charts, and a JSON summary.
Edit STOCK_TICKERS and BENCHMARK_INDEX to your desired symbols.
"""

import os
import sys
import json
import logging
from datetime import datetime
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import numpy_financial as npf  # Smaller than scipy for financial calcs
from math import erf, sqrt  # Use math module instead of scipy.stats
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- User config ----------
STOCK_TICKERS = ["9988.HK", "0005.HK"]   # modify as needed, up to 3
BENCHMARK_INDEX = "^HSI"                 # modify as needed
OUT_DIR = "static/output"
TIMEFRAMES = {"next_day":1, "2_weeks":10, "3_months":65, "6_months":130}
STRIKE_STEP = 0.01     # two decimal places
RISK_FREE_RATE = 0.02
DIVIDEND_YIELD = 0.0

DEFAULT_WEIGHTS = {
 'next_day': {'rsi':0.12,'macd':0.08,'vwma':0.25,'bb':0.28,'vol':0.12,'vwap':0.15,'rsr':0.05},
 '2_weeks': {'rsi':0.18,'macd':0.22,'vwma':0.30,'bb':0.10,'vol':0.12,'vwap':0.05,'rsr':0.08},
 '3_months':{'rsi':0.12,'macd':0.22,'vwma':0.30,'bb':0.08,'vol':0.10,'vwap':0.04,'rsr':0.14},
 '6_months':{'rsi':0.08,'macd':0.18,'vwma':0.32,'bb':0.06,'vol':0.08,'vwap':0.03,'rsr':0.15},
}

# ---------- Utilities ----------
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)

def fetch_5y(ticker):
    logging.info(f"Fetching 5y data for {ticker}")
    df = yf.Ticker(ticker).history(period="5y", interval="1d", auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    df = df.reset_index()[['Date','Open','High','Low','Close','Volume','Adj Close']]
    return df

# Minimal technical analysis implementations
def rsi(close, window=14):
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gains, np.ones(window)/window, mode='valid')
    avg_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
    rs = avg_gain / np.maximum(avg_loss, 1e-9)  # Avoid div by zero
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    exp1 = np.exp(-np.arange(fast)[::-1]/fast)
    exp2 = np.exp(-np.arange(slow)[::-1]/slow)
    exp3 = np.exp(-np.arange(signal)[::-1]/signal)
    # Normalize weights
    exp1 = exp1 / exp1.sum()
    exp2 = exp2 / exp2.sum()
    exp3 = exp3 / exp3.sum()
    # Calculate EMAs
    ema12 = np.convolve(close, exp1, mode='valid')
    ema26 = np.convolve(close, exp2, mode='valid')
    # Calculate MACD line
    macd_line = ema12[-len(ema26):] - ema26
    # Calculate signal line
    signal_line = np.convolve(macd_line, exp3, mode='valid')
    return macd_line[-len(signal_line):], signal_line

def vwma(price, vol, window):
    # Use numpy's rolling window view for better memory efficiency
    def rolling_sum(x, w):
        if len(x) < w:
            return np.array([])
        shape = (x.shape[:-1] + (x.shape[-1] - w + 1, w))
        strides = x.strides + (x.strides[-1],)
        return np.sum(np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides), -1)
    
    pv = price * vol
    pv_sum = rolling_sum(pv, window)
    v_sum = rolling_sum(vol, window)
    v_sum = np.where(v_sum == 0, np.nan, v_sum)  # Avoid div by zero
    return pv_sum / v_sum

def compute_indicators(df):
    # Convert DataFrame to numpy arrays for efficient computation
    dates = df['Date'].values
    close = df['Close'].values
    volume = df['Volume'].values
    high = df['High'].values
    low = df['Low'].values
    
    # Calculate indicators using numpy operations
    sma200 = np.convolve(close, np.ones(200)/200, mode='valid')
    sma200 = np.pad(sma200, (200-1, 0), mode='edge')  # Pad start
    
    vwma20 = vwma(close, volume, 20)
    vwma20 = np.pad(vwma20, (20-1, 0), mode='edge')
    
    vwma50 = vwma(close, volume, 50)
    vwma50 = np.pad(vwma50, (50-1, 0), mode='edge')
    
    # Bollinger Bands
    bb_window = 20
    bb_std = np.zeros_like(close)
    bb_mid = np.zeros_like(close)
    for i in range(bb_window-1, len(close)):
        window = close[i-bb_window+1:i+1]
        bb_mid[i] = window.mean()
        bb_std[i] = window.std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # RSI and MACD
    rsi_vals = rsi(close)
    rsi_vals = np.pad(rsi_vals, (14, 0), mode='edge')
    
    macd_line, signal_line = macd(close)
    macd_hist = macd_line - signal_line
    pad_size = len(close) - len(macd_line)
    macd_line = np.pad(macd_line, (pad_size, 0), mode='edge')
    signal_line = np.pad(signal_line, (pad_size, 0), mode='edge')
    macd_hist = np.pad(macd_hist, (pad_size, 0), mode='edge')
    
    # Volume
    vol20 = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol20 = np.pad(vol20, (20-1, 0), mode='edge')
    
    # VWAP
    tp = (high + low + close) / 3.0
    vwap20 = vwma(tp, volume, 20)
    vwap20 = np.pad(vwap20, (20-1, 0), mode='edge')
    
    # Create polars DataFrame
    return pl.DataFrame({
        'Date': dates,
        'Close': close,
        'Volume': volume,
        'SMA200': sma200,
        'VWMA20': vwma20,
        'VWMA50': vwma50,
        'BB_mid': bb_mid,
        'BB_upper': bb_upper,
        'BB_lower': bb_lower,
        'RSI': rsi_vals,
        'MACD': macd_line,
        'MACD_signal': signal_line,
        'MACD_hist': macd_hist,
        'Vol20': vol20,
        'VWAP20': vwap20
    })

# Stats utilities
def norm_cdf(x):
    # Approximate normal CDF using error function
    return 0.5 * (1 + erf(x / sqrt(2)))

def norm_ppf(p):
    # Approximate inverse normal CDF (percent point function)
    # Using Acklam's algorithm
    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00
    
    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01
    
    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00
    
    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00
    
    p_low = 0.02425
    p_high = 1 - p_low
    
    if p_low <= p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1)
    else:
        if p < p_low:
            q = sqrt(-2 * np.log(p))
            x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
        else:
            q = sqrt(-2 * np.log(1 - p))
            x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    return x

def compute_rsr(stock_df, index_df):
    # Join using polars
    stock_df = pl.DataFrame({
        'Date': stock_df['Date'],
        'Close_stk': stock_df['Close']
    })
    index_df = pl.DataFrame({
        'Date': index_df['Date'],
        'Close_idx': index_df['Close']
    })
    merged = stock_df.join(index_df, on='Date', how='left')
    rsr = merged.with_columns([
        (pl.col('Close_stk') / pl.col('Close_idx')).alias('RSR')
    ])
    return rsr.select(['Date', 'RSR']).fill_null(strategy='forward')

# ---------- Scoring ----------
def score_window_weighted(df, rsr_df, days, weights):
    window = df.tail(days).copy()
    if window.empty:
        return 0.0, {}
    latest = window.iloc[-1]
    comps = {}
    # RSI
    if latest['RSI'] <= 30:
        comps['rsi'] = 1.0
    elif latest['RSI'] >= 70:
        comps['rsi'] = -1.0
    else:
        comps['rsi'] = (50 - latest['RSI']) / 50.0
    # MACD cross detection in last 3 bars
    macd_diff = window['MACD'] - window['MACD_signal']
    cross = np.sign(macd_diff).diff().iloc[-3:]
    if (cross > 0).any():
        comps['macd'] = 1.0
    elif (cross < 0).any():
        comps['macd'] = -1.0
    else:
        v = macd_diff.iloc[-1]
        comps['macd'] = v / (abs(v) + 1e-6)
    # VWMA positions
    vwma50_pos = 1.0 if latest['Close'] > latest['VWMA50'] else -1.0
    vwma20_pos = 1.0 if latest['Close'] > latest['VWMA20'] else -1.0
    comps['vwma'] = 0.6 * vwma50_pos + 0.4 * vwma20_pos
    # Bollinger proximity
    if latest['Close'] > latest['BB_upper']:
        comps['bb'] = 1.0
    elif latest['Close'] < latest['BB_lower']:
        comps['bb'] = -1.0
    else:
        d = (latest['Close'] - latest['BB_mid']) / (latest['BB_std'] + 1e-6)
        comps['bb'] = np.tanh(d / 2.0)
    # Volume
    comps['vol'] = 1.0 if latest['Volume'] > latest['Vol20'] else 0.6
    # VWAP proximity
    if not np.isnan(latest.get('VWAP20', np.nan)):
        prox = (latest['Close'] - latest['VWAP20']) / (latest['VWAP20'] + 1e-9)
        comps['vwap'] = np.tanh(prox * 10.0)
    else:
        comps['vwap'] = 0.0
    # RSR slope
    rsr_window = rsr_df[rsr_df['Date'].isin(window['Date'])]['RSR'].dropna()
    if len(rsr_window) >= 3:
        slope = np.polyfit(range(len(rsr_window)), rsr_window.values, 1)[0]
        comps['rsr'] = np.tanh(slope * 100.0)
    else:
        comps['rsr'] = 0.0
    # Aggregate
    total = 0.0
    wsum = 0.0
    for k, w in weights.items():
        total += w * comps.get(k, 0.0)
        wsum += abs(w)
    composite = 0.0 if wsum == 0 else total / wsum
    composite = float(np.clip(composite, -1.0, 1.0))
    return composite, comps

# ---------- Black-Scholes helpers for strike inversion ----------
def annualized_hist_vol(close_series):
    logret = np.log(close_series / close_series.shift(1)).dropna()
    return float(logret.std() * np.sqrt(252)) if not logret.empty else 0.25

def bs_d1(S, K, r, q, sigma, T):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def call_delta_bs(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, r, q, sigma, T)
    return np.exp(-q * T) * norm.cdf(d1)

def put_delta_bs(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = bs_d1(S, K, r, q, sigma, T)
    return np.exp(-q * T) * (norm.cdf(d1) - 1)

def strike_for_target_delta(S, target_delta, is_call, r, q, sigma, T):
    def f(K):
        return (call_delta_bs(S, K, r, q, sigma, T) - target_delta) if is_call else (put_delta_bs(S, K, r, q, sigma, T) - target_delta)
    K_low = max(0.01, S * 0.2)
    K_high = S * 3.0
    try:
        K = brentq(f, K_low, K_high)
        return float(K)
    except Exception:
        return round(S, 2)

def recommend_strikes(S, close_series, composite_score, timeframe_days, strike_step=STRIKE_STEP, implied_vol=None, r=RISK_FREE_RATE, q=DIVIDEND_YIELD):
    T = max(1, timeframe_days) / 365.0
    sigma = implied_vol if implied_vol is not None else annualized_hist_vol(close_series)
    delta_map = {1:((0.45,0.55),(0.15,0.25)), 10:((0.48,0.60),(0.18,0.30)), 65:((0.50,0.65),(0.15,0.28)), 130:((0.55,0.70),(0.12,0.25))}
    days_key = min(delta_map.keys(), key=lambda k: abs(k - timeframe_days))
    buyer_range, seller_range = delta_map[days_key]
    if composite_score >= 0.6:
        side='buy'; target_delta=np.mean(buyer_range); is_call=True
    elif composite_score <= -0.6:
        side='buy'; target_delta=-np.mean(buyer_range); is_call=False
    elif composite_score > 0.2:
        side='buy_spread'; target_delta=np.mean(buyer_range); is_call=True
    elif composite_score < -0.2:
        side='buy_spread'; target_delta=-np.mean(buyer_range); is_call=False
    else:
        side='sell_credit_or_neutral'; target_delta=np.mean(seller_range); is_call=True if composite_score>0 else False
    abs_.target = abs(target_delta)
    K_target = strike_for_target_delta(S, abs_target, is_call if target_delta>0 else not is_call, r, q, sigma, T)
    K_rounded = float(np.round(np.round(K_target / strike_step) * strike_step, 2))
    K_atm = float(np.round(np.round(S / strike_step) * strike_step, 2))
    K_conservative = float(np.round(np.round(((K_rounded + K_atm) / 2) / strike_step) * strike_step, 2))
    return {'side':side,'is_call':is_call,'target_delta':float(np.sign(target_delta)*abs_target),'strike_target':K_rounded,'strike_atm':K_atm,'strike_conservative':K_conservative,'sigma_used':float(sigma),'T_years':float(T)}

# ---------- Plotting ----------
def plot_analysis(df, summary, ticker):
    df = df.sort_values('Date').reset_index(drop=True)
    window = df.tail(130).copy()
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=4, cols=1, 
                       row_heights=[0.4, 0.2, 0.2, 0.2],
                       shared_xaxes=True,
                       vertical_spacing=0.02)
    
    # Price and indicators
    fig.add_trace(go.Scatter(x=window['Date'], y=window['Close'], name='Close', line=dict(color='white')), row=1, col=1)
    fig.add_trace(go.Scatter(x=window['Date'], y=window['VWMA50'], name='VWMA50', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=window['Date'], y=window['VWMA20'], name='VWMA20', line=dict(color='cyan')), row=1, col=1)
    fig.add_trace(go.Scatter(x=window['Date'], y=window['BB_upper'], name='BB', line=dict(color='gray'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=window['Date'], y=window['BB_lower'], fill='tonexty', fillcolor='rgba(128,128,128,0.1)', 
                            line=dict(color='gray'), showlegend=False), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=window['Date'], y=window['RSI'], name='RSI(14)', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=window['Date'], y=window['MACD'], name='MACD', line=dict(color='green')), row=3, col=1)
    fig.add_trace(go.Scatter(x=window['Date'], y=window['MACD_signal'], name='Signal', line=dict(color='red')), row=3, col=1)
    fig.add_trace(go.Bar(x=window['Date'], y=window['MACD_hist'], name='MACD hist', marker_color='gray'), row=3, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=window['Date'], y=window['Volume'], name='Volume', marker_color='rgba(0,0,255,0.6)'), row=4, col=1)
    fig.add_trace(go.Scatter(x=window['Date'], y=window['Vol20'], name='Vol20', line=dict(color='red')), row=4, col=1)
    
    # Add scores annotation
    ann = '<br>'.join([f"{k}: {v['score']:.2f}" for k,v in summary.items()])
    fig.add_annotation(text=ann, xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
                      font=dict(size=10), bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1)
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Price with VWMA and Bollinger Bands",
        template="plotly_dark",
        showlegend=True,
        height=800,
        margin=dict(t=30, l=10, r=10, b=10)
    )
    
    # Save as HTML (interactive) instead of PNG
    outfn = os.path.join(OUT_DIR, f"{ticker.replace('^','_')}_analysis.html")
    fig.write_html(outfn)
    return outfn

# ---------- Orchestration ----------
def analyze_stocks(stock_tickers, benchmark_index):
    ensure_dirs()
    # Fetch data
    all_dfs = {}
    idx_df = fetch_5y(benchmark_index)
    idx_df = compute_indicators(idx_df)
    all_dfs[benchmark_index] = idx_df
    for t in stock_tickers:
        if len(all_dfs) >= 4:
            break
        df = fetch_5y(t)
        df = compute_indicators(df)
        all_dfs[t] = df
    # Analyze each stock vs index independently
    summary = {}
    for t in stock_tickers:
        logging.info(f"Analyzing {t}")
        stk = all_dfs[t]
        rsr = compute_rsr(stk, idx_df)
        stock_summary = {}
        for tf, days in TIMEFRAMES.items():
            score, comps = score_window_weighted(stk, rsr, days, DEFAULT_WEIGHTS[tf])
            strikes = recommend_strikes(float(stk['Close'].iloc[-1]), stk['Close'], score, days)
            stock_summary[tf] = {'score': score, 'components': comps, 'strike_recommendation': strikes}
        plotfile = plot_analysis(stk, {k: {'score': v['score']} for k, v in stock_summary.items()}, t)
        stock_summary['plot'] = plotfile
        summary[t] = stock_summary
    return summary

# ---------- CLI ----------
def main():
    ensure_dirs()
    tickers = STOCK_TICKERS
    index = BENCHMARK_INDEX
    if len(sys.argv) > 1:
        tickers = sys.argv[1].split(",")
    if len(sys.argv) > 2:
        index = sys.argv[2]
    summary = analyze_stocks(tickers, index)
    print(json.dumps(summary, indent=2, default=float))

if __name__ == "__main__":
    main()
