#!/usr/bin/env python3
"""
Optimized version of auto_analysis.py with minimal dependencies and efficient implementations
"""

import os
import sys
import json
import logging
from datetime import datetime
from math import erf, sqrt, exp, log
import numpy as np
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import asyncio
import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- Config ----------
STOCK_TICKERS = ["9988.HK", "0005.HK"]
BENCHMARK_INDEX = "^HSI"
OUT_DIR = "static/output"
TIMEFRAMES = {"next_day":1, "2_weeks":10, "3_months":65, "6_months":130}
STRIKE_STEP = 0.01
RISK_FREE_RATE = 0.02
DIVIDEND_YIELD = 0.0

DEFAULT_WEIGHTS = {
    'next_day': {'rsi':0.12,'macd':0.08,'vwma':0.25,'bb':0.28,'vol':0.12,'vwap':0.15,'rsr':0.05},
    '2_weeks': {'rsi':0.18,'macd':0.22,'vwma':0.30,'bb':0.10,'vol':0.12,'vwap':0.05,'rsr':0.08},
    '3_months':{'rsi':0.12,'macd':0.22,'vwma':0.30,'bb':0.08,'vol':0.10,'vwap':0.04,'rsr':0.14},
    '6_months':{'rsi':0.08,'macd':0.18,'vwma':0.32,'bb':0.06,'vol':0.08,'vwap':0.03,'rsr':0.15},
}

# ---------- Stats Utils ----------
def norm_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))

def norm_ppf(p):
    # Acklam's approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    if 0.02425 <= p <= 0.97575:
        q = p - 0.5
        r = q * q
        num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
        den = ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1
        return num/den
    
    q = sqrt(-2*log(p if p < 0.02425 else 1-p))
    num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
    den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    return -num/den if p > 0.97575 else num/den

# ---------- Technical Analysis ----------
def rsi(close, window=14):
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Use convolve for rolling average (faster than rolling window)
    avg_gain = np.convolve(gains, np.ones(window)/window, mode='valid')
    avg_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
    
    rs = avg_gain / np.maximum(avg_loss, 1e-9)  # Avoid div by zero
    rsi_vals = 100 - (100 / (1 + rs))
    return np.pad(rsi_vals, (window, 0), mode='edge')

def macd(close, fast=12, slow=26, signal=9):
    # Exponential weights
    fast_w = 2.0 / (fast + 1)
    slow_w = 2.0 / (slow + 1)
    signal_w = 2.0 / (signal + 1)
    
    # Calculate EMAs
    fast_ema = np.zeros_like(close)
    slow_ema = np.zeros_like(close)
    fast_ema[0] = slow_ema[0] = close[0]
    
    for i in range(1, len(close)):
        fast_ema[i] = close[i] * fast_w + fast_ema[i-1] * (1 - fast_w)
        slow_ema[i] = close[i] * slow_w + slow_ema[i-1] * (1 - slow_w)
    
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = np.zeros_like(macd_line)
    signal_line[0] = macd_line[0]
    for i in range(1, len(macd_line)):
        signal_line[i] = macd_line[i] * signal_w + signal_line[i-1] * (1 - signal_w)
    
    return macd_line, signal_line

def vwma(price, volume, window):
    # Efficient rolling sum using stride tricks
    def rolling_sum(x, w):
        shape = (x.shape[0] - w + 1, w)
        strides = (x.strides[0], x.strides[0])
        return np.sum(np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides), axis=1)
    
    pv = price * volume
    if len(price) < window:
        return np.array([])
    
    pv_sum = rolling_sum(pv, window)
    v_sum = rolling_sum(volume, window)
    v_sum = np.where(v_sum == 0, np.nan, v_sum)
    result = pv_sum / v_sum
    return np.pad(result, (window-1, 0), mode='edge')

# ---------- Options ----------
def call_delta(S, K, r, q, sigma, T):
    d1 = (log(S/K) + (r - q + sigma**2/2) * T) / (sigma * sqrt(T))
    return exp(-q * T) * norm_cdf(d1)

def put_delta(S, K, r, q, sigma, T):
    d1 = (log(S/K) + (r - q + sigma**2/2) * T) / (sigma * sqrt(T))
    return exp(-q * T) * (norm_cdf(d1) - 1)

def strike_for_target_delta(S, target_delta, is_call, r, q, sigma, T):
    def f(K):
        if is_call:
            return call_delta(S, K, r, q, sigma, T) - target_delta
        else:
            return put_delta(S, K, r, q, sigma, T) - target_delta
    
    # Binary search
    K_low = S * 0.5
    K_high = S * 1.5
    tol = 1e-5
    max_iter = 50
    
    for _ in range(max_iter):
        K_mid = (K_low + K_high) / 2
        f_mid = f(K_mid)
        
        if abs(f_mid) < tol:
            return K_mid
        elif f_mid > 0:
            K_low = K_mid
        else:
            K_high = K_mid
    
    return (K_low + K_high) / 2

def recommend_strikes(S, hist_prices, signal_score, days_forward):
    # Calculate historical volatility
    returns = np.diff(np.log(hist_prices[-min(len(hist_prices), 252):]))
    sigma = np.std(returns) * np.sqrt(252)
    T = days_forward / 252
    r = RISK_FREE_RATE
    q = DIVIDEND_YIELD
    
    # Map score to target delta
    target_delta = min(max(abs(signal_score), 0.2), 0.4)
    side = 'buy' if signal_score > 0 else 'sell'
    is_call = signal_score > 0
    
    # Find strikes
    K_target = strike_for_target_delta(S, target_delta, is_call, r, q, sigma, T)
    K_atm = S
    K_conservative = S * (1.1 if is_call else 0.9)
    
    return {
        'side': side,
        'is_call': is_call,
        'target_delta': target_delta,
        'strike_target': round(K_target/STRIKE_STEP)*STRIKE_STEP,
        'strike_atm': round(K_atm/STRIKE_STEP)*STRIKE_STEP,
        'strike_conservative': round(K_conservative/STRIKE_STEP)*STRIKE_STEP,
        'sigma_used': float(sigma),
        'days': days_forward
    }

# ---------- Data Processing ----------
async def fetch_stock_data(session, ticker):
    """Async wrapper for yfinance"""
    df = await asyncio.get_event_loop().run_in_executor(
        None, lambda: yf.Ticker(ticker).history(period="5y", interval="1d", auto_adjust=False)
    )
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    return df.reset_index()[['Date','Open','High','Low','Close','Volume','Adj Close']]

def compute_indicators(df):
    # Convert to numpy arrays for efficient computation
    dates = df['Date'].values
    close = df['Close'].values
    volume = df['Volume'].values
    high = df['High'].values
    low = df['Low'].values
    
    # Calculate indicators
    vwma20 = vwma(close, volume, 20)
    vwma50 = vwma(close, volume, 50)
    
    # Bollinger Bands (20,2)
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
    macd_line, signal_line = macd(close)
    macd_hist = macd_line - signal_line
    
    # Volume and VWAP
    vol20 = vwma(np.ones_like(volume), volume, 20)  # Simple moving average
    tp = (high + low + close) / 3.0
    vwap20 = vwma(tp, volume, 20)
    
    # Create polars DataFrame
    return pl.DataFrame({
        'Date': dates,
        'Close': close,
        'Volume': volume,
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

def compute_component_score(df, rsr_df, metric):
    """
    Calculate component scores for technical indicators
    Returns normalized score between -1 and 1
    """
    if df.is_empty():
        return 0.0
        
    latest = df.tail(1)
    
    if metric == 'rsi':
        rsi_val = float(latest['RSI'].iloc[0])
        if rsi_val <= 30:
            return 1.0
        elif rsi_val >= 70:
            return -1.0
        else:
            # Normalize between 30-70 to [-0.5, 0.5]
            return -((rsi_val - 50) / 40)
            
    elif metric == 'macd':
        macd = float(latest['MACD'].iloc[0])
        signal = float(latest['MACD_signal'].iloc[0])
        hist = macd - signal
        # Normalize histogram by recent max
        max_hist = abs(df['MACD_hist']).max()
        return float(np.clip(hist / max_hist if max_hist > 0 else 0, -1, 1))
        
    elif metric in ['vwma', 'vwap']:
        price = float(latest['Close'].iloc[0])
        indicator = float(latest[f'{metric.upper()}20'].iloc[0])
        # Score based on price position relative to indicator
        return float(np.clip((price/indicator - 1) * 5, -1, 1))
        
    elif metric == 'bb':
        price = float(latest['Close'].iloc[0])
        upper = float(latest['BB_upper'].iloc[0])
        lower = float(latest['BB_lower'].iloc[0])
        mid = (upper + lower) / 2
        band_width = upper - lower
        if band_width == 0:
            return 0
        # Normalize position within bands to [-1, 1]
        return float(np.clip(2 * (price - mid) / band_width, -1, 1))
        
    elif metric == 'vol':
        curr_vol = float(latest['Volume'].iloc[0])
        avg_vol = float(latest['Vol20'].iloc[0])
        if avg_vol == 0:
            return 0
        # Score volume ratio, log scale
        ratio = np.log(curr_vol / avg_vol) if curr_vol > 0 else 0
        return float(np.clip(ratio, -1, 1))
        
    elif metric == 'rsr':
        rsr_val = float(rsr_df.tail(1)['RSR'].iloc[0])
        rsr_avg = float(rsr_df['RSR'].mean())
        if rsr_avg == 0:
            return 0
        # Score relative strength ratio
        ratio = np.log(rsr_val / rsr_avg) if rsr_val > 0 else 0
        return float(np.clip(ratio * 2, -1, 1))
        
    return 0.0

# ---------- Plotting ----------
def plot_analysis(df, summary, ticker):
    df = df.sort_values('Date').reset_index(drop=True)
    window = df.tail(130).copy()
    
    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        shared_xaxes=True,
        vertical_spacing=0.02
    )
    
    # Price and indicators
    fig.add_trace(go.Scatter(
        x=window['Date'], y=window['Close'],
        name='Close', line=dict(color='white')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=window['Date'], y=window['VWMA50'],
        name='VWMA50', line=dict(color='blue')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=window['Date'], y=window['VWMA20'],
        name='VWMA20', line=dict(color='cyan')
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=window['Date'], y=window['BB_upper'],
        name='BB', line=dict(color='gray'),
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=window['Date'], y=window['BB_lower'],
        fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
        line=dict(color='gray'),
        showlegend=False
    ), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=window['Date'], y=window['RSI'],
        name='RSI(14)', line=dict(color='purple')
    ), row=2, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(
        x=window['Date'], y=window['MACD'],
        name='MACD', line=dict(color='green')
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=window['Date'], y=window['MACD_signal'],
        name='Signal', line=dict(color='red')
    ), row=3, col=1)
    
    fig.add_trace(go.Bar(
        x=window['Date'], y=window['MACD_hist'],
        name='MACD hist', marker_color='gray'
    ), row=3, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=window['Date'], y=window['Volume'],
        name='Volume', marker_color='rgba(0,0,255,0.6)'
    ), row=4, col=1)
    
    fig.add_trace(go.Scatter(
        x=window['Date'], y=window['Vol20'],
        name='Vol20', line=dict(color='red')
    ), row=4, col=1)
    
    # Add scores annotation
    ann = '<br>'.join([f"{k}: {v['score']:.2f}" for k,v in summary.items()])
    fig.add_annotation(
        text=ann,
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        showarrow=False,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Analysis",
        template="plotly_dark",
        showlegend=True,
        height=800,
        margin=dict(t=30, l=10, r=10, b=10)
    )
    
    # Save as HTML with external plotly.js
    config = {
        'plotlyServerURL': 'https://cdn.plot.ly',
        'responsive': True
    }
    
    outfn = os.path.join(OUT_DIR, f"{ticker.replace('^','_')}_analysis.html")
    fig.write_html(
        outfn,
        config=config,
        include_plotlyjs='cdn',  # Use CDN
        full_html=False  # Minimal HTML wrapper
    )
    return outfn

# ---------- Orchestration ----------
async def analyze_stocks(stock_tickers, benchmark_index):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Fetch data concurrently
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_stock_data(session, ticker)
            for ticker in [benchmark_index] + stock_tickers
        ]
        all_dfs = await asyncio.gather(*tasks)
    
    # Process data
    idx_df = compute_indicators(all_dfs[0])
    summary = {}
    
    for i, ticker in enumerate(stock_tickers, 1):
        logging.info(f"Analyzing {ticker}")
        stock_df = compute_indicators(all_dfs[i])
        rsr = compute_rsr(stock_df, idx_df)
        
        stock_summary = {}
        for tf, days in TIMEFRAMES.items():
            window = stock_df.tail(days)
            if not window.is_empty():
                rsr_window = rsr.tail(days)
                weights = DEFAULT_WEIGHTS[tf]
                
                # Calculate weighted score
                score = sum(
                    weight * component
                    for metric, weight in weights.items()
                    for component in [compute_component_score(window, rsr_window, metric)]
                )
                
                strikes = recommend_strikes(
                    float(stock_df['Close'].iloc[-1]),
                    stock_df['Close'].values,
                    score,
                    days
                )
                
                stock_summary[tf] = {
                    'score': score,
                    'strike_recommendation': strikes
                }
        
        # Generate plot
        plotfile = plot_analysis(
            stock_df,
            {k: {'score': v['score']} for k, v in stock_summary.items()},
            ticker
        )
        stock_summary['plot'] = plotfile
        summary[ticker] = stock_summary
    
    return summary

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tickers = STOCK_TICKERS
    index = BENCHMARK_INDEX
    
    if len(sys.argv) > 1:
        tickers = sys.argv[1].split(",")
    if len(sys.argv) > 2:
        index = sys.argv[2]
    
    summary = asyncio.run(analyze_stocks(tickers, index))
    print(json.dumps(summary, indent=2, default=float))

if __name__ == "__main__":
    main()