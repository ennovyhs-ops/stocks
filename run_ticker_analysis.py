# save as run_ticker_analysis.py and run: python run_ticker_analysis.py
# pip install yfinance pandas numpy matplotlib seaborn ta scipy

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import brentq
import json
import os
import ta

sns.set(style="darkgrid")
TICKERS = ["9988.HK","0005.HK","1810.HK","^HSI"]
TIMEFRAMES = {"next_day":1,"2_weeks":10,"3_months":65,"6_months":130}
DEFAULT_WEIGHTS = {
 'next_day': {'rsi':0.12,'macd':0.08,'vwma':0.25,'bb':0.28,'vol':0.12,'vwap':0.15,'rsr':0.05},
 '2_weeks': {'rsi':0.18,'macd':0.22,'vwma':0.30,'bb':0.10,'vol':0.12,'vwap':0.05,'rsr':0.08},
 '3_months':{'rsi':0.12,'macd':0.22,'vwma':0.30,'bb':0.08,'vol':0.10,'vwap':0.04,'rsr':0.14},
 '6_months':{'rsi':0.08,'macd':0.18,'vwma':0.32,'bb':0.06,'vol':0.08,'vwap':0.03,'rsr':0.15},
}

# ---------- Helpers ----------
def fetch_and_save(ticker, period="5y"):
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    df = df.reset_index().rename(columns={"Open":"Open","High":"High","Low":"Low","Close":"Close","Volume":"Volume","Adj Close":"Adj Close"})
    df = df[['Date','Open','High','Low','Close','Adj Close','Volume']].dropna().sort_values('Date').reset_index(drop=True)
    fn = f"{ticker.replace('^','_')}_5y.csv"
    df.to_csv(fn, index=False)
    return df

def vwma(price, vol, window):
    pv = (price * vol).rolling(window=window, min_periods=1).sum()
    v = vol.rolling(window=window, min_periods=1).sum().replace(0, np.nan)
    return pv / v

def compute_indicators(df):
    out = df.copy()
    out['SMA200'] = out['Close'].rolling(window=200, min_periods=1).mean()
    out['VWMA20'] = vwma(out['Close'], out['Volume'], 20)
    out['VWMA50'] = vwma(out['Close'], out['Volume'], 50)
    out['BB_mid'] = out['Close'].rolling(window=20, min_periods=1).mean()
    out['BB_std'] = out['Close'].rolling(window=20, min_periods=1).std()
    out['BB_upper'] = out['BB_mid'] + 2 * out['BB_std']
    out['BB_lower'] = out['BB_mid'] - 2 * out['BB_std']
    out['RSI'] = ta.momentum.RSIIndicator(out['Close'], window=14).rsi()
    macd = ta.trend.MACD(out['Close'], window_slow=26, window_fast=12, window_sign=9)
    out['MACD'] = macd.macd()
    out['MACD_signal'] = macd.macd_signal()
    out['MACD_hist'] = out['MACD'] - out['MACD_signal']
    out['Vol20'] = out['Volume'].rolling(window=20, min_periods=1).mean()
    tp = (out['High'] + out['Low'] + out['Close']) / 3.0
    pv = (tp * out['Volume']).rolling(window=20, min_periods=1).sum()
    v = out['Volume'].rolling(window=20, min_periods=1).sum().replace(0, np.nan)
    out['VWAP20'] = pv / v
    return out

def compute_rsr(stock_df, index_df):
    merged = pd.merge(stock_df[['Date','Close']], index_df[['Date','Close']], on='Date', how='left', suffixes=('_stk','_idx'))
    merged['RSR'] = merged['Close_stk'] / merged['Close_idx']
    merged['RSR'] = merged['RSR'].fillna(method='ffill')
    return merged[['Date','RSR']]

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
    K_low = S * 0.2
    K_high = S * 3.0
    try:
        K = brentq(f, K_low, K_high)
        return float(K)
    except Exception:
        return round(S, 2)

def recommend_strikes(S, close_series, composite_score, timeframe_days, strike_step=0.1, implied_vol=None, r=0.02, q=0.0):
    T = max(1, timeframe_days) / 365.0
    sigma = implied_vol if implied_vol is not None else annualized_hist_vol(close_series)
    delta_map = {1:((0.45,0.55),(0.15,0.25)),10:((0.48,0.60),(0.18,0.30)),65:((0.50,0.65),(0.15,0.28)),130:((0.55,0.70),(0.12,0.25))}
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
    abs_target = abs(target_delta)
    K_target = strike_for_target_delta(S, abs_target, is_call if target_delta>0 else not is_call, r, q, sigma, T)
    K_rounded = float(np.round(K_target / strike_step) * strike_step)
    K_atm = float(np.round(S / strike_step) * strike_step)
    K_conservative = float(np.round(((K_rounded + K_atm) / 2) / strike_step) * strike_step)
    return {'side':side,'is_call':is_call,'target_delta':float(np.sign(target_delta)*abs_target),'strike_target':K_rounded,'strike_atm':K_atm,'strike_conservative':K_conservative,'sigma_used':float(sigma),'T_years':float(T)}

def score_window_weighted(df, rsr_df, days, weights):
    window = df.tail(days).copy()
    if window.empty:
        return 0.0, {}
    latest = window.iloc[-1]
    comps = {}
    # RSI
    if latest['RSI'] <= 30: comps['rsi']=1.0
    elif latest['RSI'] >= 70: comps['rsi']=-1.0
    else: comps['rsi']=(50-latest['RSI'])/50.0
    # MACD cross detection recent 3 bars
    macd_diff = window['MACD'] - window['MACD_signal']
    cross = np.sign(macd_diff).diff().iloc[-3:]
    if (cross>0).any(): comps['macd']=1.0
    elif (cross<0).any(): comps['macd']=-1.0
    else: comps['macd']=macd_diff.iloc[-1]/(abs(macd_diff.iloc[-1])+1e-6)
    # VWMA
    vwma50_pos = 1.0 if latest['Close'] > latest['VWMA50'] else -1.0
    vwma20_pos = 1.0 if latest['Close'] > latest['VWMA20'] else -1.0
    comps['vwma'] = 0.6*vwma50_pos + 0.4*vwma20_pos
    # BB
    if latest['Close'] > latest['BB_upper']: comps['bb']=1.0
    elif latest['Close'] < latest['BB_lower']: comps['bb']=-1.0
    else:
        d = (latest['Close'] - latest['BB_mid']) / (latest['BB_std'] + 1e-6)
        comps['bb'] = np.tanh(d/2.0)
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
    # Weighted aggregate
    total = 0.0; wsum = 0.0
    for k,w in weights.items():
        total += w * comps.get(k,0.0); wsum += abs(w)
    composite = 0.0 if wsum==0 else total/wsum
    composite = float(np.clip(composite, -1.0, 1.0))
    return composite, comps

def plot_and_save(df, summary, ticker):
    window = df.tail(130).copy()
    fig, axes = plt.subplots(4,1, figsize=(14,12), sharex=True, gridspec_kw={'height_ratios':[3,1,1,0.8]})
    ax_price, ax_rsi, ax_macd, ax_vol = axes
    ax_price.plot(window['Date'], window['Close'], label='Close', color='black')
    ax_price.plot(window['Date'], window['VWMA50'], label='VWMA50', color='blue', linewidth=0.8)
    ax_price.plot(window['Date'], window['VWMA20'], label='VWMA20', color='cyan', linewidth=0.7)
    ax_price.fill_between(window['Date'], window['BB_lower'], window['BB_upper'], color='gray', alpha=0.12)
    ax_price.set_title(f'{ticker} Price VWMA and Bollinger Bands')
    ax_rsi.plot(window['Date'], window['RSI'], label='RSI(14)', color='purple')
    ax_rsi.axhline(70, color='red', linestyle='--'); ax_rsi.axhline(30, color='green', linestyle='--')
    ax_macd.plot(window['Date'], window['MACD'], label='MACD', color='green')
    ax_macd.plot(window['Date'], window['MACD_signal'], label='Signal', color='red', linewidth=0.8)
    ax_macd.bar(window['Date'], window['MACD_hist'], label='MACD hist', color='gray', alpha=0.7)
    ax_vol.bar(window['Date'], window['Volume'], color='tab:blue', alpha=0.6)
    ax_vol.plot(window['Date'], window['Vol20'], color='red', linewidth=0.9)
    ann = '\n'.join([f"{k}: {v['score']:.2f}" for k,v in summary.items()])
    ax_price.text(0.01, 0.98, ann, transform=ax_price.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_price.legend(loc='upper left')
    plt.tight_layout()
    outfn = f"{ticker.replace('^','_')}_analysis.png"
    fig.savefig(outfn)
    plt.close(fig)
    return outfn

# ---------- Main ----------
def main():
    os.makedirs("data", exist_ok=True)
    dfs = {}
    for t in TICKERS:
        print(f"Fetching {t} ...")
        df = fetch_and_save(t)
        dfs[t]=df
    # compute indicators
    for t,df in dfs.items():
        dfs[t]=compute_indicators(df)
    idx = dfs["^HSI"]
    results = {}
    for stock in ["9988.HK","0005.HK"]:
        stk = dfs[stock]
        rsr = compute_rsr(stk, idx)
        stock_summary = {}
        for tf,days in TIMEFRAMES.items():
            score, comps = score_window_weighted(stk, rsr, days, DEFAULT_WEIGHTS[tf])
            strikes = recommend_strikes(float(stk['Close'].iloc[-1]), stk['Close'], score, days)
            stock_summary[tf] = {'score':score,'components':comps,'strike_recommendation':strikes}
        png = plot_and_save(stk, {k: {'score':v['score']} for k,v in stock_summary.items()}, stock)
        stock_summary['plot'] = png
        results[stock] = stock_summary
    # save summary
    with open("summary.json","w") as f:
        json.dump(results,f,indent=2,default=float)
    print(json.dumps(results, indent=2, default=float))
    print("CSV files and PNG plots saved in current directory.")

if __name__ == "__main__":
    main()
