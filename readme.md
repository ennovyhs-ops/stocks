## Daily Option Strategy Analyzer
This project is a compact web app that fetches 3 years of daily OHLCV data for up to 3 stocks and one benchmark index from Yahoo Finance, computes core technical indicators, scores multiple timeframes, and provides delta-based option strike recommendations. The app returns results on-demand without storing outputs.

## Key Features
- Fetches 3 years of daily data from Yahoo Finance.
- Computes VWMA(20,50), Bollinger Bands(20,2), RSI(14), MACD(12,26,9), VWAP(20), 20-day average volume.
- Produces composite scores for Next Day, 2 Weeks, 3 Months, 6 Months.
- Recommends option strikes (call/put, long/short, conservative ATM reference) using delta inversion.
- Provides a simple web interface to enter tickers and view results immediately.
- Does not store analysis results persistently.

## Overview
- Small Flask app that fetches 3 years of daily OHLCV via yfinance, computes indicators, scores directional bias across timeframes, and recommends option strikes using delta inversion and Black‑Scholes pricing.
- Configurable via the UI.


## Usage
Enter up to 3 stock tickers separated by commas and a benchmark index.

Click Analyze to fetch live data and compute results.

The page displays a summary of component scores and recommended option plays.


# Daily Option Strategy Analyzer
## Scoring System Overview
This repository runs a compact web app that fetches 3 years of daily OHLCV data from Yahoo Finance, computes core technical indicators, produces composite directional scores across multiple timeframes, and gives explicit option strike recommendations using delta-based Black‑Scholes inversion. The app returns results on demand via a simple web interface without persistent storage.

## Scoring System
Components
- RSI — 14‑day Relative Strength Index; signals overbought/oversold bias.
- MACD — MACD line, signal line, and histogram; captures momentum and recent cross events.
- VWMA — Volume‑weighted moving averages (50 and 20); measures trend with volume weighting.
- Bollinger Bands — 20‑day mid ± 2σ; measures volatility and proximity to band edges.
- Volume — 20‑day average volume vs latest volume; confirms momentum moves.
- VWAP — 20‑day VWAP; measures average price with volume weighting for mean reversion context.
- RSR — Relative Strength Ratio vs benchmark index (stock close / index close) and its short slope; captures relative strength or weakness.

## Scoring system
- Components: RSI(14), MACD(12,26,9), VWMA20/50, Bollinger 20±2σ, Volume vs Vol20, VWAP20, RSR vs index and slope.
- Per-component scoring and weights as implemented in app.py.
- Composite normalized to [-1,1].

## Per‑component Scoring Rules
- RSI: +1.0 if RSI ≤ 30; −1.0 if RSI ≥ 70; linear mapping to 0 at RSI = 50 for in‑between values.
- MACD: +1.0 for recent bullish MACD cross; −1.0 for bearish cross; otherwise signed normalization of MACD histogram.
- VWMA: +1.0 if Close > VWMA50 and Close > VWMA20; −1.0 if below; weighted blend of 50/20 signals.
- Bollinger: +1.0 if price above upper band; −1.0 if below lower band; otherwise tanh of standardized distance from midband.
- Volume: +1.0 if latest volume > Vol20; otherwise a conservatively positive score (e.g., 0.6) to avoid punishing stable low‑volume moves.
- VWAP: tanh of standardized distance of Close from VWAP20, clipped to (−1, 1).
- RSR slope: slope of recent RSR values scaled and passed through tanh to produce (−1, 1).

## Weights by Timeframe
- Next Day: higher weight on short trend and volatility signals, moderate RSI and VWAP.
    - Example weights: RSI 0.14, MACD 0.16, VWMA 0.26, BB 0.18, Vol 0.10, VWAP 0.06, RSR 0.10.
- 2 Weeks: stronger MACD and VWMA influence, slightly higher RSI weight.
- 3 Months: heavier VWMA and MACD, lower short‑term volatility weight.
- 6 Months: strongest trend weight (VWMA), higher RSR emphasis, lower volatility weight.

Weights are configurable in the script. The composite score is the weighted sum of component scores normalized by the sum of absolute weights and clipped to [−1.0, 1.0].

## Composite Interpretation and Action Mapping
- Composite ≥ 0.6: Clear bullish premise — suggest long call or long directional call structure.
- 0.2 < Composite < 0.6: Modest bullish — suggest buy call spread or directional spread with lower capital at risk.
- −0.2 ≤ Composite ≤ 0.2: Neutral — suggest neutral or income credit structures (sell credit spreads, iron condors) or no trade.
- −0.6 < Composite < −0.2: Modest bearish — suggest buy put spread.
- Composite ≤ −0.6: Clear bearish premise — suggest long put or long directional put structure.

If a recommendation depends on future price or premium reaching a threshold, the recommendation text explicitly earmarks the trigger condition and advises waiting.

## Indicator Calculations
- Use the latest 3 years of daily data per ticker and benchmark.
- VWMA computed as rolling sum(price * volume) / rolling sum(volume).
- VWAP20 computed from typical price times volume over 20 days divided by sum(volume).
- Bollinger mid and std use 20‑day window.
- RSI uses Wilder smoothing over 14 days (rolling average of gains/losses).
- MACD uses 12/26/9 EMA standard.
- Vol20 uses 20‑day rolling volume mean.
- RSR is stock Close / index Close aligned by Date then forward‑filled where necessary; slope measured over the chosen scoring window.

All computations are vectorized with pandas for speed.

## Strike Recommendation Logic
- Use delta targets mapped to timeframe buckets (examples): Next Day 0.50, 2 Weeks 0.55, 3 Months 0.60, 6 Months 0.65 for buyer-side; smaller absolute deltas for seller-side.
- Convert target delta to strike via numerical inversion of Black‑Scholes call/put delta using historical realized volatility when implied volatility is unavailable.
- Round strikes to two decimals using a configurable strike step (default 0.01).
- Recommendation fields include side (LONG_CALL, BUY_CALL_SPREAD, LONG_PUT, SHORT_CALL, SHORT_PUT, BUY_PUT_SPREAD, SELL_CALL_SPREAD, SELL_PUT_SPREAD etc.), is_call boolean, target_delta, strike_target, strike_atm, strike_conservative, sigma_used, T_years, and an explicit note such as “Trade now” or “Wait for premium / price trigger”.

## Recommendation policy
- Two trade policies: prefer_spread (convert naked longs to defined-risk spreads) and allow_naked (allow single-leg directional).
- Delta targets per timeframe are configurable in app.py.
- Strike conservative = midpoint(strike_target, ATM) rounded to strike_step.

## Output Format and UI Behavior
The web UI accepts up to 3 tickers and one benchmark index.

The page displays for each ticker:
- Latest Close price.
- Inline chart showing Close, VWMA20, VWMA50, Bollinger Bands and RSI subpanel.
- JSON summary of composite scores, per‑component contributions, and strike recommendations for each timeframe.

No persistent storage is used by default; results are computed on request and returned immediately.

## Configuration
Edit the top of the app script to change:
- STOCK_TICKERS default list, BENCHMARK_INDEX, STRIKE_STEP, RISK_FREE_RATE, DIVIDEND_YIELD, and timeframe weights.

Adjust component weights to tune sensitivity to any single indicator.

## Example Interpretation
- Composite 0.72 for Next Day → LONG_CALL with target Δ 0.50; strike_target at calculated level; note: Trade now.
- Composite −0.66 for 2 Weeks → LONG_PUT with target Δ −0.55; note: Wait if current premium exceeds target.

## Quick Start
Clone repo, install dependencies, run app and open UI at localhost. Configuration and deploy instructions are in the main README sections.

## References
- scikit-learn IsotonicRegression documentation (used for ATM smoothing).
- yfinance project and docs for data retrieval and limitations.
- Vercel Python functions runtime docs and configuration guidance.
- SciPy optimization and brentq root-finding documentation (used for SVI and IV routines).
- DiskCache project and docs for local disk-backed caching in development.
- redis-py Python client (recommended client and installation notes).