#!/usr/bin/env python3
import json
import logging
import threading
import time
from datetime import datetime

from flask import Flask, request, jsonify, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np

from config import (
    load_config,
    save_config,
    DEFAULT_STOCKS,
    ALLOWED_TRADE_POLICIES,
    TIMEFRAMES,
    R,
    Q,
    RECALIBRATE_INTERVAL_SECONDS,
    CALIBRATION_ALERT_THRESHOLD,
)
from data_fetching import get_option_chain_cached, parse_option_chain_with_quality
from svi import calibrate_surface_ssvi, expiry_to_T, svi_iv_from_params
from analysis import (
    implied_vol_from_price,
    returns_from_close,
    ewma_vol,
    fit_garch11_annualized,
    compute_rsi,
    compute_macd,
    vwma,
)
from recommendation import recommend_with_policy, score_and_components
from plotting import plot_base64

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dos_analyzer")

# --- Flask app ---
app = Flask(__name__)

# --- Helpers ---
def now_iso():
    return datetime.utcnow().isoformat() + "Z"

# --- Backtest / recalibration ---
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
                    parsed = get_option_chain_cached(t, spot, force_refresh=True)
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

            # try to use latest intraday price for same-day freshness
            tk = yf.Ticker(t)
            latest_price = float(df['Close'].iloc[-1])
            try:
                intraday = tk.history(period="1d", interval="1m", auto_adjust=False)
                if not intraday.empty:
                    latest_price = float(intraday['Close'].iloc[-1])
            except Exception:
                pass

            # get cached option chain and fitted surface (daily keyed)
            parsed = get_option_chain_cached(t, latest_price)
            iv_surface = calibrate_surface_ssvi(parsed, latest_price) if parsed else {}

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


@app.route("/", methods=["GET"])
def index():
    return render_template_string("""
    <!doctype html>
    <html><head><meta charset="utf-8"><title>Analyzer</title></head>
    <body style="font-family:system-ui;padding:2rem">
      <h2>Daily Option Strategy Analyzer</h2>
      <ul>
        <li><a href="/config">GET /config (JSON)</a></li>
        <li>POST /analyze (JSON) - body optional: {"stocks":["9988.HK","0005.HK"]}</li>
      </ul>
    </body></html>
    """")


if __name__ == "__main__":
    logger.info("Starting app.")
    start_background_jobs()
    app.run(host="0.0.0.0", port=5000, threaded=True)
