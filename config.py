import json
import os

# --- Config & constants ---
DEFAULT_CONFIG_PATH = "config.json"
ALLOWED_TRADE_POLICIES = ["prefer_spread", "allow_naked"]
DEFAULT_STOCKS = ["9988.HK", "0005.HK"]
STRIKE_STEP = 0.01
R = 0.02
Q = 0.0
TIMEFRAMES = {"next_day": 1, "2_weeks": 10, "3_months": 65, "6_months": 130}
DELTA_MAP = {1: (0.50, 0.20), 10: (0.50, 0.25), 65: (0.48, 0.22), 130: (0.42, 0.20)}
PREFER_OTM = {1: False, 10: True, 65: True, 130: True}
TRADE_POLICY = "prefer_spread"

# Option quality thresholds and caching
OPTION_MIN_VOLUME = 1
OPTION_MIN_OI = 1
MAX_BID_ASK_SPREAD_PCT = 0.30
MIN_STRIKES_PER_SLICE = 6

CACHE_DIR = "cache_dir"
CACHE_TTL_SECONDS = 30 * 60  # intra-day freshness for cached items
RECALIBRATE_INTERVAL_SECONDS = 24 * 3600  # daily recalibration
CALIBRATION_ALERT_THRESHOLD = 0.10  # RMSE threshold -> simple alert

# --- Config load/save ---
def load_config(path=DEFAULT_CONFIG_PATH):
    if not os.path.exists(path):
        sample = {"stocks": DEFAULT_STOCKS, "index": "^HSI", "strike_step": STRIKE_STEP, "trade_policy": TRADE_POLICY}
        with open(path, "w") as f:
            json.dump(sample, f, indent=2)
        return sample
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("stocks", DEFAULT_STOCKS)
    cfg.setdefault("index", "^HSI")
    cfg.setdefault("strike_step", STRIKE_STEP)
    cfg.setdefault("trade_policy", TRADE_POLICY)
    return cfg

def save_config(cfg, path=DEFAULT_CONFIG_PATH):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
