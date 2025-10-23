import logging
from datetime import datetime
import yfinance as yf
import numpy as np
try:
    from diskcache import Cache
    CACHE_AVAILABLE = True
except Exception:
    Cache = None
    CACHE_AVAILABLE = False

from config import (
    MAX_BID_ASK_SPREAD_PCT,
    OPTION_MIN_VOLUME,
    OPTION_MIN_OI,
    CACHE_DIR,
    CACHE_TTL_SECONDS,
)

logger = logging.getLogger(__name__)

# initialize cache
cache = Cache(CACHE_DIR) if CACHE_AVAILABLE else None
if not CACHE_AVAILABLE:
    logger.info("diskcache not available; caching disabled (install diskcache for persistent caching)")

def today_str():
    return datetime.today().isoformat()

def parse_option_chain_with_quality(tk, spot):
    out = {}
    try:
        expiries = getattr(tk, "options", []) or []
    except Exception:
        return {}
    for exp in expiries[:8]:
        try:
            chain = tk.option_chain(exp)
            calls = chain.calls
            puts = chain.puts
            strikes = sorted(set(calls['strike'].tolist() + puts['strike'].tolist()))
            rows = []
            for K in strikes:
                call_mid = None; put_mid = None
                crow = calls[calls['strike'] == K]
                prow = puts[puts['strike'] == K]
                # Calls
                if not crow.empty:
                    try:
                        cbid = float(crow['bid'].iloc[0]); cask = float(crow['ask'].iloc[0])
                        if cbid > 0 and cask > 0:
                            mid = (cbid + cask) / 2.0
                            spread_pct = (cask - cbid) / max(1e-8, mid)
                            vol = int(crow.get('volume', [0]).iloc[0]) if 'volume' in crow else 0
                            oi = int(crow.get('openInterest', [0]).iloc[0]) if 'openInterest' in crow else 0
                            if spread_pct <= MAX_BID_ASK_SPREAD_PCT and (vol >= OPTION_MIN_VOLUME or oi >= OPTION_MIN_OI):
                                call_mid = mid
                        if call_mid is None:
                            last = float(crow.get('lastPrice', np.nan).iloc[0]) if 'lastPrice' in crow else None
                            if last and last > 0:
                                call_mid = last
                    except Exception:
                        pass
                # Puts
                if not prow.empty:
                    try:
                        pbid = float(prow['bid'].iloc[0]); pask = float(prow['ask'].iloc[0])
                        if pbid > 0 and pask > 0:
                            mid = (pbid + pask) / 2.0
                            spread_pct = (pask - pbid) / max(1e-8, mid)
                            vol = int(prow.get('volume', [0]).iloc[0]) if 'volume' in prow else 0
                            oi = int(prow.get('openInterest', [0]).iloc[0]) if 'openInterest' in prow else 0
                            if spread_pct <= MAX_BID_ASK_SPREAD_PCT and (vol >= OPTION_MIN_VOLUME or oi >= OPTION_MIN_OI):
                                put_mid = mid
                        if put_mid is None:
                            last = float(prow.get('lastPrice', np.nan).iloc[0]) if 'lastPrice' in prow else None
                            if last and last > 0:
                                put_mid = last
                    except Exception:
                        pass
                rows.append({"strike": float(K), "call_mid": call_mid, "put_mid": put_mid})
            out[datetime.fromisoformat(exp).isoformat()] = rows
        except Exception:
            logger.exception("Failed parsing option chain for expiry %s", exp)
            continue
    return out

def cache_key_option_chain(ticker):
    return f"optchain:{ticker}:{today_str()}"

def get_option_chain_cached(ticker, spot, force_refresh=False):
    if not CACHE_AVAILABLE:
        return parse_option_chain_with_quality(yf.Ticker(ticker), spot)
    key = cache_key_option_chain(ticker)
    if force_refresh:
        cache.delete(key)
    data = cache.get(key)
    if data is not None:
        logger.debug("Option chain cache hit %s", ticker)
        return data
    tk = yf.Ticker(ticker)
    parsed = parse_option_chain_with_quality(tk, spot)
    cache.set(key, parsed, expire=CACHE_TTL_SECONDS)
    logger.debug("Option chain cached %s", ticker)
    return parsed
