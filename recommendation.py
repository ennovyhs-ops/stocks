import numpy as np
from config import DELTA_MAP, PREFER_OTM, R, Q
from analysis import call_price_bs, put_price_bs, call_delta, put_delta, bs_d1
from svi import expiry_to_T
from scipy.optimize import brentq

def round_strike(x):
    from config import STRIKE_STEP
    return float(np.round(np.round(x / STRIKE_STEP) * STRIKE_STEP, 2))

def blended_sigma(close_series, iv_surface=None, S=None, K=None, expiry=None, weights=None):
    from analysis import ewma_vol, fit_garch11_annualized
    from svi import svi_iv_from_params

    if weights is None:
        weights = {"implied": 0.6, "ewma": 0.25, "garch": 0.15}
    ewma = ewma_vol(close_series, span_days=60)
    garch = fit_garch11_annualized(close_series, horizon_days=1)
    implied = None
    if iv_surface and expiry and S and K:
        params = iv_surface.get(expiry)
        if params:
            T = expiry_to_T(expiry)
            try:
                implied = svi_iv_from_params(params, K, S, T)
            except Exception:
                implied = None
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

def strike_for_delta(S, target_delta, is_call, r, q, sigma, T, prefer_otm=False):
    from config import STRIKE_STEP
    abs_target = abs(target_delta)
    def f(K):
        if sigma <= 1e-6 or T <= 1e-12:
            return (1.0 if (is_call and S > K) else 0.0) - abs_target
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
        K = brentq(f, low, high, maxiter=300)
        return round_strike(K)
    except Exception:
        # logger.debug("strike_for_delta brent failed S=%.4f target=%.3f T=%.4f", S, target_delta, T)
        return round_strike(S)

def recommend_with_policy(S, close_series, composite, days, iv_surface=None, policy="prefer_spread"):
    T = max(1, days) / 365.0
    expiry = next(iter(iv_surface.keys()), None) if iv_surface else None
    sigma_blend = blended_sigma(close_series, iv_surface, S=S, K=S, expiry=expiry)
    nearest = min(DELTA_MAP.keys(), key=lambda k: abs(k - days))
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
        side = "NEUTRAL_SELL_CREDIT"; is_call = composite > 0; target_delta = sell_delta if composite > 0 else -sell_delta
    abs_target = abs(target_delta)
    strike_target = strike_for_delta(S, abs_target, is_call, R, Q, sigma_blend, T, prefer_otm=prefer_otm)
    atm = round_strike(S)
    conservative = round_strike((strike_target + atm) / 2.0)
    final = {"side": side, "is_call": bool(is_call), "target_delta": float(np.sign(target_delta) * abs_target),
             "strike_target": strike_target, "strike_atm": atm, "strike_conservative": conservative,
             "sigma_used": round(sigma_blend, 4), "T_years": round(T, 4)}
    if is_call:
        long_price = call_price_bs(S, conservative, R, Q, sigma_blend, T)
        short_price = call_price_bs(S, strike_target, R, Q, sigma_blend, T)
    else:
        long_price = put_price_bs(S, conservative, R, Q, sigma_blend, T)
        short_price = put_price_bs(S, strike_target, R, Q, sigma_blend, T)
    if policy == "prefer_spread" and final["side"] in ("LONG_CALL", "LONG_PUT"):
        spread_side = "BUY_CALL_SPREAD" if is_call else "BUY_PUT_SPREAD"
        long_leg = conservative
        short_leg = strike_target if ((is_call and strike_target > long_leg) or (not is_call and strike_target < long_leg)) else atm
        if is_call:
            long_p = call_price_bs(S, long_leg, R, Q, sigma_blend, T); short_p = call_price_bs(S, short_leg, R, Q, sigma_blend, T)
        else:
            long_p = put_price_bs(S, long_leg, R, Q, sigma_blend, T); short_p = put_price_bs(S, short_leg, R, Q, sigma_blend, T)
        spread_cost = round(long_p - short_p, 6)
        final.update({"side": spread_side, "long_strike": long_leg, "short_strike": short_leg, "long_premium": round(long_p, 6), "short_premium": round(short_p, 6), "spread_cost": spread_cost})
    else:
        final.update({"leg_strike": conservative, "leg_premium": round(long_price, 6), "note": "Trade now." if abs(composite) >= 0.3 else "Consider wait for better setup or premium."})
    return final
