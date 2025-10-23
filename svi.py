import logging
import numpy as np
from scipy.optimize import least_squares
from math import sqrt, log

try:
    from sklearn.isotonic import IsotonicRegression
    ISOTONIC_AVAILABLE = True
except Exception:
    ISOTONIC_AVAILABLE = False

try:
    import py_ssvi
    SSVI_AVAILABLE = True
except Exception:
    py_ssvi = None
    SSVI_AVAILABLE = False

from config import MIN_STRIKES_PER_SLICE

logger = logging.getLogger(__name__)

def expiry_to_T(exp_iso_str):
    """Converts ISO 8601 expiry string to time-to-expiry in years."""
    from datetime import datetime
    expiry_dt = datetime.fromisoformat(exp_iso_str.replace('Z', '+00:00'))
    now_dt = datetime.now(expiry_dt.tzinfo)
    return (expiry_dt - now_dt).total_seconds() / (365.0 * 24 * 3600)


def svi_total_variance(params, x):
    a, b, rho, m, sigma = params
    return a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))

def svi_residuals(params, x, w_market):
    return svi_total_variance(params, x) - w_market

def fit_svi_slice(strikes, ivs, F, T):
    if T <= 0 or len(strikes) < MIN_STRIKES_PER_SLICE:
        return None
    x = np.log(np.array(strikes) / float(F))
    w_market = (np.array(ivs) ** 2) * T
    a0 = max(1e-8, float(np.mean(w_market) * 0.8))
    p0 = [a0, 0.1, 0.0, float(np.median(x)), 0.2]
    lower = [0.0, 1e-8, -0.999, float(np.min(x))*2, 1e-8]
    upper = [float(np.max(w_market))*5 + 1.0, 5.0, 0.999, float(np.max(x))*2, 5.0]
    try:
        res = least_squares(svi_residuals, p0, bounds=(lower, upper), args=(x, w_market), xtol=1e-8, ftol=1e-8, max_nfev=5000)
        if res.success:
            return res.x.tolist()
        logger.debug("SVI fit not successful: %s", res.message)
        return None
    except Exception:
        logger.exception("SVI fit exception")
        return None

def svi_iv_from_params(params, K, F, T):
    a, b, rho, m, sigma = params
    x = log(K / F)
    w = a + b * (rho * (x - m) + sqrt((x - m)**2 + sigma**2))
    return sqrt(max(w / max(T, 1e-12), 0.0))

def enforce_atm_monotonic_isotonic(svi_surface):
    items = []
    for exp_iso, params in svi_surface.items():
        if not params:
            continue
        T = expiry_to_T(exp_iso)
        w_atm = svi_total_variance(params, 0.0)
        items.append((exp_iso, T, float(w_atm)))
    if not items:
        return svi_surface
    items.sort(key=lambda x: x[1])
    Ts = np.array([x[1] for x in items])
    ws = np.array([x[2] for x in items])
    if ISOTONIC_AVAILABLE:
        try:
            ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
            w_iso = ir.fit_transform(Ts, ws)
            for (exp_iso, _, _), w_target in zip(items, w_iso):
                params = svi_surface.get(exp_iso)
                if not params:
                    continue
                a,b,rho,m,sig = params
                atm_contrib = b * (rho * (-m) + sqrt(m**2 + sig**2))
                params[0] = max(0.0, float(w_target) - atm_contrib)
                svi_surface[exp_iso] = params
            return svi_surface
        except Exception:
            logger.exception("Isotonic enforcement failed; falling back to naive monotonic lift")
    # naive fallback: non-decreasing ATM by lifting small ones
    prev = -1.0
    for exp_iso, T, w in items:
        params = svi_surface.get(exp_iso)
        if not params:
            continue
        if w < prev:
            bump = prev - w + 1e-8
            params[0] = params[0] + bump
            svi_surface[exp_iso] = params
            w = w + bump
        prev = w
    return svi_surface

def calibrate_surface_ssvi(parsed_option_rows, spot):
    if SSVI_AVAILABLE and py_ssvi is not None:
        try:
            surface_input = {}
            for exp_iso, rows in parsed_option_rows.items():
                strikes = []
                ivs = []
                T = expiry_to_T(exp_iso)
                for r in rows:
                    k = r['strike']
                    iv = r.get('iv_call') if r.get('iv_call') is not None else r.get('iv_put')
                    if iv is not None and iv > 0:
                        strikes.append(k); ivs.append(iv)
                if len(strikes) >= MIN_STRIKES_PER_SLICE:
                    surface_input[exp_iso] = {"T": T, "strikes": strikes, "ivs": ivs}
            if not surface_input:
                raise RuntimeError("No valid slices for SSVI")
            if hasattr(py_ssvi, "calibrate_surface"):
                ssvi_surface = py_ssvi.calibrate_surface(surface_input, spot=spot)
                logger.info("SSVI calibrated via py_ssvi.calibrate_surface")
                return ssvi_surface
            if hasattr(py_ssvi, "ssvi") and hasattr(py_ssvi.ssvi, "calibrate_surface"):
                ssvi_surface = py_ssvi.ssvi.calibrate_surface(surface_input, spot=spot)
                logger.info("SSVI calibrated via py_ssvi.ssvi.calibrate_surface")
                return ssvi_surface
            logger.info("py_ssvi present but no known calibrate_surface API; falling back")
        except Exception:
            logger.exception("SSVI calibration via py_ssvi failed; falling back")
    # Fallback: per-slice SVI fits then isotonic ATM smoothing
    svi_surface = {}
    for exp_iso, rows in parsed_option_rows.items():
        strikes = []
        ivs = []
        for r in rows:
            k = r['strike']
            iv = r.get('iv_call') if r.get('iv_call') is not None else r.get('iv_put')
            if iv is not None and iv > 0:
                strikes.append(k); ivs.append(iv)
        if len(strikes) < MIN_STRIKES_PER_SLICE:
            svi_surface[exp_iso] = None
            continue
        params = fit_svi_slice(strikes, ivs, spot, expiry_to_T(exp_iso))
        svi_surface[exp_iso] = params
    svi_surface = enforce_atm_monotonic_isotonic(svi_surface)
    return svi_surface
