# vpvr_microbot.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class VPVRConfig:
    n_bins: int = 64
    lookback_bars: int = 240
    poc_tolerance: float = 0.002
    min_profile_volume: float = 1.0
    lookback_retest_bars: int = 3
    side_bias: Optional[str] = None  # 'LONG' or 'SHORT' or None

def ensure_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()

def compute_vpvr(df: pd.DataFrame, n_bins: int = 64):
    prices = df['close'].astype(float).values
    vols = df['volume'].astype(float).values
    if len(prices) == 0:
        return None, None, None
    p_min, p_max = float(np.min(prices)), float(np.max(prices))
    if p_max == p_min:
        return None, None, None
    bins = np.linspace(p_min, p_max, n_bins + 1)
    vol_hist = np.zeros(n_bins)
    bin_idx = np.digitize(prices, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    for i, b in enumerate(bin_idx):
        vol_hist[b] += vols[i]
    poc_idx = int(np.argmax(vol_hist))
    poc_price = float((bins[poc_idx] + bins[poc_idx + 1]) / 2.0)
    return bins, vol_hist, poc_price

def generate_signal(df_5m: pd.DataFrame, cfg: VPVRConfig = VPVRConfig()) -> Dict[str, Any]:
    df_5m = ensure_index(df_5m)
    if len(df_5m) < max(cfg.lookback_bars, 10):
        return {'name': 'VPVR_MICROBOT', 'action': 'HOLD', 'score': 0.0, 'confidence': 0.0, 'context': {'reason': 'insufficient_bars'}}

    profile_df = df_5m.iloc[-cfg.lookback_bars:]
    bins, vol_hist, poc = compute_vpvr(profile_df, n_bins=cfg.n_bins)
    if bins is None:
        return {'name': 'VPVR_MICROBOT', 'action': 'HOLD', 'score': 0.0, 'confidence': 0.0, 'context': {'reason': 'vpvr_fail'}}

    recent_close = float(df_5m['close'].iloc[-1])
    tol_price = poc * cfg.poc_tolerance
    within = abs(recent_close - poc) <= tol_price

    recent_prices = df_5m['close'].iloc[-cfg.lookback_retest_bars - 1:-1].astype(float).values
    retest = any(abs(p - poc) <= tol_price for p in recent_prices) if len(recent_prices) > 0 else False

    action = 'HOLD'; score = 0.0; conf = 0.0; entry = None; stop = None
    if within and retest and float(np.sum(vol_hist)) >= cfg.min_profile_volume:
        last_close = recent_close
        if cfg.side_bias == 'LONG':
            action, score, conf = 'BUY', 0.8, 0.7
        elif cfg.side_bias == 'SHORT':
            action, score, conf = 'SELL', 0.8, 0.7
        else:
            action = 'BUY' if last_close >= poc else 'SELL'
            score, conf = 0.75, 0.6
        entry = last_close
        bins_arr = np.array(bins)
        bin_w = bins_arr[1] - bins_arr[0]
        stop = poc - 1.5 * bin_w if action == 'BUY' else poc + 1.5 * bin_w

    return {
        'name': 'VPVR_MICROBOT',
        'action': action,
        'score': float(score),
        'confidence': float(conf),
        'entry': float(entry) if entry is not None else None,
        'stop': float(stop) if stop is not None else None,
        'context': {'poc': float(poc), 'within_tol': bool(within), 'retest': bool(retest), 'vol_sum': float(np.sum(vol_hist))}
    }
