
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from utils.time_manager import get_time_manager

try:
    from data.data_manager import get_data_manager  # type: ignore
except Exception:
    get_data_manager = None

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a

@dataclass
class VolSpikeConfig:
    # legacy params kept for backward compatibility
    lookback: int = 20
    vol_threshold: float = 1.1   # legacy: last_vol / vol_ma >= threshold
    # dynamic params
    window: int = 30             # rolling window for dynamic statistics (excludes last bar)
    mult: float = 1.5           # median multiplier for spike detection (dynamic)
    z_thresh: float = 1.0        # z-score threshold for dynamic detection
    min_volume: float = 0.1     # minimum absolute volume to consider

class VolSpike:
    """VOL_SPIKE detector - dynamic mode (median + z-score) while keeping legacy outputs.
    Preserves existing class name and output keys for compatibility.
    """
    def __init__(self, cfg: VolSpikeConfig = VolSpikeConfig()):
        self.cfg = cfg
        self.tm = get_time_manager()

    def _no_signal_result(self,**kwargs):
        return {
            'name': 'VOL_SPIKE',
            'action': 'HOLD',
            'score': 0.0,
            'timestamp': self.tm.get_current_time(),
            'context': kwargs
        }

    def on_kline_close_3m(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Validate input
        if df is None or len(df) < max(5, self.cfg.window + 1):
            return self._no_signal_result(reason="insufficient_data")

        # Require necessary columns
        for col in ('open', 'high', 'low', 'close', 'volume'):
            if col not in df.columns:
                return self._no_signal_result(reason=f"missing_column_{col}")

        df = df.sort_index()
        v_series = df['volume'].astype(float)
        last_vol = float(v_series.iloc[-1])
        if last_vol < self.cfg.min_volume:
            return self._no_signal_result(reason="low_absolute_volume", last_vol=last_vol)

        # Compute legacy vol_ma (rolling mean) excluding last bar for compatibility
        try:
            vol_ma = float(v_series.rolling(self.cfg.lookback, min_periods=1).mean().iloc[-2])
        except Exception:
            vol_ma = float(np.mean(v_series[:-1])) if len(v_series) > 1 else float(v_series.iloc[-1])

        # Prepare historical window for dynamic stats excluding last bar
        if len(v_series) > (self.cfg.window + 1):
            hist = v_series.iloc[-(self.cfg.window+1):-1].astype(float).values
        else:
            hist = v_series.iloc[:-1].astype(float).values

        if len(hist) < 3:
            # fallback to all but last
            hist = v_series.iloc[:-1].astype(float).values

        vol_median = float(np.median(hist)) if len(hist) > 0 else float(vol_ma if vol_ma is not None else 0.0)
        vol_mean = float(np.mean(hist)) if len(hist) > 0 else vol_median
        vol_std = float(np.std(hist, ddof=0)) if len(hist) > 1 else 0.0

        # Ratios and z-score
        vol_ratio = (last_vol / (vol_median if vol_median > 0 else 1.0))
        legacy_ratio = (last_vol / (vol_ma if vol_ma > 0 else 1.0))
        vol_z = (last_vol - vol_mean) / vol_std if vol_std > 0 else float('inf')

        # Price breakout detection (conservative)
        closes = df['close'].astype(float).values
        highs = df['high'].astype(float).values
        lows = df['low'].astype(float).values
        last_close = float(closes[-1])
        prev_close = float(closes[-2]) if len(closes) > 1 else last_close
        recent_high = float(np.max(highs[-(self.cfg.window+1):-1])) if len(highs) > 1 else float(highs[-1])
        recent_low = float(np.min(lows[-(self.cfg.window+1):-1])) if len(lows) > 1 else float(lows[-1])

        price_break_up = last_close > recent_high and last_close > prev_close
        price_break_down = last_close < recent_low and last_close < prev_close

        # Dynamic detection criteria
        is_spike_dynamic = (vol_ratio >= self.cfg.mult) or (vol_z >= self.cfg.z_thresh)
        # Legacy detection (kept for compatibility)
        is_spike_legacy = (legacy_ratio >= self.cfg.vol_threshold)

        is_spike = bool(is_spike_dynamic or is_spike_legacy)

        # Score composition: give more weight to dynamic signals but include legacy
        try:
            score_vol = _clamp(np.log1p(max(0.0, vol_ratio - 1.0)) / np.log1p(self.cfg.mult * 5.0))
        except Exception:
            score_vol = 0.0
        score_z = _clamp(min(1.0, vol_z / (self.cfg.z_thresh * 2.0))) if vol_std > 0 else 0.0
        score_legacy = _clamp((legacy_ratio - 1.0) / max(1e-6, (self.cfg.vol_threshold - 1.0))) if self.cfg.vol_threshold > 1.0 else 0.0

        score_price = 0.0
        if price_break_up or price_break_down:
            score_price = 0.2

        # Weighted aggregation: dynamic-focused
        raw_score = score_vol * 0.5 + score_z * 0.25 + score_legacy * 0.15 + score_price * 0.10
        score = _clamp(raw_score, 0.0, 1.0)

        # Decide action
        action = "HOLD"
        if is_spike:
            if price_break_up:
                action = "BUY"
            elif price_break_down:
                action = "SELL"
            else:
                action = "BUY" if last_close > prev_close else "SELL"

        return {
            'name': 'VOL_SPIKE',
            'action': action,
            'score': float(score),
            'timestamp': self.tm.get_current_time(),
            'context': {
                'vol_ratio': float(vol_ratio),
                'legacy_ratio': float(legacy_ratio),
                'vol_median': float(vol_median),
                'vol_mean': float(vol_mean),
                'vol_std': float(vol_std),
                'vol_ma': float(vol_ma) if vol_ma is not None else None,
                'last_vol': float(last_vol),
                'is_spike_dynamic': bool(is_spike_dynamic),
                'is_spike_legacy': bool(is_spike_legacy),
                'price_break_up': bool(price_break_up),
                'price_break_down': bool(price_break_down),
            }
        }

