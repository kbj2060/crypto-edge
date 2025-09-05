
# zscore_mean_reversion_bot.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

@dataclass
class ZScoreConfig:
    window: int = 60
    use_log: bool = False
    z_thresh: float = 2.0
    exit_z: float = 0.5
    atr_period: int = 14
    stop_atr_mult: float = 2.0
    take_profit_atr_mult: Optional[float] = 2.5
    min_volume: float = 0.0
    min_history: int = 200
    mode: str = "price"         # or "vwap_residual"
    vwap_window: int = 390

def ensure_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()

def vwap_session(df: pd.DataFrame) -> pd.Series:
    pv = (df['close'].astype(float) * df['volume'].astype(float)).cumsum()
    volcum = df['volume'].astype(float).cumsum().replace(0, 1e-9)
    return pv / volcum

def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=1).mean().shift(1)
    std = series.rolling(window, min_periods=1).std(ddof=0).shift(1).replace(0, np.nan)
    z = (series - mean) / std
    return z.fillna(0.0)

def generate_signal(df: pd.DataFrame, cfg: ZScoreConfig = ZScoreConfig()) -> Dict[str, Any]:
    df = ensure_index(df)
    if len(df) < max(cfg.min_history, cfg.window + 5):
        return {'name': 'Z_SCORE_MEANREV', 'action': 'HOLD', 'score': 0.0, 'confidence': 0.0, 'context': {'reason': 'insufficient_history', 'n': len(df)}}

    last_vol = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0.0
    if cfg.min_volume and last_vol < cfg.min_volume:
        return {'name': 'Z_SCORE_MEANREV', 'action': 'HOLD', 'score': 0.0, 'confidence': 0.0, 'context': {'reason': 'low_volume', 'last_vol': last_vol}}

    if cfg.mode == "vwap_residual":
        full_vwap = vwap_session(df)
        series = (df['close'].astype(float) - full_vwap).astype(float)
    else:
        series = df['close'].astype(float)
        if cfg.use_log:
            series = np.log(series.replace(0, np.nan)).fillna(method='ffill')

    z = compute_zscore(series, cfg.window)
    last_z = float(z.iloc[-1])

    atr_val = float(atr(df, cfg.atr_period).iloc[-1]) if len(df) > 0 else 0.0
    last_price = float(df['close'].iloc[-1])

    action = 'HOLD'; score = 0.0; conf = 0.0; entry = last_price; stop = None; tp = None
    if last_z >= cfg.z_thresh:
        action = 'SELL'
        score = min(1.0, abs(last_z) / (cfg.z_thresh * 2.0))
        conf = min(1.0, (abs(last_z) - cfg.z_thresh + 1.0) / (abs(last_z) + 1.0))
        entry = last_price
        stop = entry + max(1e-6, cfg.stop_atr_mult * atr_val)
        if cfg.take_profit_atr_mult is not None:
            tp = entry - cfg.take_profit_atr_mult * atr_val
    elif last_z <= -cfg.z_thresh:
        action = 'BUY'
        score = min(1.0, abs(last_z) / (cfg.z_thresh * 2.0))
        conf = min(1.0, (abs(last_z) - cfg.z_thresh + 1.0) / (abs(last_z) + 1.0))
        entry = last_price
        stop = entry - max(1e-6, cfg.stop_atr_mult * atr_val)
        if cfg.take_profit_atr_mult is not None:
            tp = entry + cfg.take_profit_atr_mult * atr_val

    return {
        'name': 'Z_SCORE_MEANREV',
        'action': action,
        'score': float(score),
        'confidence': float(conf),
        'entry': float(entry) if entry is not None else None,
        'stop': float(stop) if stop is not None else None,
        'tp': float(tp) if tp is not None else None,
        'context': {'last_z': last_z, 'z_threshold': cfg.z_thresh, 'atr': atr_val, 'mode': cfg.mode}
    }
