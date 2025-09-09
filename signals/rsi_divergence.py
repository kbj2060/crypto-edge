# htf_rsi_divergence.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from data.binance_dataloader import BinanceDataLoader
from data.data_manager import get_data_manager
from utils.time_manager import get_time_manager

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / (down + 1e-9)
    return 100 - (100 / (1 + rs))

def _stoch_rsi(rsi: pd.Series, period: int = 14) -> pd.Series:
    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()
    return (rsi - min_rsi) / (max_rsi - min_rsi + 1e-9)
# --- Replace HTFRSIDivCfg and RSIDivergence with the following improved implementation ---
from dataclasses import dataclass

@dataclass
class HTFRSIDivCfg:
    symbol: str = "ETHUSDT"
    interval: str = "3m"
    rsi_period: int = 8
    stoch_period: int = 14
    lookback_bars: int = 300
    swing_window: int = 3            # local extrema window
    min_rsi_delta: float = 1.0       # minimum RSI rise/fall to count as divergence
    min_price_move_pct: float = 0.0005
    min_score: float = 0.35
    use_volume_filter: bool = False
    min_volume_multiplier: float = 0.5
    debug: bool = False              # set True to print debug lines

class RSIDivergence:
    def __init__(self, cfg: HTFRSIDivCfg = HTFRSIDivCfg()):
        self.cfg = cfg
        self.data_manager = get_data_manager()
        self.tm = get_time_manager()

    # helper: local extrema indices
    def _local_extrema_idxs(self, series: pd.Series, window: int, kind: str = "low"):
        idxs = []
        L = len(series)
        # 기존: 완벽한 극값만 찾음 (너무 엄격)
        # 수정: 약간 완화된 조건
        for i in range(window, L - window):
            if kind == "low":
                # 완화된 조건: 중심값이 양쪽 대부분보다 낮으면 OK
                left_lower = sum(series.iloc[i] <= series.iloc[i-j] for j in range(1, window+1))
                right_lower = sum(series.iloc[i] <= series.iloc[i+j] for j in range(1, window+1))
                if left_lower >= (window * 0.7) and right_lower >= (window * 0.7):  # 70% 이상
                    idxs.append(i)
            else:  # "high"
                left_higher = sum(series.iloc[i] >= series.iloc[i-j] for j in range(1, window+1))
                right_higher = sum(series.iloc[i] >= series.iloc[i+j] for j in range(1, window+1))
                if left_higher >= (window * 0.7) and right_higher >= (window * 0.7):
                    idxs.append(i)
        return idxs

    def _score_divergence(self, price_prev, price_now, rsi_prev, rsi_now, vol_now=None, cfg=None):
        cfg = cfg or self.cfg
        # price move percent (prev -> now)
        price_pct = (price_prev - price_now) / price_prev if price_prev > 0 else 0.0
        rsi_delta = rsi_now - rsi_prev
        s = 0.0
        # price credit (partial)
        if price_pct >= cfg.min_price_move_pct:
            s += 0.35
        else:
            s += 0.15 * (price_pct / (cfg.min_price_move_pct + 1e-12))
        # rsi credit
        if rsi_delta >= cfg.min_rsi_delta:
            s += 0.45
        else:
            s += 0.2 * (rsi_delta / (cfg.min_rsi_delta + 1e-12))
        # volume soft bonus/penalty
        if cfg.use_volume_filter and vol_now is not None:
            # if recent volume is weak penalize a bit
            avg_vol = vol_now if np.isscalar(vol_now) else np.mean(vol_now) if len(vol_now)>0 else None
            if avg_vol is not None and avg_vol < (cfg.min_volume_multiplier * avg_vol):
                s *= 0.9
        s = max(0.0, min(1.0, s))
        return s, price_pct, rsi_delta

    def on_kline_close_3m(self) -> Optional[Dict[str, Any]]:
        # fetch lookback bars
        df = self.data_manager.get_latest_data(self.cfg.lookback_bars)
        if df is None or len(df) < max(self.cfg.rsi_period + self.cfg.swing_window + 5, 30):
            return None

        close = pd.to_numeric(df["close"].astype(float)).reset_index(drop=True)
        rsi_series = _rsi(close, self.cfg.rsi_period)
        stoch_series = _stoch_rsi(rsi_series, self.cfg.stoch_period)

        # find local lows/highs using swing_window
        lows_idx = self._local_extrema_idxs(close, self.cfg.swing_window, kind="low")
        highs_idx = self._local_extrema_idxs(close, self.cfg.swing_window, kind="high")

        action = "HOLD"
        score = 0.0
        conf = 0.0
        context = {}

        # bullish divergence: need at least two local lows (prev, curr)
        if len(lows_idx) >= 2:
            prev = lows_idx[-2]
            curr = lows_idx[-1]
            price_prev = float(close.iloc[prev])
            price_now = float(close.iloc[curr])
            rsi_prev = float(rsi_series.iloc[prev])
            rsi_now = float(rsi_series.iloc[curr])
            vol_recent = df['volume'].astype(float).iloc[-(self.cfg.swing_window*2):].values if 'volume' in df.columns else None
            s, price_pct, rsi_delta = self._score_divergence(price_prev, price_now, rsi_prev, rsi_now, vol_recent, self.cfg)
            
            if s >= self.cfg.min_score:
                action = "BUY"
                score = float(s)
                context.update({
                    "type":"bull_div",
                    "prev_idx": int(prev), "curr_idx": int(curr),
                    "prev_low": price_prev, "curr_low": price_now,
                    "rsi_prev": rsi_prev, "rsi_now": rsi_now,
                    "price_drop_pct": price_pct, "rsi_delta": rsi_delta
                })

        # bearish divergence
        if action == "HOLD" and len(highs_idx) >= 2:
            prev = highs_idx[-2]
            curr = highs_idx[-1]
            price_prev = float(close.iloc[prev])
            price_now = float(close.iloc[curr])
            rsi_prev = float(rsi_series.iloc[prev])
            rsi_now = float(rsi_series.iloc[curr])
            vol_recent = df['volume'].astype(float).iloc[-(self.cfg.swing_window*2):].values if 'volume' in df.columns else None
            # for bearish, invert price delta logic (price rose) and demand rsi drop
            price_rise = (price_now - price_prev) / price_prev if price_prev>0 else 0.0
            # reuse scoring but flip sign expectation for rsi_delta (want negative)
            # feed absolute rsi delta into scoring but adjust check below
            s, _pp, _rd = self._score_divergence(price_prev, price_now, rsi_prev, rsi_now, vol_recent, self.cfg)
            rsi_delta = rsi_now - rsi_prev
            
            # require rsi went down sufficiently (negative)
            if rsi_delta <= -self.cfg.min_rsi_delta and s >= self.cfg.min_score:
                action = "SELL"
                score = float(s)
                context.update({
                    "type":"bear_div",
                    "prev_idx": int(prev), "curr_idx": int(curr),
                    "prev_high": price_prev, "curr_high": price_now,
                    "rsi_prev": rsi_prev, "rsi_now": rsi_now,
                    "price_rise_pct": price_rise, "rsi_delta": rsi_delta
                })

        conf = score

        return {
            "name": "RSI_DIV",
            "action": action,
            "score": float(score),
            "timestamp": self.tm.get_current_time(),
            "context": context
        }
# --- end replacement ---
