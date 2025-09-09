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
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def _stoch_rsi(rsi: pd.Series, period: int = 14) -> pd.Series:
    min_rsi = rsi.rolling(window=period).min()
    max_rsi = rsi.rolling(window=period).max()
    return (rsi - min_rsi) / (max_rsi - min_rsi + 1e-9)
# --- Replace HTFRSIDivCfg and RSIDivergence with the following improved implementation ---
from dataclasses import dataclass

@dataclass
class HTFRSIDivCfg:
    symbol: str = "ETHUSDC"
    interval: str = "3m"
    rsi_period: int = 7          # 5 -> 7 (더 안정적)
    stoch_period: int = 10       # 14 -> 10 (더 민감)
    lookback_bars: int = 200     # 300 -> 200 (단축)
    swing_window: int = 3        # 2 -> 3 (완화)
    min_rsi_delta: float = 0.2   # 0.5 -> 0.2 (완화)
    min_price_move_pct: float = 0.0001  # 0.0002 -> 0.0001 (완화)
    min_score: float = 0.15      # 0.25 -> 0.15 (완화)
    use_volume_filter: bool = False
    min_volume_multiplier: float = 0.3  # 0.5 -> 0.3 (완화)
    debug: bool = True           # True로 변경 (디버깅 활성화)

class RSIDivergence:
    def __init__(self, cfg: HTFRSIDivCfg = HTFRSIDivCfg()):
        self.cfg = cfg
        self.data_manager = get_data_manager()
        self.tm = get_time_manager()

    # helper: local extrema indices
    def _local_extrema_idxs(self, series: pd.Series, window: int, kind: str = "low"):
        idxs = []
        L = len(series)
        # 더 완화된 극값 감지 조건
        for i in range(window, L - window):
            if kind == "low":
                # 더 완화된 조건: 중심값이 양쪽 절반 이상보다 낮으면 OK
                left_lower = sum(series.iloc[i] <= series.iloc[i-j] for j in range(1, window+1))
                right_lower = sum(series.iloc[i] <= series.iloc[i+j] for j in range(1, window+1))
                if left_lower >= (window * 0.5) and right_lower >= (window * 0.5):  # 50% 이상으로 완화
                    idxs.append(i)
            else:  # "high"
                left_higher = sum(series.iloc[i] >= series.iloc[i-j] for j in range(1, window+1))
                right_higher = sum(series.iloc[i] >= series.iloc[i+j] for j in range(1, window+1))
                if left_higher >= (window * 0.5) and right_higher >= (window * 0.5):  # 50% 이상으로 완화
                    idxs.append(i)
        return idxs

    def _score_divergence(self, price_prev, price_now, rsi_prev, rsi_now, vol_now=None, cfg=None):
        cfg = cfg or self.cfg
        # price move percent (prev -> now)
        price_pct = (price_prev - price_now) / price_prev if price_prev > 0 else 0.0
        rsi_delta = rsi_now - rsi_prev
        s = 0.0
        
        # 더 유연한 가격 크레딧 계산
        if price_pct >= cfg.min_price_move_pct:
            s += 0.40  # 0.35 -> 0.40 (증가)
        else:
            # 부분 점수도 더 관대하게
            partial_score = 0.25 * (price_pct / (cfg.min_price_move_pct + 1e-12))  # 0.15 -> 0.25
            s += partial_score
        
        # RSI 크레딧도 더 유연하게
        if rsi_delta >= cfg.min_rsi_delta:
            s += 0.50  # 0.45 -> 0.50 (증가)
        else:
            # 부분 점수도 더 관대하게
            partial_score = 0.30 * (rsi_delta / (cfg.min_rsi_delta + 1e-12))  # 0.20 -> 0.30
            s += partial_score
        
        # 볼륨 필터 완화
        if cfg.use_volume_filter and vol_now is not None:
            avg_vol = vol_now if np.isscalar(vol_now) else np.mean(vol_now) if len(vol_now)>0 else None
            if avg_vol is not None and avg_vol < (cfg.min_volume_multiplier * avg_vol):
                s *= 0.95  # 0.9 -> 0.95 (완화)
        
        # 최소 점수 보장 (다이버전스 감지 시 최소 0.1 점수)
        if abs(rsi_delta) > 0.05:  # RSI 변화가 있으면
            s = max(s, 0.1)
            
        s = max(0.0, min(1.0, s))
        return s, price_pct, rsi_delta

    def on_kline_close_3m(self) -> Optional[Dict[str, Any]]:
        # fetch lookback bars
        df = self.data_manager.get_latest_data(self.cfg.lookback_bars)
        if df is None or len(df) < max(self.cfg.rsi_period + self.cfg.swing_window + 5, 30):
            if self.cfg.debug:
                print(f"[RSI_DIV] 데이터 부족: 필요={self.cfg.rsi_period + self.cfg.swing_window + 5}, 실제={len(df) if df is not None else 'None'}")
            return None

        close = pd.to_numeric(df["close"].astype(float)).reset_index(drop=True)
        rsi_series = _rsi(close, self.cfg.rsi_period)
        stoch_series = _stoch_rsi(rsi_series, self.cfg.stoch_period)

        # find local lows/highs using swing_window
        lows_idx = self._local_extrema_idxs(close, self.cfg.swing_window, kind="low")
        highs_idx = self._local_extrema_idxs(close, self.cfg.swing_window, kind="high")

        if self.cfg.debug:
            print(f"[RSI_DIV] 극값 감지: 저점 {len(lows_idx)}개, 고점 {len(highs_idx)}개")

        action = "HOLD"
        score = 0.0
        context = {}

        # bullish divergence: need at least two local lows (prev, curr)
        if len(lows_idx) >= 2:
            prev = lows_idx[-2]
            curr = lows_idx[-1]
            price_prev = float(close.iloc[prev])
            price_now = float(close.iloc[curr])
            rsi_prev = float(stoch_series.iloc[prev])
            rsi_now = float(stoch_series.iloc[curr])
            vol_recent = df['quote_volume'].astype(float).iloc[-(self.cfg.swing_window*2):].values if 'quote_volume' in df.columns else None
            s, price_pct, rsi_delta = self._score_divergence(price_prev, price_now, rsi_prev, rsi_now, vol_recent, self.cfg)
            
            if self.cfg.debug:
                print(f"[RSI_DIV] 강세 다이버전스 체크: 점수={s:.3f}, 가격변화={price_pct:.4f}, RSI변화={rsi_delta:.3f}")
            
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
                if self.cfg.debug:
                    print(f"[RSI_DIV] 강세 다이버전스 신호 생성 ✓ (점수: {score:.3f})")
            else:
                if self.cfg.debug:
                    print(f"[RSI_DIV] 강세 다이버전스 점수 부족: {s:.3f} < {self.cfg.min_score}")

        # bearish divergence
        if action == "HOLD" and len(highs_idx) >= 2:
            prev = highs_idx[-2]
            curr = highs_idx[-1]
            price_prev = float(close.iloc[prev])
            price_now = float(close.iloc[curr])
            rsi_prev = float(stoch_series.iloc[prev])
            rsi_now = float(stoch_series.iloc[curr])
            vol_recent = df['quote_volume'].astype(float).iloc[-(self.cfg.swing_window*2):].values if 'quote_volume' in df.columns else None
            # for bearish, invert price delta logic (price rose) and demand rsi drop
            price_rise = (price_now - price_prev) / price_prev if price_prev>0 else 0.0
            # reuse scoring but flip sign expectation for rsi_delta (want negative)
            # feed absolute rsi delta into scoring but adjust check below
            s, _pp, _rd = self._score_divergence(price_prev, price_now, rsi_prev, rsi_now, vol_recent, self.cfg)
            rsi_delta = rsi_now - rsi_prev
            
            if self.cfg.debug:
                print(f"[RSI_DIV] 약세 다이버전스 체크: 점수={s:.3f}, 가격상승={price_rise:.4f}, RSI변화={rsi_delta:.3f}")
            
            # require rsi went down sufficiently (negative) - 조건 완화
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
                if self.cfg.debug:
                    print(f"[RSI_DIV] 약세 다이버전스 신호 생성 ✓ (점수: {score:.3f})")
            else:
                if self.cfg.debug:
                    print(f"[RSI_DIV] 약세 다이버전스 조건 미충족: RSI변화={rsi_delta:.3f}, 점수={s:.3f}")

        if self.cfg.debug and action == "HOLD":
            print(f"[RSI_DIV] 최종 결과: HOLD (극값 부족 또는 조건 미충족)")
        
        return {
            "name": "RSI_DIV",
            "action": action,
            "score": float(score),
            "timestamp": self.tm.get_current_time(),
            "context": context
        }
# --- end replacement ---
