from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import pandas as pd

from data.binance_dataloader import BinanceDataLoader
from data.data_manager import get_data_manager
from indicators.global_indicators import get_opening_range

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a


@dataclass
class VolSpikeConfig:
    lookback: int = 20
    vol_threshold: float = 1.6   # last_vol / vol_ma >= threshold
    vol_ratio_min: float = 1.2
    use_quote_volume: bool = True
    price_break_margin: float = 0.0  # optional margin above OR high/low

class VolSpike3m:
    """3분봉에서 볼륨 스파이크 + 가격 돌파를 확인해 브레이크아웃 신호를 낸다."""

    def __init__(self, cfg: VolSpikeConfig = VolSpikeConfig()):
        self.cfg = cfg

    def on_kline_close_3m(self) -> Optional[Dict[str, Any]]:
        dm = get_data_manager()
        if dm is None:
            return None

        df = dm.get_latest_data(count=self.cfg.lookback + 2)
        

        if df is None or len(df) < self.cfg.lookback + 2:
            return None

        # compute volume series
        v_series = pd.to_numeric(df['quote_volume'].astype(float))
        

        vol_ma = v_series.rolling(self.cfg.lookback, min_periods=1).mean().iloc[-2]
        last_vol = float(v_series.iloc[-1])
        vol_ratio = (last_vol / (vol_ma if vol_ma > 0 else 1.0))

        # price breakout check - try to use opening range if available
        price_break_up = False
        price_break_down = False
        try:
            or_high, or_low = get_opening_range()
            last_close = float(df['close'].iloc[-1])
            if or_high is not None:
                price_break_up = last_close >= (or_high + self.cfg.price_break_margin)
                price_break_down = last_close <= (or_low - self.cfg.price_break_margin)
        except Exception:
            recent_high = pd.to_numeric(df['high'].astype(float)).iloc[-2:-1].max()
            recent_low = pd.to_numeric(df['low'].astype(float)).iloc[-2:-1].min()
            last_close = float(df['close'].iloc[-1])
            price_break_up = last_close > recent_high
            price_break_down = last_close < recent_low

        score = 0.0
        action = None
        conf = 'LOW'

        if vol_ratio >= self.cfg.vol_threshold:
            base = _clamp((vol_ratio - 1.0) / (self.cfg.vol_threshold - 1.0), 0.0, 1.0)
            if price_break_up:
                action = 'BUY'
                score = _clamp(base, 0.0, 1.0)
            elif price_break_down:
                action = 'SELL'
                score = _clamp(base, 0.0, 1.0)
            else:
                action = None
                score = _clamp(base * 0.5, 0.0, 1.0)
        else:
            if vol_ratio >= self.cfg.vol_ratio_min and (price_break_up or price_break_down):
                base = _clamp((vol_ratio - 1.0) / (self.cfg.vol_threshold - 1.0), 0.0, 1.0)
                action = 'BUY' if price_break_up else 'SELL'
                score = _clamp(base * 0.6, 0.0, 1.0)

        if score >= 0.8:
            conf = 'HIGH'
        elif score >= 0.5:
            conf = 'MEDIUM'
        else:
            conf = 'LOW'
        
        if action is None:
            return None

        return {
            'name': 'VOL_SPIKE_3m',
            'action': action,
            'score': float(score),
            'confidence': conf,
            'timestamp': datetime.utcnow(),
            'context': {
                'vol_ratio': float(vol_ratio),
                'vol_ma': float(vol_ma) if vol_ma is not None else None,
                'last_vol': float(last_vol),
                'price_break_up': bool(price_break_up),
                'price_break_down': bool(price_break_down)
            }
        }