from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from managers.data_manager import get_data_manager
from utils.time_manager import get_time_manager

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a

@dataclass
class OrderflowCVDConfig:
    lookback_bars: int = 60
    z_scale: float = 3
    min_notional: float = 500

class OrderflowCVD:
    """간단한 체결 불균형 근사(CVD 스타일). 실제 체결측 데이터가 있으면 더 정밀하게 개선 가능."""

    def __init__(self, cfg: OrderflowCVDConfig = OrderflowCVDConfig()):
        self.cfg = cfg
        self.time_manager = get_time_manager()
    
    def _no_signal_result(self,**kwargs):
        return {
            'name': 'ORDERFLOW_CVD',
            'action': 'HOLD',
            'score': 0.0,
            'timestamp': self.time_manager.get_current_time(),
            'context': kwargs
        }

    def on_kline_close_3m(self) -> Optional[Dict[str, Any]]:
        dm = get_data_manager()
        if dm is None:
            return self._no_signal_result()

        df = None
        try:
            df = dm.get_latest_data(count=max(200, self.cfg.lookback_bars*2))
        except Exception:
            df = None
                

        if df is None or len(df) < self.cfg.lookback_bars + 10:
            return self._no_signal_result()

        try:
            close = pd.to_numeric(df['close'].astype(float))
            open_ = pd.to_numeric(df['open'].astype(float))
            vol = pd.to_numeric(df['volume'].astype(float)) if 'volume' in df.columns else pd.Series([1.0]*len(df))
        except Exception:
            return self._no_signal_result()

        imbalance = (close - open_) * vol
        recent_sum = imbalance.iloc[-self.cfg.lookback_bars:].sum()

        hist_sums = [imbalance.iloc[i:i+self.cfg.lookback_bars].sum() for i in range(0, max(1, len(imbalance)-self.cfg.lookback_bars))]
        if len(hist_sums) < 10:
            mu = float(np.mean(hist_sums)) if hist_sums else 0.0
            sd = float(np.std(hist_sums, ddof=1)) if len(hist_sums) > 1 else 1.0
        else:
            mu = float(np.mean(hist_sums[:-1])) if hist_sums[:-1] else 0.0
            sd = float(np.std(hist_sums[:-1], ddof=1)) if len(hist_sums[:-1])>1 else max(1.0, abs(mu)*0.01)

        z = (recent_sum - mu) / (sd if sd != 0 else 1.0)

        action = "HOLD"
        score = 0.0

        if abs(z) >= 0.8:
            action = 'BUY' if z > 0 else 'SELL'
            score = _clamp(abs(z) / self.cfg.z_scale, 0.0, 1.0)
        else:
            action = "HOLD"
            score = 0.0
        
        return {
            'name': 'ORDERFLOW_CVD',
            'action': action,
            'score': float(score),
            'timestamp': self.time_manager.get_current_time(),
            'context': {
                'recent_sum': float(recent_sum),
                'mu': float(mu),
                'sd': float(sd),
                'z': float(z),
                'lookback_bars': int(self.cfg.lookback_bars)
            }
        }
