from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd

from managers.binance_dataloader import BinanceDataLoader
from utils.time_manager import get_time_manager

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a
    
@dataclass
class EMATrendConfig:
    ema_short: int = 50
    ema_long: int = 200
    score_scale: float = 0.002   # 0.5% diff -> score ~=1
    min_bars: int = 60
    age_limit_min: int = 60

class EMATrend15m:
    """15분 EMA 트렌드를 계산해 HTF 컨펌용 신호를 출력합니다.
    반환 포맷:
        {'name':'EMA_TREND_15m', 'action':'BUY'|'SELL'|None, 'score':0..1,
        'timestamp': datetime, 'context': {...}}
    """

    def __init__(self, cfg: EMATrendConfig = EMATrendConfig()):
        self.cfg = cfg
        self.data_loader = BinanceDataLoader()
        self.time_manager = get_time_manager()

    def _get_last_15min_boundary(self, current_time: datetime) -> datetime:
        """현재 시간에서 가장 가까운 과거의 15분 경계 시간을 반환합니다.
        
        예시:
        - 현재 시간이 14:23 → 14:15 반환
        - 현재 시간이 14:37 → 14:30 반환
        - 현재 시간이 14:44 → 14:30 반환
        """
        # 현재 분을 15로 나눈 몫에 15를 곱하여 15분 경계 시간 계산
        boundary_minute = (current_time.minute // 15) * 15
        return current_time.replace(minute=boundary_minute, second=0, microsecond=0)
    
    def on_kline_close_15m(self) -> Optional[Dict[str, Any]]:
        now = self.time_manager.get_current_time()
        # 15분 경계 시간 계산
        last_15min_boundary = self._get_last_15min_boundary(now)
        
        # 데이터 시작 시간을 15분 경계 기준으로 설정
        start_time = last_15min_boundary - timedelta(minutes=15*self.cfg.min_bars)
        
        df = self.data_loader.fetch_data(interval="15m", symbol='ETHUSDT', start_time=start_time, end_time=last_15min_boundary)

        if df is None or len(df) < self.cfg.ema_long + 2:
            return None

        close = pd.to_numeric(df['close'].astype(float))

        ema_s = close.ewm(span=self.cfg.ema_short, adjust=False).mean().iloc[-1]
        ema_l = close.ewm(span=self.cfg.ema_long, adjust=False).mean().iloc[-1]
        diff = ema_s - ema_l
        diff_pct = diff / (ema_l if ema_l != 0 else 1.0)

        score = 0.0
        action = "HOLD"

        thresh = self.cfg.score_scale
        if diff_pct >= thresh:
            action = 'BUY'
            score = _clamp(abs(diff_pct) / thresh, 0.0, 1.0)
        elif diff_pct <= -thresh:
            action = 'SELL'
            score = _clamp(abs(diff_pct) / thresh, 0.0, 1.0)
        else:
            action = None
            score = 0.0
        
        return {
            'name': 'EMA_TREND_15m',
            'action': action,
            'score': float(score),
            'timestamp': datetime.utcnow(),
            'context': {
                'ema_short': float(ema_s),
                'ema_long': float(ema_l),
                'diff': float(diff),
                'diff_pct': float(diff_pct)
            }
        }