from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from data.data_manager import get_data_manager
from indicators.global_indicators import get_atr


@dataclass
class BBSqueezeCfg:
    """
    Bollinger Bands Squeeze Strategy Config.
    - ma_period: Moving Average period for BB (e.g., 20)
    - ma_type: 'SMA' or 'EMA'
    - std_period: Standard Deviation period for BB (e.g., 20)
    - std_dev: Standard Deviation multiplier for BB (e.g., 2.0)
    - squeeze_lookback: Period to calculate BB width moving average for squeeze condition.
    - squeeze_threshold: Squeeze is active when BB width is below this threshold (e.g., 0.9).
    - breakout_lookback: Period to check for a strong breakout candle.
    - tp_R1 / tp_R2: Targets in multiples of R (Risk).
    - stop_atr_mult: ATR multiplier for stop loss.
    - tick: Price tick size.
    """
    ma_period: int = 20
    std_period: int = 20
    std_dev: float = 2.0
    squeeze_lookback: int = 100
    squeeze_threshold: float = 0.95
    breakout_lookback: int = 5
    tp_R1: float = 1.0
    tp_R2: float = 2.0
    stop_atr_mult: float = 1.5
    tick: float = 0.01

class BollingerSqueezeStrategy:

    def __init__(self, cfg: BBSqueezeCfg):
        self.cfg = cfg
        self.is_squeezed = False
        self.last_signal_time = None

    def evaluate(self) -> Optional[Dict[str, Any]]:
        data_manager = get_data_manager()
        df = data_manager.get_latest_data(self.cfg.squeeze_lookback)


        if len(df) < max(self.cfg.ma_period, self.cfg.std_period, self.cfg.squeeze_lookback) + 2:
            return None

        last = df.iloc[-1]
        
        # 1. 볼린저 밴드 및 밴드 너비 계산
        ma = df['close'].rolling(self.cfg.ma_period).mean()
        std = df['close'].rolling(self.cfg.std_period).std()
        upper_band = ma + (std * self.cfg.std_dev)
        lower_band = ma - (std * self.cfg.std_dev)
        bb_width = (upper_band - lower_band) / ma * 100

        # 2. 스퀴즈 상태 감지
        bb_width_ma = bb_width.rolling(self.cfg.squeeze_lookback).mean()
        is_squeezed = bb_width.iloc[-1] < bb_width_ma.iloc[-1] * self.cfg.squeeze_threshold
        
        if is_squeezed:
            self.is_squeezed = True
        
        # 스퀴즈 상태가 해제되었고, 마지막 캔들이 강한 돌파를 보였는지 확인
        if self.is_squeezed and not is_squeezed:
            self.is_squeezed = False
            
            # 3. 돌파 조건 확인
            # 상단 밴드 돌파
            long_breakout = last['close'] > upper_band.iloc[-1]
            # 하단 밴드 돌파
            short_breakout = last['close'] < lower_band.iloc[-1]
            
            # 마지막 돌파 캔들의 2~3개 봉 전까지 고점/저점과 비교하여 '강한 돌파'인지 확인
            is_strong_breakout = (last['close'] > last['open'] and 
                                  last['high'] == last['high'].rolling(self.cfg.breakout_lookback).max().iloc[-1]
                                  ) or (
                                  last['close'] < last['open'] and 
                                  last['low'] == last['low'].rolling(self.cfg.breakout_lookback).min().iloc[-1]
                                  )

            if long_breakout and is_strong_breakout:
                # 롱 포지션 신호 생성
                atr = get_atr(df)
                entry = last['close'] + self.cfg.tick
                stop = last['close'] - float(atr) * self.cfg.stop_atr_mult
                R = entry - stop
                tp1, tp2 = entry + self.cfg.tp_R1 * R, entry + self.cfg.tp_R2 * R

                return {
                    "stage": "ENTRY", "action": "BUY", "entry": float(entry), "stop": float(stop),
                    "targets": [float(tp1), float(tp2)],
                    "context": {
                        "mode": "BB_SQUEEZE", "bb_width": float(bb_width.iloc[-1]), "atr": float(atr),
                        "upper_band": float(upper_band.iloc[-1]), "lower_band": float(lower_band.iloc[-1])
                    }
                }
            
            if short_breakout and is_strong_breakout:
                # 숏 포지션 신호 생성
                atr = get_atr(df)
                entry = last['close'] - self.cfg.tick
                stop = last['close'] + float(atr) * self.cfg.stop_atr_mult
                R = stop - entry
                tp1, tp2 = entry - self.cfg.tp_R1 * R, entry - self.cfg.tp_R2 * R
                
                return {
                    "stage": "ENTRY", "action": "SELL", "entry": float(entry), "stop": float(stop),
                    "targets": [float(tp1), float(tp2)],
                    "context": {
                        "mode": "BB_SQUEEZE", "bb_width": float(bb_width.iloc[-1]), "atr": float(atr),
                        "upper_band": float(upper_band.iloc[-1]), "lower_band": float(lower_band.iloc[-1])
                    }
                }

        return None