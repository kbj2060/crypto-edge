# bollinger_squeeze_strategy.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from data.data_manager import get_data_manager
from indicators.global_indicators import get_atr

@dataclass
class BBSqueezeCfg:
    ma_period: int = 8                # shorter MA -> 더 빠르게 반응
    std_period: int = 8               # shorter STD
    std_dev: float = 1.6              # tighter bands (민감)
    squeeze_lookback: int = 20        # lookback 축소 (더 자주 체크)
    squeeze_threshold: float = 1.0   # threshold 늘려 더 자주 '스퀴즈'로 인식
    breakout_lookback: int = 2        # 최근 2봉으로 강한 돌파 판단 (민감)
    tp_R1: float = 0.8                # 목표를 조금 보수적으로(짧은 R)
    tp_R2: float = 1.6
    stop_atr_mult: float = 0.8        # 스탑을 더 타이트하게 (ATR 기반)
    tick: float = 0.01

class BollingerSqueezeStrategy:

    def __init__(self, cfg: BBSqueezeCfg = BBSqueezeCfg()):
        self.cfg = cfg
        self.is_squeezed = False
        self.last_signal_time = None

    def evaluate(self) -> Optional[Dict[str, Any]]:
        data_manager = get_data_manager()
        df = data_manager.get_latest_data(self.cfg.squeeze_lookback + max(self.cfg.ma_period, self.cfg.std_period) + 5)

        if df is None or len(df) < max(self.cfg.ma_period, self.cfg.std_period, self.cfg.squeeze_lookback) + 2:
            print(f"🔍 [BB Squeeze] 데이터 부족: 필요한 데이터 길이={max(self.cfg.ma_period, self.cfg.std_period, self.cfg.squeeze_lookback) + 2}")
            return None

        last = df.iloc[-1]

        # Bollinger
        ma = df['close'].rolling(self.cfg.ma_period).mean()
        std = df['close'].rolling(self.cfg.std_period).std()
        upper_band = ma + (std * self.cfg.std_dev)
        lower_band = ma - (std * self.cfg.std_dev)
        bb_width = (upper_band - lower_band) / ma * 100

        # squeeze detect (more sensitive)
        bb_width_ma = bb_width.rolling(self.cfg.squeeze_lookback).mean().ffill()
        is_squeezed = bb_width.iloc[-1] < bb_width_ma.iloc[-1] * self.cfg.squeeze_threshold

        if is_squeezed:
            self.is_squeezed = True

        # when squeeze releases
        if self.is_squeezed and not is_squeezed:
            self.is_squeezed = False

            long_breakout = last['close'] > upper_band.iloc[-1]
            short_breakout = last['close'] < lower_band.iloc[-1]

            # strong breakout: recent close is extreme relative to breakout_lookback
            highs = df['high'].rolling(self.cfg.breakout_lookback).max()
            lows  = df['low'].rolling(self.cfg.breakout_lookback).min()
            is_strong_breakout = (last['close'] > last['open'] and last['close'] >= highs.iloc[-1]) or \
                                 (last['close'] < last['open'] and last['close'] <= lows.iloc[-1])

            if long_breakout and is_strong_breakout:
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
