# bollinger_squeeze_strategy_relaxed.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from data.data_manager import get_data_manager
from indicators.global_indicators import get_atr

@dataclass
class BBSqueezeCfg:
    # 더 예민하게 반응하도록 기본값 완화
    ma_period: int = 5                # shorter MA -> 더 빠르게 반응
    std_period: int = 5               # shorter STD
    std_dev: float = 1.4              # tighter bands (민감)
    squeeze_lookback: int = 10        # lookback 축소 (더 자주 체크)
    squeeze_threshold: float = 1.5    # threshold 증가 -> 더 자주 '스퀴즈' 인식
    breakout_lookback: int = 1        # 최근 1봉으로 빠르게 돌파 판단 (민감)
    tp_R1: float = 0.8                # 목표를 조금 보수적으로(짧은 R)
    tp_R2: float = 1.6
    stop_atr_mult: float = 0.8        # 스탑을 더 타이트하게 (ATR 기반)
    tick: float = 0.01
    # breakout 완화 파라미터
    min_body_ratio: float = 0.15      # 바디비율 기준을 낮춤 -> 작은 몸통도 허용
    allow_wick_break: bool = True     # 윅(꼬리) 중심 돌파 허용
    debug: bool = False               # 디버그 출력 옵션

class BollingerSqueezeStrategy:

    def __init__(self, cfg: BBSqueezeCfg = BBSqueezeCfg()):
        self.cfg = cfg
        self.is_squeezed = False
        self.last_signal_time = None

    def evaluate(self) -> Optional[Dict[str, Any]]:
        data_manager = get_data_manager()
        req_len = self.cfg.squeeze_lookback + max(self.cfg.ma_period, self.cfg.std_period) + 5
        df = data_manager.get_latest_data(req_len)

        if df is None or len(df) < max(self.cfg.ma_period, self.cfg.std_period, self.cfg.squeeze_lookback) + 2:
            if self.cfg.debug:
                print(f"🔍 [BB Squeeze] 데이터 부족: 필요한 데이터 길이={req_len}, 실제={len(df) if df is not None else 'None'}")
            return None

        last = df.iloc[-1]

        # Bollinger bands
        ma = df['close'].rolling(self.cfg.ma_period).mean()
        std = df['close'].rolling(self.cfg.std_period).std().fillna(0.0)
        upper_band = ma + (std * self.cfg.std_dev)
        lower_band = ma - (std * self.cfg.std_dev)
        # relative width in percent
        bb_width = (upper_band - lower_band) / ma * 100

        # squeeze detection (더 관대하게)
        bb_width_ma = bb_width.rolling(self.cfg.squeeze_lookback).mean().ffill()
        # when squeeze_threshold > 1.0, 더 자주 스퀴즈로 판정됨
        is_squeezed_now = bb_width.iloc[-1] < bb_width_ma.iloc[-1] * self.cfg.squeeze_threshold

        if is_squeezed_now:
            # enter squeezed state
            self.is_squeezed = True

        # when squeeze releases (width expands above threshold) -> check breakout
        if self.is_squeezed and not is_squeezed_now:
            self.is_squeezed = False

            long_breakout = last['close'] > upper_band.iloc[-1] or (self.cfg.allow_wick_break and last['high'] >= upper_band.iloc[-1])
            short_breakout = last['close'] < lower_band.iloc[-1] or (self.cfg.allow_wick_break and last['low'] <= lower_band.iloc[-1])

            # more permissive breakout strength check:
            #  - allow wick touches as valid breakout if allow_wick_break True
            #  - or require a minimum body ratio (smaller than before)
            highs = df['high'].rolling(self.cfg.breakout_lookback).max()
            lows = df['low'].rolling(self.cfg.breakout_lookback).min()

            body = abs(last['close'] - last['open'])
            rng = max(1e-9, float(last['high'] - last['low']))
            body_ratio = (body / rng) if rng > 0 else 0.0

            # permissive is_strong_breakout:
            # - primary: close past band OR wick touches band
            # - secondary: OR body_ratio >= min_body_ratio
            bull_touch = (last['close'] > upper_band.iloc[-1]) or (last['high'] >= upper_band.iloc[-1])
            bear_touch = (last['close'] < lower_band.iloc[-1]) or (last['low'] <= lower_band.iloc[-1])

            is_strong_breakout = False
            # bullish breakout conditions
            if bull_touch:
                # allow if wick break allowed OR body is decisive OR close above recent highs
                if self.cfg.allow_wick_break:
                    is_strong_breakout = True
                elif body_ratio >= self.cfg.min_body_ratio:
                    is_strong_breakout = True
                elif last['close'] >= highs.iloc[-1]:
                    is_strong_breakout = True
            # bearish breakout conditions
            if bear_touch and not is_strong_breakout:
                if self.cfg.allow_wick_break:
                    is_strong_breakout = True
                elif body_ratio >= self.cfg.min_body_ratio:
                    is_strong_breakout = True
                elif last['close'] <= lows.iloc[-1]:
                    is_strong_breakout = True

            if self.cfg.debug:
                print(f"[BB Squeeze DEBUG] bb_width={bb_width.iloc[-1]:.4f} bb_ma={bb_width_ma.iloc[-1]:.4f} "
                      f"is_squeezed_now={is_squeezed_now} long_breakout={long_breakout} short_breakout={short_breakout} "
                      f"body_ratio={body_ratio:.3f} is_strong_breakout={is_strong_breakout}")

            # BUY signal
            if long_breakout and is_strong_breakout:
                atr = get_atr()
                entry = last['close'] + self.cfg.tick
                stop = last['close'] - float(atr) * self.cfg.stop_atr_mult
                R = entry - stop
                tp1, tp2 = entry + self.cfg.tp_R1 * R, entry + self.cfg.tp_R2 * R
                return {
                    "stage": "ENTRY", "action": "BUY", "entry": float(entry), "stop": float(stop),
                    "targets": [float(tp1), float(tp2)],
                    "context": {
                        "mode": "BB_SQUEEZE", "bb_width": float(bb_width.iloc[-1]), "atr": float(atr),
                        "upper_band": float(upper_band.iloc[-1]), "lower_band": float(lower_band.iloc[-1]),
                        "body_ratio": float(body_ratio), "allow_wick_break": bool(self.cfg.allow_wick_break)
                    }
                }

            # SELL signal
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
                        "upper_band": float(upper_band.iloc[-1]), "lower_band": float(lower_band.iloc[-1]),
                        "body_ratio": float(body_ratio), "allow_wick_break": bool(self.cfg.allow_wick_break)
                    }
                }

        return None
