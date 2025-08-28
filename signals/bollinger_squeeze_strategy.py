
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
from data.data_manager import get_data_manager
from indicators.global_indicators import get_atr

@dataclass
class BBSqueezeCfg:
    """
    Bollinger Bands Squeeze Strategy Config.
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

    # ---- scoring helpers ----
    def _score_ratio(self, x: float, lo: float, hi: float) -> float:
        if hi == lo: return 0.0
        return max(0.0, min(1.0, (x - lo) / (hi - lo)))

    def _conf_bucket(self, v: float) -> str:
        if v >= 0.75: return "HIGH"
        if v >= 0.50: return "MEDIUM"
        return "LOW"

    def evaluate(self) -> Optional[Dict[str, Any]]:
        data_manager = get_data_manager()
        df = data_manager.get_latest_data(self.cfg.squeeze_lookback)

        if df is None or len(df) < max(self.cfg.ma_period, self.cfg.std_period, self.cfg.squeeze_lookback) + 2:
            return None

        last = df.iloc[-1]
        
        # 1. Bollinger Bands & width
        ma = df['close'].rolling(self.cfg.ma_period).mean()
        std = df['close'].rolling(self.cfg.std_period).std()
        upper_band = ma + (std * self.cfg.std_dev)
        lower_band = ma - (std * self.cfg.std_dev)
        bb_width = (upper_band - lower_band) / ma * 100

        # 2. Squeeze detection
        bb_width_ma = bb_width.rolling(self.cfg.squeeze_lookback).mean()
        is_squeezed = bb_width.iloc[-1] < bb_width_ma.iloc[-1] * self.cfg.squeeze_threshold
        
        if is_squeezed:
            self.is_squeezed = True
        
        # Squeeze release + strong breakout
        if self.is_squeezed and not is_squeezed:
            self.is_squeezed = False
            
            # 3. Breakout checks
            long_breakout = last['close'] > upper_band.iloc[-1]
            short_breakout = last['close'] < lower_band.iloc[-1]
            is_strong_breakout = (last['close'] > last['open'] and 
                                  last['high'] == last['high'].rolling(self.cfg.breakout_lookback).max().iloc[-1]
                                  ) or (
                                  last['close'] < last['open'] and 
                                  last['low'] == last['low'].rolling(self.cfg.breakout_lookback).min().iloc[-1]
                                  )

            if long_breakout and is_strong_breakout:
                atr = get_atr(df)
                atr = float(atr)
                entry = float(last['close']) + self.cfg.tick
                stop = float(last['close']) - atr * self.cfg.stop_atr_mult
                R = entry - stop
                tp1, tp2 = entry + self.cfg.tp_R1 * R, entry + self.cfg.tp_R2 * R

                # ---- scoring ----
                ratio = float(bb_width.iloc[-1] / max(1e-9, bb_width_ma.iloc[-1]))
                squeeze_score = 1.0 - self._score_ratio(ratio, 0.6, 1.0)  # tighter -> higher
                stdv = float(std.iloc[-1] or 1e-9)
                brk_units = abs(float(last['close']) - float(upper_band.iloc[-1])) / stdv
                break_score = self._score_ratio(brk_units, 0.0, 1.5)
                RR = abs(tp1 - entry) / max(1e-9, abs(entry - stop))
                rr_score = self._score_ratio(RR, 0.8, 2.0)

                score = max(0.0, min(1.0, 0.45*squeeze_score + 0.35*break_score + 0.20*rr_score))
                confidence = self._conf_bucket(score)
                reasons: List[str] = [
                    f"Squeeze ratio={ratio:.2f} (score {squeeze_score:.2f})",
                    f"Breakout units={brk_units:.2f}σ (score {break_score:.2f})",
                    f"R/R={RR:.2f} (score {rr_score:.2f})"
                ]

                print(f"🎯 [BB Squeeze] 신호: BUY | 진입=${entry:.4f} | 손절=${stop:.4f} | 목표=${tp1:.4f}, ${tp2:.4f} | 신뢰도={confidence:.0%} | 점수={score:.2f}")

                return {
                    "stage": "ENTRY", "action": "BUY", "entry": float(entry), "stop": float(stop),
                    "targets": [float(tp1), float(tp2)],
                    "context": {
                        "mode": "BB_SQUEEZE", "bb_width": float(bb_width.iloc[-1]), "atr": float(atr),
                        "upper_band": float(upper_band.iloc[-1]), "lower_band": float(lower_band.iloc[-1])
                    },
                    "score": float(score),
                    "confidence": confidence,
                    "reasons": reasons
                }
            
            if short_breakout and is_strong_breakout:
                atr = get_atr(df)
                atr = float(atr)
                entry = float(last['close']) - self.cfg.tick
                stop = float(last['close']) + atr * self.cfg.stop_atr_mult
                R = stop - entry
                tp1, tp2 = entry - self.cfg.tp_R1 * R, entry - self.cfg.tp_R2 * R

                # ---- scoring ----
                ratio = float(bb_width.iloc[-1] / max(1e-9, bb_width_ma.iloc[-1]))
                squeeze_score = 1.0 - self._score_ratio(ratio, 0.6, 1.0)
                stdv = float(std.iloc[-1] or 1e-9)
                brk_units = abs(float(last['close']) - float(lower_band.iloc[-1])) / stdv
                break_score = self._score_ratio(brk_units, 0.0, 1.5)
                RR = abs(tp1 - entry) / max(1e-9, abs(entry - stop))
                rr_score = self._score_ratio(RR, 0.8, 2.0)

                score = max(0.0, min(1.0, 0.45*squeeze_score + 0.35*break_score + 0.20*rr_score))
                confidence = self._conf_bucket(score)
                reasons: List[str] = [
                    f"Squeeze ratio={ratio:.2f} (score {squeeze_score:.2f})",
                    f"Breakout units={brk_units:.2f}σ (score {break_score:.2f})",
                    f"R/R={RR:.2f} (score {rr_score:.2f})"
                ]
                
                print(f"🎯 [BB Squeeze] 신호: SELL | 진입=${entry:.4f} | 손절=${stop:.4f} | 목표=${tp1:.4f}, ${tp2:.4f} | 신뢰도={confidence:.0%} | 점수={score:.2f}")

                return {
                    "stage": "ENTRY", "action": "SELL", "entry": float(entry), "stop": float(stop),
                    "targets": [float(tp1), float(tp2)],
                    "context": {
                        "mode": "BB_SQUEEZE", "bb_width": float(bb_width.iloc[-1]), "atr": float(atr),
                        "upper_band": float(upper_band.iloc[-1]), "lower_band": float(lower_band.iloc[-1])
                    },
                    "score": float(score),
                    "confidence": confidence,
                    "reasons": reasons
                }

        return None
