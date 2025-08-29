# bollinger_squeeze_strategy.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from data.data_manager import get_data_manager
from indicators.global_indicators import get_atr

@dataclass
class BBSqueezeCfg:
    ma_period: int = 4                # 매우 짧게
    std_period: int = 4
    std_dev: float = 1.2              # 밴드 더 타이트하게
    squeeze_lookback: int = 12        # 짧게
    squeeze_threshold: float = 1.8    # 더 관대하게(현재 폭이 평균*1.8보다 작으면 squeeze)
    breakout_lookback: int = 1        # 최근 1봉 체크
    tp_R1: float = 0.8
    tp_R2: float = 1.4
    stop_atr_mult: float = 0.7
    tick: float = 0.01
    require_strong_body: float = 0.03 # 매우 작은 바디도 인정
    allow_wick_touch: bool = True
    debug: bool = True     

class BollingerSqueezeStrategy:

    def __init__(self, cfg: BBSqueezeCfg = BBSqueezeCfg()):
        self.cfg = cfg
        self.is_squeezed = False
        self.last_signal_time = None

    def _safe_get_atr(self, df: Optional[pd.DataFrame] = None) -> float:
        """
        Try to call get_atr(df) if supported, otherwise fallback to get_atr()
        and finally estimate from close-series if needed.
        """
        try:
            # some implementations accept df
            atr_val = get_atr(df) if df is not None else get_atr()
        except TypeError:
            # fallback: call without args
            try:
                atr_val = get_atr()
            except Exception:
                atr_val = None
        except Exception:
            atr_val = None

        if atr_val is None and df is not None and len(df) >= 2:
            # simple ATR-like estimate from last few bars
            h = df['high'].astype(float)
            l = df['low'].astype(float)
            c = df['close'].astype(float)
            prev_c = c.shift(1).fillna(c.iloc[0])
            tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
            atr_val = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
        # ensure numeric
        try:
            return float(atr_val or 0.0)
        except Exception:
            return 0.0

    def evaluate(self) -> Optional[Dict[str, Any]]:
        data_manager = get_data_manager()
        need_bars = self.cfg.squeeze_lookback + max(self.cfg.ma_period, self.cfg.std_period) + 5
        df = data_manager.get_latest_data(need_bars)

        if df is None or len(df) < max(self.cfg.ma_period, self.cfg.std_period, self.cfg.squeeze_lookback) + 2:
            return None

        last = df.iloc[-1]

        # compute Bollinger bands
        close = df['close'].astype(float)
        ma = close.rolling(self.cfg.ma_period).mean()
        std = close.rolling(self.cfg.std_period).std().fillna(0.0)
        upper_band = ma + (std * self.cfg.std_dev)
        lower_band = ma - (std * self.cfg.std_dev)

        # width in percent (avoid divide by zero)
        bb_width = ((upper_band - lower_band) / ma.replace(0, pd.NA)).fillna(0) * 100.0

        # squeeze detect (more permissive)
        bb_width_ma = bb_width.rolling(self.cfg.squeeze_lookback).mean().ffill()
        # if mean is NaN (very short history) fallback to last width
        base = bb_width_ma.iloc[-1] if not pd.isna(bb_width_ma.iloc[-1]) else bb_width.iloc[-1]
        is_squeezed = bb_width.iloc[-1] < (base * self.cfg.squeeze_threshold)

        if self.cfg.debug:
            print(f"[BB_SQUEEZE DEBUG] bb_width={bb_width.iloc[-1]:.3f} bb_width_ma={base:.3f} is_squeezed={is_squeezed}")

        # update squeeze state
        if is_squeezed:
            self.is_squeezed = True

        # when squeeze releases (previously squeezed but now not squeezed)
        if self.is_squeezed and not is_squeezed:
            # mark reset
            self.is_squeezed = False

            # breakout detection - more permissive:
            # - close outside band OR wick touched band (if allowed)
            # - also accept re-entry: prev close inside & last close outside
            prev = df.iloc[-2]
            highs = df['high'].rolling(self.cfg.breakout_lookback).max().ffill()
            lows  = df['low'].rolling(self.cfg.breakout_lookback).min().ffill()

            prev_close = float(prev['close'])
            last_close = float(last['close'])
            last_high = float(last['high'])
            last_low = float(last['low'])
            last_open = float(last['open'])

            upper = float(upper_band.iloc[-1])
            lower = float(lower_band.iloc[-1])

            # BUY breakout conditions (relaxed)
            long_breakout = (
                (last_close > upper) or
                (self.cfg.allow_wick_touch and last_high >= upper) or
                (prev_close <= upper and last_close > upper)  # reentry-like
            )

            # SHORT breakout conditions (relaxed)
            short_breakout = (
                (last_close < lower) or
                (self.cfg.allow_wick_touch and last_low <= lower) or
                (prev_close >= lower and last_close < lower)  # reentry-like
            )

            # strong breakout flag: either close is extreme vs recent highs/lows or body is strong
            highs_recent = float(highs.iloc[-1]) if not pd.isna(highs.iloc[-1]) else last_high
            lows_recent  = float(lows.iloc[-1])  if not pd.isna(lows.iloc[-1])  else last_low

            long_strong = ((last_close > highs_recent) or (last_high >= highs_recent) or (last_close > last_open and (abs(last_close-last_open) / max(1e-9, last_high-last_low)) >= self.cfg.require_strong_body))
            short_strong = ((last_close < lows_recent) or (last_low <= lows_recent) or (last_close < last_open and (abs(last_close-last_open) / max(1e-9, last_high-last_low)) >= self.cfg.require_strong_body))

            # accept if either breakout condition AND (strong OR permissive small-body allowance)
            accept_long = long_breakout and (long_strong or (abs(last_close - last_open) / max(1e-9, last_high-last_low) >= (self.cfg.require_strong_body * 0.6)))
            accept_short = short_breakout and (short_strong or (abs(last_close - last_open) / max(1e-9, last_high-last_low) >= (self.cfg.require_strong_body * 0.6)))

            if self.cfg.debug:
                print(f"[BB_SQUEEZE DEBUG] long_breakout={long_breakout} short_breakout={short_breakout} long_strong={long_strong} short_strong={short_strong} accept_long={accept_long} accept_short={accept_short}")

            atr = self._safe_get_atr(df)

            # finalize BUY signal
            if accept_long:
                entry = last_close + self.cfg.tick
                stop = last_close - float(atr) * self.cfg.stop_atr_mult if atr > 0 else last_close - 0.01
                R = entry - stop
                tp1, tp2 = entry + self.cfg.tp_R1 * R, entry + self.cfg.tp_R2 * R
                if self.cfg.debug:
                    print(f"[BB_SQUEEZE] BUY entry={entry:.4f} stop={stop:.4f} tp1={tp1:.4f} tp2={tp2:.4f} atr={atr:.4f}")
                return {
                    "stage": "ENTRY", "action": "BUY", "entry": float(entry), "stop": float(stop),
                    "targets": [float(tp1), float(tp2)],
                    "context": {
                        "mode": "BB_SQUEEZE", "bb_width": float(bb_width.iloc[-1]), "atr": float(atr),
                        "upper_band": float(upper), "lower_band": float(lower)
                    }
                }

            # finalize SELL signal
            if accept_short:
                entry = last_close - self.cfg.tick
                stop = last_close + float(atr) * self.cfg.stop_atr_mult if atr > 0 else last_close + 0.01
                R = stop - entry
                tp1, tp2 = entry - self.cfg.tp_R1 * R, entry - self.cfg.tp_R2 * R
                if self.cfg.debug:
                    print(f"[BB_SQUEEZE] SELL entry={entry:.4f} stop={stop:.4f} tp1={tp1:.4f} tp2={tp2:.4f} atr={atr:.4f}")
                return {
                    "stage": "ENTRY", "action": "SELL", "entry": float(entry), "stop": float(stop),
                    "targets": [float(tp1), float(tp2)],
                    "context": {
                        "mode": "BB_SQUEEZE", "bb_width": float(bb_width.iloc[-1]), "atr": float(atr),
                        "upper_band": float(upper), "lower_band": float(lower)
                    }
                }

        return None
