from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime
from data.data_manager import get_data_manager
from indicators.global_indicators import get_atr
from utils.time_manager import get_time_manager

def _clamp(x, a=0.0, b=1.0):
    try:
        return max(a, min(b, float(x)))
    except Exception:
        return a

@dataclass
class BBSqueezeCfg:
    # Aggressive / high-leverage tuning (10x-20x). More sensitive => earlier entries, tighter stops.
    ma_period: int = 20                # short MA for responsiveness (lower => faster signaling)
    std_period: int = 20               # STD period aligned with MA
    std_dev: float = 1.8               # band width multiplier (1.8 is a good balance for tight but not noise-driven)
    squeeze_lookback: int = 60        # lookback for bb width mean (e.g. ~6 hours on 3m bars)
    squeeze_threshold: float = 0.08    # normalized strength threshold (higher => fewer false squeezes)
    breakout_lookback: int = 2         # recent bars for breakout highs/lows (allow small confirmation)
    tp_R1: float = 1.0                 # first target in R multiples (scalping-friendly)
    tp_R2: float = 2.0                 # second target in R multiples (optional extended target)
    stop_atr_mult: float = 1.5         # ATR multiplier for stop placement (avoid too tight stops)
    tick: float = 0.01
    min_body_ratio: float = 0.08      # require a slightly larger candle body to confirm breakout
    allow_wick_break: bool = False    # avoid wick-only triggers (require real body breakout)
    debug: bool = False

    # score composition weights (squeeze_strength, momentum, volume)
    w_squeeze: float = 0.40
    w_momentum: float = 0.45
    w_volume: float = 0.15

    # minimal passive contribution when squeezed but no full breakout
    passive_score_mul: float = 0.20
    # require a more meaningful immediate strength to fire without long squeeze
    immediate_fire_min_strength: float = 0.04

class BollingerSqueezeStrategy:
    def __init__(self, cfg: BBSqueezeCfg = BBSqueezeCfg()):
        self.cfg = cfg
        self.is_squeezed = False
        self.last_squeeze_time = None
        self.last_signal_time = None
        self.time_manager = get_time_manager()

    def evaluate(self) -> Optional[Dict[str, Any]]:
        data_manager = get_data_manager()
        req_len = self.cfg.squeeze_lookback + max(self.cfg.ma_period, self.cfg.std_period) + 5
        df = data_manager.get_latest_data(req_len)

        if df is None or len(df) < max(self.cfg.ma_period, self.cfg.std_period, self.cfg.squeeze_lookback) + 2:
            if self.cfg.debug:
                print(f"🔍 [BB Squeeze Agg] 데이터 부족: 필요한 데이터 길이={req_len}, 실제={len(df) if df is not None else 'None'}")
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Bollinger bands
        ma = df['close'].rolling(self.cfg.ma_period).mean().ffill()
        std = df['close'].rolling(self.cfg.std_period).std().fillna(0.0)
        upper_band = ma + (std * self.cfg.std_dev)
        lower_band = ma - (std * self.cfg.std_dev)

        eps = 1e-9
        bb_width = ((upper_band - lower_band) / (ma.replace(0, eps))).fillna(0.0)

        # squeeze strength (normalized): how much narrower current width is vs recent mean
        bb_width_ma = bb_width.rolling(self.cfg.squeeze_lookback, min_periods=1).mean().ffill().fillna(0.0)
        cur_w = float(bb_width.iloc[-1])
        ma_w = float(bb_width_ma.iloc[-1]) if bb_width_ma.iloc[-1] is not None else 0.0
        squeeze_strength = 0.0
        if ma_w > 0:
            squeeze_strength = max(0.0, (ma_w - cur_w) / (ma_w + eps))  # 0..1
        else:
            squeeze_strength = 0.0

        # detect squeezed now
        is_squeezed_now = squeeze_strength >= self.cfg.squeeze_threshold

        # track last squeeze time for permissive immediate breakouts
        now = self.time_manager.get_current_time()
        if is_squeezed_now:
            self.is_squeezed = True
            self.last_squeeze_time = now

        recent_squeeze = False
        # permissive recent squeeze: if we've seen squeeze in this run recently, treat as recent
        if getattr(self, 'last_squeeze_time', None) is not None:
            recent_squeeze = True

        # breakout detection (allow wick touches)
        last_close = float(last['close'])
        last_high = float(last['high'])
        last_low = float(last['low'])

        upper_now = float(upper_band.iloc[-1])
        lower_now = float(lower_band.iloc[-1])

        long_breakout = (last_close > upper_now) or (self.cfg.allow_wick_break and last_high >= upper_now)
        short_breakout = (last_close < lower_now) or (self.cfg.allow_wick_break and last_low <= lower_now)

        # body ratio
        body = abs(last['close'] - last['open'])
        rng = max(1e-9, float(last['high'] - last['low']))
        body_ratio = (body / rng) if rng > 0 else 0.0

        highs = df['high'].rolling(self.cfg.breakout_lookback, min_periods=1).max().ffill()
        lows = df['low'].rolling(self.cfg.breakout_lookback, min_periods=1).min().ffill()

        bull_touch = (last_close > upper_now) or (last_high >= upper_now)
        bear_touch = (last_close < lower_now) or (last_low <= lower_now)

        is_strong_breakout = False
        if bull_touch:
            if self.cfg.allow_wick_break:
                is_strong_breakout = True
            elif body_ratio >= self.cfg.min_body_ratio:
                is_strong_breakout = True
            elif last_close >= float(highs.iloc[-1]):
                is_strong_breakout = True
        if bear_touch and not is_strong_breakout:
            if self.cfg.allow_wick_break:
                is_strong_breakout = True
            elif body_ratio >= self.cfg.min_body_ratio:
                is_strong_breakout = True
            elif last_close <= float(lows.iloc[-1]):
                is_strong_breakout = True

        # volume component (optional): compare last volume to recent MA (quote_volume preferred)
        vol_comp = 0.0
        try:
            if 'quote_volume' in df.columns:
                v_series = pd.to_numeric(df['quote_volume'].astype(float))
            elif 'volume' in df.columns:
                v_series = pd.to_numeric(df['volume'].astype(float) * df['close'].astype(float))
            else:
                v_series = None
            if v_series is not None:
                vol_ma = v_series.rolling(self.cfg.squeeze_lookback, min_periods=1).mean().iloc[-2]
                last_vol = float(v_series.iloc[-1])
                vol_ratio = last_vol / (vol_ma if vol_ma and vol_ma>0 else 1.0)
                # map vol_ratio to 0..1 with soft cap at 3x
                vol_comp = _clamp((vol_ratio - 1.0) / 2.0, 0.0, 1.0)
            else:
                vol_comp = 0.0
        except Exception:
            vol_comp = 0.0

        # momentum component: recent close move normalized to 0..1 (0.5% -> 1.0)
        prev_close = float(prev['close'])
        momentum = 0.0
        if prev_close != 0:
            momentum = (last_close - prev_close) / prev_close
        mom_norm = _clamp(abs(momentum) / 0.005, 0.0, 1.0)

        # determine allowed_to_fire: aggressive rules (RELAXED)
        allowed_to_fire = False
        fire_reasons = []
        # thresholds for non-squeeze breakouts (tunable)
        
        body_fire_thresh = max(self.cfg.min_body_ratio, 0.12)   # 몸통 기준: 0.12 (권장 범위 0.10~0.18)
        mom_fire_thresh  = 0.12                                # 약 0.6% 움직임 기준 (0.10~0.18 범위)
        vol_fire_thresh  = 0.06                                  # 약 1.16x 볼륨 기준 (0.06~0.14 범위)

        if is_strong_breakout:
            # prefer squeeze-based firing first
            if self.is_squeezed or squeeze_strength >= self.cfg.immediate_fire_min_strength or recent_squeeze:
                allowed_to_fire = True
                fire_reasons.append('squeeze_ok')
            # if not squeeze-validated, allow non-squeeze strong breakout based on body/momentum/volume
            if not allowed_to_fire:
                if body_ratio >= body_fire_thresh:
                    allowed_to_fire = True
                    fire_reasons.append(f'body_ratio:{body_ratio:.3f}')
                elif mom_norm >= mom_fire_thresh:
                    allowed_to_fire = True
                    fire_reasons.append(f'momentum:{mom_norm:.3f}')
                elif vol_comp >= vol_fire_thresh:
                    allowed_to_fire = True
                    fire_reasons.append(f'vol:{vol_comp:.3f}')
        else:
            # also allow small immediate 'touch and go' when squeeze very strong and wick break occurs
            if bull_touch or bear_touch:
                if squeeze_strength >= max(self.cfg.squeeze_threshold, 0.12):
                    allowed_to_fire = True
                    fire_reasons.append('touch_and_go')
        # debug reasons if not allowed
        if self.cfg.debug and not allowed_to_fire:
            print(f"[BB Agg DEBUG] allowed_to_fire=False reasons_missing: squeeze={self.is_squeezed}, squeeze_strength={squeeze_strength:.3f}, "
                f"body_ratio={body_ratio:.3f} (need>={body_fire_thresh}), mom_norm={mom_norm:.3f} (need>={mom_fire_thresh}), vol_comp={vol_comp:.3f} (need>={vol_fire_thresh})")
        elif self.cfg.debug and allowed_to_fire:
            print(f"[BB Agg DEBUG] allowed_to_fire=True fire_reasons={fire_reasons}")

        # score composition: squeeze_strength(50%) + momentum(30%) + volume(20%)
        score = 0.0
        if allowed_to_fire:
            score = _clamp(self.cfg.w_squeeze * squeeze_strength + self.cfg.w_momentum * mom_norm + self.cfg.w_volume * vol_comp, 0.0, 1.0)
        else:
            # passive hint when squeezed but no full breakout
            if is_squeezed_now:
                score = _clamp(self.cfg.passive_score_mul * (self.cfg.w_squeeze * squeeze_strength + self.cfg.w_momentum * mom_norm + self.cfg.w_volume * vol_comp), 0.0, 1.0)
            else:
                score = 0.0

        # map to confidence
        if score >= 0.85:
            conf = 'HIGH'
        elif score >= 0.5:
            conf = 'MEDIUM'
        elif score > 0.0:
            conf = 'LOW'
        else:
            conf = 'LOW'

        if self.cfg.debug:
            print(f"[BB Agg DEBUG] cur_w={cur_w:.6f} ma_w={ma_w:.6f} squeeze_strength={squeeze_strength:.3f} "
                f"is_squeezed_now={is_squeezed_now} long_breakout={long_breakout} short_breakout={short_breakout} "
                f"body_ratio={body_ratio:.3f} is_strong_breakout={is_strong_breakout} allowed_to_fire={allowed_to_fire} "
                f"mom={momentum:.4f} mom_norm={mom_norm:.3f} vol_comp={vol_comp:.3f} score={score:.3f} conf={conf}")

        # generate entry if allowed and breakout direction exists
        if allowed_to_fire and is_strong_breakout:
            atr = get_atr()
            if long_breakout:
                entry = last_close + self.cfg.tick
                stop = last_close - float(atr) * self.cfg.stop_atr_mult
                R = entry - stop if entry > stop else float(atr) * self.cfg.stop_atr_mult
                tp1, tp2 = entry + self.cfg.tp_R1 * R, entry + self.cfg.tp_R2 * R
                self.last_signal_time = now
                return {
                    "stage": "ENTRY", "action": "BUY", "entry": float(entry), "stop": float(stop),
                    "targets": [float(tp1), float(tp2)], "score": float(score), "confidence": conf,
                    "context": {
                        "mode": "BB_SQUEEZE_AGG", "bb_width": float(cur_w), "bb_width_ma": float(ma_w),
                        "squeeze_strength": float(squeeze_strength), "atr": float(atr),
                        "upper_band": float(upper_now), "lower_band": float(lower_now),
                        "body_ratio": float(body_ratio), "vol_comp": float(vol_comp)
                    }
                }
            if short_breakout:
                atr = get_atr()
                entry = last_close - self.cfg.tick
                stop = last_close + float(atr) * self.cfg.stop_atr_mult
                R = stop - entry if stop > entry else float(atr) * self.cfg.stop_atr_mult
                tp1, tp2 = entry - self.cfg.tp_R1 * R, entry - self.cfg.tp_R2 * R
                self.last_signal_time = now
                return {
                    "stage": "ENTRY", "action": "SELL", "entry": float(entry), "stop": float(stop),
                    "targets": [float(tp1), float(tp2)], "score": float(score), "confidence": conf,
                    "context": {
                        "mode": "BB_SQUEEZE_AGG", "bb_width": float(cur_w), "bb_width_ma": float(ma_w),
                        "squeeze_strength": float(squeeze_strength), "atr": float(atr),
                        "upper_band": float(upper_now), "lower_band": float(lower_now),
                        "body_ratio": float(body_ratio), "vol_comp": float(vol_comp)
                    }
                }
        
        return {
                    "stage": "NONE", "action": "HOLD", "entry": None, "stop": None,
                    "targets": [None, None], "score": float(score), "confidence": conf,
                    "context": {}
                }
