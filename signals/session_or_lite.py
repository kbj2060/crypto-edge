
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import pandas as pd

from data.data_manager import DataManager, get_data_manager
from indicators.global_indicators import get_atr, get_opening_range, get_vwap
from utils.time_manager import get_time_manager

@dataclass
class SessionORLiteCfg:
    """
    Lightweight Opening Range strategy config.
    - or_minutes: minutes to build OR (session open -> lock)
    - valid_minutes_after_open: only trade within this window after session open
    - body_ratio_min: min candle body/range ratio for a breakout candle
    - retest_atr: ATR multiplier buffer around OR edge to validate retest
    - retest_atr_mult_short: extra buffer multiplier for SHORT side retest
    - atr_stop_mult: base stop sizing (used with 0.5xATR for OR anchor stop)
    - tp_R1 / tp_R2: targets in multiples of R
    - vwap_filter_mode: 'off' | 'location' | 'slope'
        - location: long c>=vwap, short c<=vwap
        - slope:   uses vwap_prev; long if vwap>=vwap_prev else short
    - allow_wick_break: allow wick-based breakout in addition to body close
    - wick_needs_body_sign: if wick breakout used, body must agree with direction
    """
    or_minutes: int = 30
    body_ratio_min: float = 0.10

    retest_atr: float = 0.50
    retest_atr_mult_short: float = 2.0  # SHORT only buffer multiplier

    atr_stop_mult: float = 1.5
    tp_R1: float = 1.2
    tp_R2: float = 2.0
    tick: float = 0.1

    vwap_filter_mode: str = "off"  # 'off' | 'location' | 'slope'
    allow_wick_break: bool = True
    wick_needs_body_sign: bool = False


class SessionORLite:
    """Simplified Opening Range breakout→retest strategy."""

    def __init__(self, cfg: SessionORLiteCfg = SessionORLiteCfg()):
        self.cfg = cfg
        self.session_open: Optional[datetime] = None
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None
        self.time_manager = get_time_manager()
        self.data_manager = get_data_manager()

        # Simple debug counters to diagnose side bias
        self.debug = {
            "break_long": 0, "break_short": 0,
            "retest_long_miss": 0, "retest_short_miss": 0,
            "vwap_long_block": 0, "vwap_short_block": 0
        }

    # ---- main hook (3m close) ----
    def on_kline_close_3m(
        self,
        df3: pd.DataFrame,
        vwap_prev: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate on 3m candle close.
        df3: pandas DataFrame with columns open, high, low, close (3m)
        vwap, vwap_std, atr: session-anchored preferred (floats)
        vwap_prev: previous value for slope filtering (optional)
        returns: signal dict or None
        """
        now = self.time_manager.get_current_time()

        if not self.session_open:
            return None
        
        if df3 is None or len(df3) < 2:
            return None
        
        self.or_high, self.or_low = get_opening_range()
        vwap, vwap_std = get_vwap()
        atr = get_atr()
        
        last = df3.iloc[-1]
        prev = df3.iloc[-2]
        o = float(last["open"]); h = float(last["high"]); l = float(last["low"]); c = float(last["close"])
        ph = float(prev["high"]); pl = float(prev["low"])

        # 1) Build/lock OR (include the last candle if now == or_end)
        or_end = self.session_open + timedelta(minutes=self.cfg.or_minutes)


        # safety
        if self.or_high is None or self.or_low is None or self.or_high <= self.or_low:
            print('High, Low 데이터 없음')
            return None

        # 2) Breakout qualification (body or wick-based, configurable)
        rng = h - l
        if rng <= 0:
            return None
            
        body = abs(c - o)
        body_ok = (body / rng) >= self.cfg.body_ratio_min

        # wick based breakout allowance
        wick_break_long  = (h >= self.or_high + self.cfg.tick)
        wick_break_short = (l <= self.or_low  - self.cfg.tick)

        wick_body_ok_long  = (c > o) if self.cfg.wick_needs_body_sign else True
        wick_body_ok_short = (c < o) if self.cfg.wick_needs_body_sign else True

        break_long_ok = (body_ok and (c >= self.or_high + self.cfg.tick)) or (self.cfg.allow_wick_break and wick_break_long and wick_body_ok_long)
        break_short_ok = (body_ok and (c <= self.or_low  - self.cfg.tick)) or (self.cfg.allow_wick_break and wick_break_short and wick_body_ok_short)

        if break_long_ok:
            self.debug["break_long"] += 1
        if break_short_ok:
            self.debug["break_short"] += 1

        # 3) Retest near the OR edge (allow previous candle to count)
        buf_long  = self.cfg.retest_atr * float(atr)
        buf_short = self.cfg.retest_atr * self.cfg.retest_atr_mult_short * float(atr)

        # use min low for long (deeper touch), max high for short (shallower touch)
        min_low = min(l, pl)
        max_high = max(h, ph)
        
        touched_long  = (min_low >= self.or_high - buf_long) and (min_low <= self.or_high + buf_long)
        touched_short = (max_high <= self.or_low + buf_short) and (max_high >= self.or_low - buf_short)

        if not touched_long:
            self.debug["retest_long_miss"] += 1
        if not touched_short:
            self.debug["retest_short_miss"] += 1

        # 4) VWAP filter
        vwap_ok_long = vwap_ok_short = True
        mode = (self.cfg.vwap_filter_mode or "off").lower()
        
        if mode == "location":
            vwap_ok_long  = c >= float(vwap)
            vwap_ok_short = c <= float(vwap)
        elif mode == "slope" and vwap_prev is not None:
            slope_up = float(vwap) >= float(vwap_prev)
            vwap_ok_long, vwap_ok_short = slope_up, (not slope_up)

        if not vwap_ok_long:
            self.debug["vwap_long_block"] += 1
        if not vwap_ok_short:
            self.debug["vwap_short_block"] += 1

        # 5) Signals (one per side per session)
        sigs = []
        
        if break_long_ok and touched_long and vwap_ok_long:
            entry = h + self.cfg.tick
            stop  = min(l, self.or_high - self.cfg.atr_stop_mult * float(atr)) - self.cfg.tick
            R = entry - stop
            tp1, tp2 = entry + self.cfg.tp_R1 * R, entry + self.cfg.tp_R2 * R
            
            sigs.append({
                "stage": "ENTRY", "action": "BUY", "entry": float(entry), "stop": float(stop),
                "targets": [float(tp1), float(tp2)],
                "context": {
                    "mode": "SESSION_OR_LITE", "or_high": float(self.or_high),
                    "atr": float(atr), "vwap": float(vwap), "vwap_std": float(vwap_std),
                    "touched_buf": float(buf_long), "body_ok": body_ok, "wick_break": wick_break_long
                }
            })
        if break_short_ok and touched_short and vwap_ok_short:
            entry = l - self.cfg.tick
            stop  = max(h, self.or_low + self.cfg.atr_stop_mult * float(atr)) + self.cfg.tick
            R = stop - entry
            tp1, tp2 = entry - self.cfg.tp_R1 * R, entry - self.cfg.tp_R2 * R
            
            sigs.append({
                "stage": "ENTRY", "action": "SELL", "entry": float(entry), "stop": float(stop),
                "targets": [float(tp1), float(tp2)],
                "context": {
                    "mode": "SESSION_OR_LITE", "or_low": float(self.or_low),
                    "atr": float(atr), "vwap": float(vwap), "vwap_std": float(vwap_std),
                    "touched_buf": float(buf_short), "body_ok": body_ok, "wick_break": wick_break_short
                }
            })

        return sigs[0] if sigs else None
