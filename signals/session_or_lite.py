# session_or_lite.py
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import pandas as pd

from data.data_manager import get_data_manager
from utils.time_manager import get_time_manager
from indicators.global_indicators import get_atr, get_opening_range, get_vwap

@dataclass
class SessionORLiteCfg:
    or_minutes: int = 30
    body_ratio_min: float = 0.03    # more permissive: smaller body accepted
    retest_atr: float = 0.45        # larger retest buffer (more permissive)
    retest_atr_mult_short: float = 1.2
    atr_stop_mult: float = 1.0
    tp_R1: float = 1.0
    tp_R2: float = 1.6
    tick: float = 0.03
    vwap_filter_mode: str = "off"   # default off
    allow_wick_break: bool = True
    wick_needs_body_sign: bool = False
    # extra permissive flags
    allow_either_touched_or_wick: bool = True
    low_conf_trade_scale: float = 0.35
    debug_print: bool = False

class SessionORLite:
    def __init__(self, cfg: SessionORLiteCfg = SessionORLiteCfg()):
        self.cfg = cfg
        self.session_open: Optional[datetime] = None
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None
        self.time_manager = get_time_manager()
        self.data_manager = get_data_manager()
        # counters for debugging / telemetry
        self.debug = {
            "break_long": 0, "break_short": 0,
            "retest_long_miss": 0, "retest_short_miss": 0,
            "vwap_long_block": 0, "vwap_short_block": 0,
            "low_conf_signals": 0
        }
    
    def on_kline_close_3m(self, df3: pd.DataFrame, session_activated: bool, vwap_prev: Optional[float] = None) -> Optional[Dict[str, Any]]:
        now = self.time_manager.get_current_time()

        if session_activated:
            self.session_open = self.time_manager.get_current_session_info(now).open_time

        if df3 is None or len(df3) < 2:
            print("df3 is None or len(df3) < 2")
            return None

        # opening range + indicators
        self.or_high, self.or_low = get_opening_range()
        vwap, vwap_std = get_vwap()
        atr = get_atr() or 0.0

        last = df3.iloc[-1]; prev = df3.iloc[-2]
        o = float(last["open"]); h = float(last["high"]); l = float(last["low"]); c = float(last["close"])
        ph = float(prev["high"]); pl = float(prev["low"])

        if self.or_high is None or self.or_low is None or self.or_high <= self.or_low:
            print("self.or_high is None or self.or_low is None or self.or_high <= self.or_low")
            return None

        rng = h - l
        if rng < 0:
            print("rng <= 0")
            return None

        body = abs(c - o)
        body_ok = (body / rng) >= self.cfg.body_ratio_min

        # wick breaks (allow small tick tolerance)
        wick_break_long  = (h >= self.or_high + self.cfg.tick)
        wick_break_short = (l <= self.or_low  - self.cfg.tick)

        wick_body_ok_long  = (c > o) if self.cfg.wick_needs_body_sign else True
        wick_body_ok_short = (c < o) if self.cfg.wick_needs_body_sign else True

        # improved break check: allow wick-only breaks (configurable)
        break_long_ok = (body_ok and (c >= self.or_high + self.cfg.tick)) or (self.cfg.allow_wick_break and wick_break_long and wick_body_ok_long)
        break_short_ok = (body_ok and (c <= self.or_low  - self.cfg.tick)) or (self.cfg.allow_wick_break and wick_break_short and wick_body_ok_short)

        if break_long_ok:
            self.debug["break_long"] += 1
        if break_short_ok:
            self.debug["break_short"] += 1

        # retest buffer (made more permissive)
        buf_long  = self.cfg.retest_atr * float(atr) if atr else self.cfg.retest_atr
        buf_short = self.cfg.retest_atr * self.cfg.retest_atr_mult_short * float(atr) if atr else self.cfg.retest_atr * self.cfg.retest_atr_mult_short

        # allow either the current bar or the prior to be within buffer (more permissive)
        min_low = min(l, pl)
        max_high = max(h, ph)

        touched_long  = (min_low >= self.or_high - buf_long) and (min_low <= self.or_high + buf_long)
        touched_short = (max_high <= self.or_low + buf_short) and (max_high >= self.or_low - buf_short)

        if not touched_long:
            self.debug["retest_long_miss"] += 1
        if not touched_short:
            self.debug["retest_short_miss"] += 1

        # VWAP-based gating (configurable modes)
        vwap_ok_long = vwap_ok_short = True
        mode = (self.cfg.vwap_filter_mode or "off").lower()
        if mode == "location":
            try:
                vwap_ok_long  = c >= float(vwap)
                vwap_ok_short = c <= float(vwap)
            except Exception:
                vwap_ok_long = vwap_ok_short = True
        elif mode == "slope" and vwap_prev is not None:
            try:
                slope_up = float(vwap) >= float(vwap_prev)
                vwap_ok_long, vwap_ok_short = slope_up, (not slope_up)
            except Exception:
                vwap_ok_long = vwap_ok_short = True

        if not vwap_ok_long:
            self.debug["vwap_long_block"] += 1
        if not vwap_ok_short:
            self.debug["vwap_short_block"] += 1

        sigs = []
        # permissive acceptance rule:
        # require break_ok AND (touched OR wick_break) AND vwap_ok
        accept_long = break_long_ok and (touched_long or wick_break_long) and vwap_ok_long
        accept_short = break_short_ok and (touched_short or wick_break_short) and vwap_ok_short

        # detect low-confidence cases (wick-only with low body or missing retest)
        low_conf_long = False
        low_conf_short = False

        # volume-based softness: if volume column exists, allow tiny-volume-based downgrades
        vol_ok = True
        vol_ratio = None
        if 'quote_volume' in last.index or 'volume' in last.index:
            try:
                if 'quote_volume' in df3.columns:
                    v_series = df3['quote_volume'].astype(float)
                else:
                    v_series = (df3['volume'] * df3['close']).astype(float)
                ma = v_series.rolling(20, min_periods=1).mean().iloc[-1]
                last_v = float(v_series.iloc[-1])
                vol_ratio = last_v / (ma if ma>0 else 1.0)
                vol_ok = vol_ratio >= 0.6  # slightly permissive: 60% of MA considered ok
            except Exception:
                vol_ok = True
                vol_ratio = None

        # long low-conf detection
        if accept_long:
            if (not body_ok) or (not touched_long and wick_break_long):
                if not vol_ok:
                    low_conf_long = True
                elif not touched_long:
                    low_conf_long = True

        # short low-conf detection
        if accept_short:
            if (not body_ok) or (not touched_short and wick_break_short):
                if not vol_ok:
                    low_conf_short = True
                elif not touched_short:
                    low_conf_short = True

        # prepare signals (with low_confidence metadata)
        if accept_long:
            entry = h + self.cfg.tick
            stop  = min(l, self.or_high - self.cfg.atr_stop_mult * float(atr)) - self.cfg.tick if atr else (min(l, self.or_high) - self.cfg.tick)
            R = entry - stop
            tp1, tp2 = entry + self.cfg.tp_R1 * R, entry + self.cfg.tp_R2 * R
            low_conf = bool(low_conf_long)
            trade_scale = float(self.cfg.low_conf_trade_scale) if low_conf else 1.0
            if low_conf:
                self.debug["low_conf_signals"] += 1
            sigs.append({
                "stage": "ENTRY", "action": "BUY", "entry": float(entry), "stop": float(stop),
                "targets": [float(tp1), float(tp2)],
                "context": {
                    "mode": "SESSION_OR_LITE", "or_high": float(self.or_high),
                    "atr": float(atr), "vwap": float(vwap), "vwap_std": float(vwap_std if vwap_std is not None else 0.0),
                    "touched_buf": float(buf_long), "body_ok": body_ok, "wick_break": wick_break_long,
                    "vol_ratio": float(vol_ratio) if vol_ratio is not None else None
                },
                "low_confidence": low_conf,
                "trade_size_scale": trade_scale
            })

        if accept_short:
            entry = l - self.cfg.tick
            stop  = max(h, self.or_low + self.cfg.atr_stop_mult * float(atr)) + self.cfg.tick if atr else (max(h, self.or_low) + self.cfg.tick)
            R = stop - entry
            tp1, tp2 = entry - self.cfg.tp_R1 * R, entry - self.cfg.tp_R2 * R
            low_conf = bool(low_conf_short)
            trade_scale = float(self.cfg.low_conf_trade_scale) if low_conf else 1.0
            if low_conf:
                self.debug["low_conf_signals"] += 1
            sigs.append({
                "stage": "ENTRY", "action": "SELL", "entry": float(entry), "stop": float(stop),
                "targets": [float(tp1), float(tp2)],
                "context": {
                    "mode": "SESSION_OR_LITE", "or_low": float(self.or_low),
                    "atr": float(atr), "vwap": float(vwap), "vwap_std": float(vwap_std if vwap_std is not None else 0.0),
                    "touched_buf": float(buf_short), "body_ok": body_ok, "wick_break": wick_break_short,
                    "vol_ratio": float(vol_ratio) if vol_ratio is not None else None
                },
                "low_confidence": low_conf,
                "trade_size_scale": trade_scale
            })

        if not sigs:
            if self.cfg.debug_print:
                print("[SESSION_OR] no signals: break_long_ok=%s touched_long=%s wick_long=%s vwap_ok_long=%s | break_short_ok=%s touched_short=%s wick_short=%s vwap_ok_short=%s" %
                      (break_long_ok, touched_long, wick_break_long, vwap_ok_long, break_short_ok, touched_short, wick_break_short, vwap_ok_short))
            print("no signals")
            return None

        sigs_sorted = sorted(sigs, key=lambda s: (0 if not s.get("low_confidence", False) else 1, -abs((s["entry"] - s["stop"])) ))
        chosen = sigs_sorted[0]
        if self.cfg.debug_print:
            print(f"[SESSION_OR] chosen {chosen['action']} entry={chosen['entry']} stop={chosen['stop']} low_conf={chosen.get('low_confidence')} trade_scale={chosen.get('trade_size_scale')}")
        return chosen
