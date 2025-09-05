from typing import Dict, Any, Optional, Tuple
import pandas as pd
from indicators.global_indicators import get_atr, get_vwap


class SessionORAnalyzer:
    """세션 오프닝 레인지 분석을 담당하는 클래스"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.debug = {
            "break_long": 0, "break_short": 0,
            "retest_long_miss": 0, "retest_short_miss": 0,
            "vwap_long_block": 0, "vwap_short_block": 0,
            "low_conf_signals": 0
        }

    def analyze_candle_data(self, df3: pd.DataFrame, or_high: float, or_low: float) -> Dict[str, Any]:
        """캔들 데이터 분석"""
        if df3 is None or len(df3) < 2:
            return {"valid": False, "reason": "insufficient_data"}

        last = df3.iloc[-1]
        prev = df3.iloc[-2]
        
        o = float(last["open"])
        h = float(last["high"])
        l = float(last["low"])
        c = float(last["close"])
        ph = float(prev["high"])
        pl = float(prev["low"])

        if or_high is None or or_low is None or or_high <= or_low:
            return {"valid": False, "reason": "invalid_or_levels"}

        rng = h - l
        if rng < 0:
            return {"valid": False, "reason": "invalid_range"}

        return {
            "valid": True,
            "ohlc": {"o": o, "h": h, "l": l, "c": c},
            "prev_ohlc": {"h": ph, "l": pl},
            "range": rng
        }

    def check_body_conditions(self, ohlc: Dict[str, float], range_val: float) -> Dict[str, Any]:
        """바디 조건 체크"""
        o, h, l, c = ohlc["o"], ohlc["h"], ohlc["l"], ohlc["c"]
        
        body = abs(c - o)
        body_ok = (body / range_val) >= self.cfg.body_ratio_min
        
        return {
            "body": body,
            "body_ok": body_ok,
            "body_ratio": body / range_val if range_val > 0 else 0
        }

    def check_break_conditions(self, ohlc: Dict[str, float], or_high: float, or_low: float) -> Dict[str, Any]:
        """브레이크 조건 체크"""
        o, h, l, c = ohlc["o"], ohlc["h"], ohlc["l"], ohlc["c"]
        
        # 윅 브레이크
        wick_break_long = (h >= or_high + self.cfg.tick)
        wick_break_short = (l <= or_low - self.cfg.tick)

        # 윅 바디 조건
        wick_body_ok_long = (c > o) if self.cfg.wick_needs_body_sign else True
        wick_body_ok_short = (c < o) if self.cfg.wick_needs_body_sign else True

        # 브레이크 조건 (바디 또는 윅)
        break_long_ok = (self.cfg.body_ratio_min and (c >= or_high + self.cfg.tick)) or \
                       (self.cfg.allow_wick_break and wick_break_long and wick_body_ok_long)
        break_short_ok = (self.cfg.body_ratio_min and (c <= or_low - self.cfg.tick)) or \
                        (self.cfg.allow_wick_break and wick_break_short and wick_body_ok_short)

        return {
            "wick_break_long": wick_break_long,
            "wick_break_short": wick_break_short,
            "wick_body_ok_long": wick_body_ok_long,
            "wick_body_ok_short": wick_body_ok_short,
            "break_long_ok": break_long_ok,
            "break_short_ok": break_short_ok
        }

    def check_retest_conditions(self, ohlc: Dict[str, float], prev_ohlc: Dict[str, float], 
                               or_high: float, or_low: float) -> Dict[str, Any]:
        """리테스트 조건 체크"""
        h, l = ohlc["h"], ohlc["l"]
        ph, pl = prev_ohlc["h"], prev_ohlc["l"]
        
        atr = get_atr()
        buf_long = self.cfg.retest_atr * float(atr) if atr else self.cfg.retest_atr
        buf_short = self.cfg.retest_atr * self.cfg.retest_atr_mult_short * float(atr) if atr else \
                   self.cfg.retest_atr * self.cfg.retest_atr_mult_short

        # 현재 또는 이전 바의 최고/최저가가 버퍼 내에 있는지 체크
        min_low = min(l, pl)
        max_high = max(h, ph)
        
        touched_long = (min_low >= or_high - buf_long) and (min_low <= or_high + buf_long)
        touched_short = (max_high <= or_low + buf_short) and (max_high >= or_low - buf_short)

        return {
            "touched_long": touched_long,
            "touched_short": touched_short,
            "buf_long": buf_long,
            "buf_short": buf_short
        }

    def check_vwap_conditions(self, close: float, vwap_prev: Optional[float] = None) -> Dict[str, Any]:
        """VWAP 조건 체크"""
        vwap, vwap_std = get_vwap()
        vwap_ok_long = vwap_ok_short = True
        
        mode = (self.cfg.vwap_filter_mode or "off").lower()
        
        if mode == "location":
            try:
                vwap_ok_long = close >= float(vwap)
                vwap_ok_short = close <= float(vwap)
            except Exception:
                vwap_ok_long = vwap_ok_short = True
        elif mode == "slope" and vwap_prev is not None:
            try:
                slope_up = float(vwap) >= float(vwap_prev)
                vwap_ok_long, vwap_ok_short = slope_up, (not slope_up)
            except Exception:
                vwap_ok_long = vwap_ok_short = True

        return {
            "vwap_ok_long": vwap_ok_long,
            "vwap_ok_short": vwap_ok_short,
            "vwap": vwap,
            "vwap_std": vwap_std
        }

    def check_volume_conditions(self, df3: pd.DataFrame) -> Dict[str, Any]:
        """볼륨 조건 체크"""
        vol_ok = True
        vol_ratio = None
        
        try:
            if 'quote_volume' in df3.columns:
                v_series = df3['quote_volume'].astype(float)
            elif 'volume' in df3.columns and 'close' in df3.columns:
                v_series = (df3['volume'].astype(float) * df3['close'].astype(float))
            else:
                v_series = None

            if v_series is not None:
                ma = v_series.rolling(20, min_periods=1).mean().iloc[-1]
                last_v = float(v_series.iloc[-1])
                vol_ratio = last_v / (ma if ma > 0 else 1.0)
                vol_threshold = getattr(self.cfg, 'vol_ok_threshold', 0.5)
                vol_ok = vol_ratio >= vol_threshold
            else:
                vol_ok = True
                vol_ratio = None
        except Exception as e:
            vol_ok = True
            vol_ratio = None
            if getattr(self.cfg, 'debug_print', False):
                print('[SESSION_OR] volume calc error:', repr(e))

        return {
            "vol_ok": vol_ok,
            "vol_ratio": vol_ratio
        }

    def calculate_scores(self, conditions: Dict[str, Any]) -> Dict[str, float]:
        """점수 계산"""
        W = {"break": 0.30, "touched": 0.30, "wick": 0.20, "vwap": 0.10, "vol": 0.10}
        
        score_long = (W['break'] * float(conditions['break_long_ok']) +
                     W['touched'] * float(conditions['touched_long']) +
                     W['wick'] * float(conditions['wick_break_long'] and conditions['wick_body_ok_long']) +
                     W['vwap'] * float(conditions['vwap_ok_long']) +
                     W['vol'] * float(conditions['vol_ok']))

        score_short = (W['break'] * float(conditions['break_short_ok']) +
                      W['touched'] * float(conditions['touched_short']) +
                      W['wick'] * float(conditions['wick_break_short'] and conditions['wick_body_ok_short']) +
                      W['vwap'] * float(conditions['vwap_ok_short']) +
                      W['vol'] * float(conditions['vol_ok']))

        return {
            "score_long": score_long,
            "score_short": score_short
        }

    def check_acceptance_conditions(self, conditions: Dict[str, Any], scores: Dict[str, float]) -> Dict[str, Any]:
        """수락 조건 체크"""
        SCORE_THRESHOLD = getattr(self.cfg, "session_score_threshold", 0.50)
        
        # 기본 조건들
        comp_break_long = conditions['break_long_ok']
        comp_break_short = conditions['break_short_ok']
        comp_wick_long = conditions['wick_break_long'] and conditions['wick_body_ok_long']
        comp_wick_short = conditions['wick_break_short'] and conditions['wick_body_ok_short']
        comp_touched_long = conditions['touched_long']
        comp_touched_short = conditions['touched_short']
        comp_vwap_long = conditions['vwap_ok_long']
        comp_vwap_short = conditions['vwap_ok_short']
        comp_vol = conditions['vol_ok']

        # 최종 수락 조건
        accept_long = (scores['score_long'] >= SCORE_THRESHOLD) and \
                     (comp_break_long or comp_wick_long or comp_touched_long)
        accept_short = (scores['score_short'] >= SCORE_THRESHOLD) and \
                      (comp_break_short or comp_wick_short or comp_touched_short)

        return {
            "accept_long": accept_long,
            "accept_short": accept_short,
            "comp_break_long": comp_break_long,
            "comp_break_short": comp_break_short,
            "comp_wick_long": comp_wick_long,
            "comp_wick_short": comp_wick_short,
            "comp_touched_long": comp_touched_long,
            "comp_touched_short": comp_touched_short,
            "comp_vwap_long": comp_vwap_long,
            "comp_vwap_short": comp_vwap_short,
            "comp_vol": comp_vol
        }
