from typing import Dict, Any, List, Optional
from indicators.global_indicators import get_atr


class SessionORSignalGenerator:
    """세션 오프닝 레인지 신호 생성을 담당하는 클래스"""
    
    def __init__(self, cfg):
        self.cfg = cfg

    def generate_signals(self, analysis_result: Dict[str, Any], 
                        ohlc: Dict[str, float], or_high: float, or_low: float,
                        vwap_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """신호 생성"""
        signals = []
        
        if analysis_result['accept_long']:
            signal = self._create_long_signal(analysis_result, ohlc, or_high, vwap_data)
            if signal:
                signals.append(signal)
        
        if analysis_result['accept_short']:
            signal = self._create_short_signal(analysis_result, ohlc, or_low, vwap_data)
            if signal:
                signals.append(signal)
        
        return signals

    def _create_long_signal(self, analysis_result: Dict[str, Any], ohlc: Dict[str, float], 
                           or_high: float, vwap_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """롱 신호 생성"""
        h, l, c = ohlc["h"], ohlc["l"], ohlc["c"]
        
        # 진입가, 스탑가, 목표가 계산
        entry = h + self.cfg.tick
        atr = get_atr()
        stop = min(l, or_high - self.cfg.atr_stop_mult * float(atr)) - self.cfg.tick if atr else \
               (min(l, or_high) - self.cfg.tick)
        R = entry - stop
        tp1 = entry + self.cfg.tp_R1 * R
        tp2 = entry + self.cfg.tp_R2 * R
        
        trade_scale = 1.0
        
        # 이유 생성
        reasons = self._generate_reasons(analysis_result, True)
        
        return {
            "name": "SESSION",
            "action": "BUY",
            "entry": float(entry),
            "stop": float(stop),
            "targets": [float(tp1), float(tp2)],
            "context": {
                "mode": "SESSION_OR_LITE",
                "or_high": float(or_high),
                "atr": float(atr) if atr else 0.0,
                "vwap": float(vwap_data['vwap']) if vwap_data['vwap'] else 0.0,
                "vwap_std": float(vwap_data['vwap_std']) if vwap_data['vwap_std'] else 0.0,
                "touched_buf": float(analysis_result.get('buf_long', 0)),
                "body_ok": analysis_result['comp_break_long'],
                "wick_break": analysis_result['wick_break_long'],
                "vol_ratio": float(analysis_result.get('vol_ratio', 0)) if analysis_result.get('vol_ratio') else None
            },
            "trade_size_scale": trade_scale,
            "score": float(analysis_result['score_long']),
            "reasons": reasons
        }

    def _create_short_signal(self, analysis_result: Dict[str, Any], ohlc: Dict[str, float], 
                            or_low: float, vwap_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """숏 신호 생성"""
        h, l, c = ohlc["h"], ohlc["l"], ohlc["c"]
        
        # 진입가, 스탑가, 목표가 계산
        entry = l - self.cfg.tick
        atr = get_atr()
        stop = max(h, or_low + self.cfg.atr_stop_mult * float(atr)) + self.cfg.tick if atr else \
               (max(h, or_low) + self.cfg.tick)
        R = stop - entry
        tp1 = entry - self.cfg.tp_R1 * R
        tp2 = entry - self.cfg.tp_R2 * R
        
        trade_scale = 1.0
        
        # 이유 생성
        reasons = self._generate_reasons(analysis_result, False)
        
        return {
            "name": "SESSION",
            "action": "SELL",
            "entry": float(entry),
            "stop": float(stop),
            "targets": [float(tp1), float(tp2)],
            "context": {
                "mode": "SESSION_OR_LITE",
                "or_low": float(or_low),
                "atr": float(atr) if atr else 0.0,
                "vwap": float(vwap_data['vwap']) if vwap_data['vwap'] else 0.0,
                "vwap_std": float(vwap_data['vwap_std']) if vwap_data['vwap_std'] else 0.0,
                "touched_buf": float(analysis_result.get('buf_short', 0)),
                "body_ok": analysis_result['comp_break_short'],
                "wick_break": analysis_result['wick_break_short'],
                "vol_ratio": float(analysis_result.get('vol_ratio', 0)) if analysis_result.get('vol_ratio') else None
            },
            "trade_size_scale": trade_scale,
            "score": float(analysis_result['score_short']),
            "reasons": reasons
        }


    def _generate_reasons(self, analysis_result: Dict[str, Any], is_long: bool) -> List[str]:
        """이유 생성"""
        reasons = []
        
        if is_long:
            if analysis_result['comp_break_long']:
                reasons.append("break_body")
            if analysis_result['wick_break_long']:
                reasons.append("wick_break")
            if analysis_result['touched_long']:
                reasons.append("retest")
            if analysis_result['comp_vwap_long']:
                reasons.append("vwap_ok")
        else:
            if analysis_result['comp_break_short']:
                reasons.append("break_body")
            if analysis_result['wick_break_short']:
                reasons.append("wick_break")
            if analysis_result['touched_short']:
                reasons.append("retest")
            if analysis_result['comp_vwap_short']:
                reasons.append("vwap_ok")
        
        if analysis_result.get('vol_ratio') is not None:
            reasons.append(f"vol_ratio={analysis_result['vol_ratio']:.2f}")
        
        return reasons


    def select_best_signal(self, signals: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """최적의 신호 선택"""
        if not signals:
            return None
        
        # 진입-스탑 거리로 정렬
        signals_sorted = sorted(signals, key=lambda s: -abs((s["entry"] - s["stop"])))
        
        return signals_sorted[0]
