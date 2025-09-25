# session_or_lite.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

from managers.data_manager import get_data_manager
from utils.session_manager import get_session_manager
from managers.time_manager import get_time_manager
from indicators.global_indicators import get_opening_range
from signals.session_or_analyzer import SessionORAnalyzer
from signals.session_or_signal_generator import SessionORSignalGenerator

@dataclass
class SessionORLiteCfg:
    or_minutes: int = 30
    body_ratio_min: float = 0.03    # more permissive: smaller body accepted
    retest_atr: float = 0.25        # larger retest buffer (more permissive)
    retest_atr_mult_short: float = 1.2
    atr_stop_mult: float = 1.5
    tp_R1: float = 2.0
    tp_R2: float = 3.0
    tick: float = 0.01
    vwap_filter_mode: str = "off"   # default off
    allow_wick_break: bool = True
    wick_needs_body_sign: bool = False
    # extra permissive flags
    allow_either_touched_or_wick: bool = True
    low_conf_trade_scale: float = 0.3
    debug_print: bool = False
    session_score_threshold: float = 0.65
class SessionORLite:
    def __init__(self, cfg: SessionORLiteCfg = SessionORLiteCfg()):
        self.cfg = cfg
        self.session_open: Optional[datetime] = None
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None
        self.session_manager = get_session_manager()
        self.time_manager = get_time_manager()
        self.data_manager = get_data_manager()
        
        # 리팩토링된 컴포넌트들
        self.analyzer = SessionORAnalyzer(cfg)
        self.signal_generator = SessionORSignalGenerator(cfg)
    
    def on_kline_close_3m(self, df3: pd.DataFrame, vwap_prev: Optional[float] = None, now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """3분봉 마감 시 세션 오프닝 레인지 전략 실행"""
        now = now or self.time_manager.get_current_time()
        session_activated = self.session_manager.is_session_active()
        
        if session_activated:
            self.session_open = self.session_manager.get_current_session_info().open_time

        # 오프닝 레인지 레벨 가져오기
        self.or_high, self.or_low = get_opening_range()
        
        if self.or_high is None or self.or_low is None or self.or_high <= self.or_low:
            return None

        # 캔들 데이터 분석
        candle_analysis = self.analyzer.analyze_candle_data(df3, self.or_high, self.or_low)
        if not candle_analysis["valid"]:
            return None
            
        ohlc = candle_analysis["ohlc"]
        prev_ohlc = candle_analysis["prev_ohlc"]
        range_val = candle_analysis["range"]

        # 각종 조건들 분석
        body_conditions = self.analyzer.check_body_conditions(ohlc, range_val)
        break_conditions = self.analyzer.check_break_conditions(ohlc, self.or_high, self.or_low)
        retest_conditions = self.analyzer.check_retest_conditions(ohlc, prev_ohlc, self.or_high, self.or_low)
        vwap_conditions = self.analyzer.check_vwap_conditions(ohlc["c"], vwap_prev)
        volume_conditions = self.analyzer.check_volume_conditions(df3)

        # 모든 조건을 하나의 딕셔너리로 통합
        all_conditions = {
            **body_conditions,
            **break_conditions,
            **retest_conditions,
            **vwap_conditions,
            **volume_conditions
        }

        # 점수 계산
        scores = self.analyzer.calculate_scores(all_conditions)

        # 수락 조건 체크
        acceptance = self.analyzer.check_acceptance_conditions(all_conditions, scores)

        # 최종 분석 결과
        analysis_result = {
            **all_conditions,
            **scores,
            **acceptance
        }

        # 신호 생성
        signals = self.signal_generator.generate_signals(
            analysis_result, ohlc, self.or_high, self.or_low, vwap_conditions
        )

        # 최적의 신호 선택
        chosen_signal = self.signal_generator.select_best_signal(signals)

        if not chosen_signal and self.cfg.debug_print:
            print("[SESSION] no signals generated")

        return chosen_signal
