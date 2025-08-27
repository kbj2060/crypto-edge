#!/usr/bin/env python3
"""
세션 기반 고급 전략 (Session-Based Advanced Strategy)
- 플레이북 A: 오프닝 드라이브 풀백 매수/매도
- 플레이북 B: 유동성 스윕 & 리클레임
- 플레이북 C: VWAP 리버전(평균회귀) 페이드
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pytz

from indicators.global_indicators import get_atr, get_opening_range, get_vwap
from indicators.moving_averages import calculate_ema
from utils.time_manager import get_time_manager


@dataclass
class SessionConfig:
    """세션 기반 전략 설정"""
    # 기본 설정
    symbol: str = "ETHUSDT"
    timeframe: str = "3m"          # 그대로 사용(또는 "5m")
    
    # 세션 설정
    ses_vwap_start_utc: str = "13:30 UTC"  # NY Open (KST 22:30, DST중)
    london_session_start_utc: str = "07:00 UTC"  # London Open (KST 16:00)
    or_minutes: int = 30           # 15 → 30 (OR 신뢰도↑)
    
    # 지표 설정
    ema_fast: int = 9
    ema_slow: int = 26             # 20 → 26 (추세필터 완만)
    atr_len: int = 14
    trend_filter_ma: int = 100     # 50 → 100 (큰 흐름 우선)
    
    # 플레이북 A: 오프닝 드라이브 풀백 (단타용 튜닝)
    min_drive_return_R: float = 1.0            # 0.8 → 1.0
    pullback_depth_atr: Tuple[float, float] = (0.7, 1.6)     # 범위 약간 넓혀 변동성 흡수
    trigger_type: str = "close_reject"  # 'close_reject' 또는 'wick_touch'
    stop_atr_mult: float = 1.2                 # 1.0 → 1.2
    tp1_R: float = 1.5                         # 1.2 → 1.5
    tp2_to_level: str = "OR_ext|PrevHigh|VWAP"  # 2차 목표 우선순위
    partial_out: float = 0.4                   # 0.5 → 0.4 (러너 더 보유)
    max_hold_min: int = 180                  # 60 → 180
    max_slippage_pct: float = 0.02             # 0.025 → 0.02
    
    # 플레이북 B: 유동성 스윕 & 리클레임 (단타용 튜닝)
    sweep_depth_atr_min: float = 0.35          # 0.25 → 0.35
    reclaim_close_rule: str = "close_above_level"  # 롱: 레벨 위 종가 마감
    stop_buffer_atr: float = 0.7               # 0.5 → 0.7
    tp1_to_b: str = "VWAP"  # 1차 목표 (Play B용)
    tp2_to_b: str = "opposite_range_edge"  # 2차 목표 (Play B용)
    
    # 플레이북 C: VWAP 리버전(평균회귀) 페이드 (단타용 튜닝)
    sd_k_enter: float = 2.0                    # 1.8 → 2.0 (더 보수적)
    sd_k_reenter: float = 1.5
    stop_outside_sd_k: float = 3.0             # 2.5 → 3.0
    tp1_to_c: str = "VWAP"  # 1차 목표: VWAP 터치 (Play C용)
    tp2_to_c: float = 0.5                      # 0.4 → 0.5
    trend_filter_slope: float = 0.0005         # 0.0 → 0.0005 (강추세 역추세 페이드 억제)
    
    # 단계형 신호 설정 (ENTRY 소폭↑)
    entry_thresh: float = 0.62
    setup_thresh: float = 0.42
    headsup_thresh: float = 0.30
    
    # Gate 설정
    min_sweep_depth_atr: float = 0.35
    max_slippage_gate: float = 0.02
    min_volume_ratio: float = 0.7              # 0.5 → 0.7 (체결 질 우선)
    
    # Score 가중치 (추세/구조 비중↑)
    weight_direction: float = 0.30
    weight_breakout_sweep: float = 0.22
    weight_pullback: float = 0.12
    weight_baseline: float = 0.08
    weight_timing: float = 0.08
    weight_orderflow: float = 0.08
    weight_risk: float = 0.12
    
    # OR 정책 (단타는 완전 OR 선호)
    strict_or: bool = True
    min_or_bars: int = 0          # 무시(엄격하게 ready만)
    partial_or_tier_cap: str = "HEADSUP"


class SessionBasedStrategy:
    """세션 기반 고급 전략"""
    
    def __init__(self, config: SessionConfig):
        self.config = config
        self.session_start_time = None
        self.session_vwap = None
        self.session_std = None
        self.opening_range = None
        self.prev_day_hlc = None
        self.last_swing_hl = None
        
        # Time Manager 초기화
        self.time_manager = get_time_manager()
        
    def calculate_session_vwap(
        self, df: pd.DataFrame, session_start: datetime, session_end: datetime
    ) -> Tuple[float, float]:
        """세션 구간 VWAP 및 표준편차 계산 (반개구간 [start, end), 누적 σ) - 글로벌 지표로 대체됨"""
        if df.empty:
            return np.nan, np.nan
        # 안전장치: tz-aware & 정렬
        assert df.index.tz is not None, "df.index must be tz-aware(UTC)"
        df = df.sort_index()

        # 세션 구간 반개구간으로 슬라이스 (다음 세션 첫 봉 중복 방지)
        mask = (df.index >= session_start) & (df.index < session_end)
        s = df.loc[mask]
        if s.empty:
            return np.nan, np.nan

        # VWAP: typical price * volume 가중 (close만 써도 되지만 안정성↑)
        price = (s["high"] + s["low"] + s["close"]) / 3.0
        vol = s["volume"].astype("float64")
        v_sum = np.maximum(vol.sum(), 1e-9)
        vwap = float((price * vol).sum() / v_sum)

        # 세션 누적 표준편차: expanding std의 마지막 값 사용 (ddof=0 권장)
        # (세션 밴드 = 가격의 분산을 세션 누적 관점으로 측정)
        std = float(price.expanding().std(ddof=0).iloc[-1])
        return vwap, std

    def _session_slice(self, df: pd.DataFrame, session_start: datetime) -> pd.DataFrame:
        """세션 시작부터 다음 세션 시작 전까지의 데이터 슬라이스 (세션 경계 정확)"""
        if df.empty:
            return df
        
        # DataFrame 복사 및 인덱스 timezone 처리
        df_copy = df.copy()
        df_copy = df_copy.sort_index()
        
        session_end = self.time_manager.get_next_session_start(session_start)
        return df_copy.loc[(df_copy.index >= session_start) & (df_copy.index < session_end)]
    
    def process_liquidation_stream(self, liquidation_events: List[Dict], 
                                    current_time: datetime) -> Dict[str, Any]:
        """청산 스트림 처리: 1초bin 누적 ≥1h + SELL/BUY→롱/숏 청산 매핑 고정"""
        try:
            if not liquidation_events:
                return {}
            
            # 1시간(3600초) 동안의 청산 이벤트만 필터링
            one_hour_ago = current_time - timedelta(seconds=3600)
            recent_events = [
                event for event in liquidation_events 
                if self.time_manager.get_timestamp_datetime(event.get('timestamp', current_time)) >= one_hour_ago
            ]
            
            if not recent_events:
                return {}
            
            # SELL/BUY→롱/숏 청산 매핑 고정
            long_liquidations = [e for e in recent_events if e.get('side') == 'SELL']
            short_liquidations = [e for e in recent_events if e.get('side') == 'BUY']
            
            # 누적 청산량 계산
            long_volume = sum(e.get('size') for e in long_liquidations)
            short_volume = sum(e.get('size') for e in short_liquidations)
            
            # 청산 강도 계산 (LPI 기반)
            long_intensity = np.mean([e.get('lpi') for e in long_liquidations])
            short_intensity = np.mean([e.get('lpi') for e in short_liquidations])
            
            return {
                'long_liquidations': long_liquidations,
                'short_liquidations': short_liquidations,
                'long_volume': long_volume,
                'short_volume': short_volume,
                'long_intensity': long_intensity,
                'short_intensity': short_intensity,
                'total_events': len(recent_events),
                'time_window': '1h'
            }
            
        except Exception as e:
            print(f"❌ 청산 스트림 처리 오류: {e}")
            return {}
    
    def check_gates(
            self, 
            df: pd.DataFrame, 
            session_vwap: float, 
            opening_range: Dict[str, float], 
            atr: float, 
            playbook: str, 
            side: str, 
            key_levels: Dict[str, float] = None,
            liquidation_data: Dict[str, float] = None
        ) -> Tuple[bool, Dict[str, Any]]:
        """Gate(필수 최소 조건) 확인"""
        try:
            # DataFrame이 비어있거나 인덱싱이 불가능한 경우 체크
            if df.empty or len(df) == 0:
                return False, {}
                
            current_price = df['close'].iloc[-1]
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            
            # EMA 계산
            ema_fast = calculate_ema(df['close'], self.config.ema_fast)
            ema_slow = calculate_ema(df['close'], self.config.ema_slow)
            
            # EMA가 충분한 데이터를 가지고 있는지 확인
            if ema_fast.empty or ema_slow.empty or len(ema_fast) == 0 or len(ema_slow) == 0:
                return False, {}
            
            gate_results = {}
            
            # === 방향 게이트 ===
            if side == 'LONG':
                direction_gate_a = ema_fast.iloc[-1] > ema_slow.iloc[-1]
                direction_gate_b = current_price > session_vwap
            else:  # SHORT
                direction_gate_a = ema_fast.iloc[-1] < ema_slow.iloc[-1]
                direction_gate_b = current_price < session_vwap
            
            # A 플레이북도 OR 조건으로 완화 (시그널 생성 증가)
            if playbook == 'A':
                # SETUP: EMA정렬 OR 가격·VWAP 정렬
                # ENTRY: EMA정렬 AND 가격·VWAP 정렬
                direction_gate = direction_gate_a or direction_gate_b
            else:
                direction_gate = direction_gate_a or direction_gate_b
            gate_results['direction'] = direction_gate
            
            # === 구조 게이트 ===
            structure_gate = False
            if playbook == 'A':  # 오프닝 드라이브 풀백
                # opening_range가 유효한지 확인
                if not opening_range or 'high' not in opening_range or 'low' not in opening_range:
                    structure_gate = False
                else:
                    if side == 'LONG':
                        structure_gate = current_high > opening_range['high']
                    else:
                        structure_gate = current_low < opening_range['low']
            elif playbook == 'B':  # 유동성 스윕 & 리클레임
                # 키 레벨 스윕 확인 (OR이 아닌 prev_day_high/low 등)
                pdh = (key_levels or {}).get('prev_day_high')
                pdl = (key_levels or {}).get('prev_day_low')
                sweep_depth = 0.0
                
                if side == 'LONG' and pdl is not None:
                    sweep_depth = max(0.0, (pdl - current_low) / atr) if atr > 0 else 0.0
                    # 스윕 깊이 조건
                    sweep_condition = sweep_depth >= self.config.min_sweep_depth_atr
                    
                    # 리클레임 확증: 현재 저가가 레벨 근처 (더 관대하게)
                    reclaim_condition = current_low >= (pdl - atr * 0.5)  # 레벨에서 0.5ATR 이내
                    
                    structure_gate = sweep_condition and reclaim_condition
                    
                elif side == 'SHORT' and pdh is not None:
                    sweep_depth = max(0.0, (current_high - pdh) / atr) if atr > 0 else 0.0
                    # 스윕 깊이 조건
                    sweep_condition = sweep_depth >= self.config.min_sweep_depth_atr
                    
                    # 리클레임 확증: 현재 고가가 레벨 근처 (더 관대하게)
                    reclaim_condition = current_high <= (pdh + atr * 0.5)  # 레벨에서 0.5ATR 이내
                    
                    structure_gate = sweep_condition and reclaim_condition
                    
                else:
                    structure_gate = False
                
                gate_results['sweep_atr'] = max(0.0, sweep_depth)
                gate_results['reclaim_confirmed'] = structure_gate  # 리클레임 확증 상태 저장
            elif playbook == 'C':  # VWAP 리버전 페이드 (임계값 완화)
                if self.session_std is not None and self.session_std > 0:
                    if side == 'LONG':
                        # -2σ → -1.8σ로 완화 (설정값 사용)
                        structure_gate = current_price < (session_vwap - self.config.sd_k_enter * self.session_std)
                    else:
                        # +2σ → +1.8σ로 완화 (설정값 사용)
                        structure_gate = current_price > (session_vwap + self.config.sd_k_enter * self.session_std)
                else:
                    # session_std가 없을 때는 VWAP 기준으로만 판단 (더 관대하게)
                    if side == 'LONG':
                        structure_gate = current_price < session_vwap * 0.998  # VWAP 대비 0.2% 하락 (0.5% → 0.2%)
                    else:
                        structure_gate = current_price > session_vwap * 1.002  # VWAP 대비 0.2% 상승 (0.5% → 0.2%)
            
            gate_results['structure'] = structure_gate
            
            # === 실행/유동성 게이트 ===
            # 슬리피지 계산 (간단한 구현)
            if playbook == 'A' and opening_range and 'high' in opening_range and 'low' in opening_range:
                if side == 'LONG':
                    slippage = abs(current_price - opening_range['high']) / current_price if current_price > 0 else 0.01
                elif side == 'SHORT':
                    slippage = abs(current_price - opening_range['low']) / current_price if current_price > 0 else 0.01
                else:
                    slippage = 0.01  # 기본값
            else:
                slippage = 0.01  # 기본값
            
            slippage_gate = slippage <= self.config.max_slippage_gate
            gate_results['slippage'] = slippage_gate
            
            # 거래량 게이트
            if len(df) >= 20:
                recent_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].iloc[-20:].mean()
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 0
                volume_gate = volume_ratio >= self.config.min_volume_ratio
            else:
                volume_gate = True  # 데이터 부족 시 통과
            
            gate_results['volume'] = volume_gate
            
            # 모든 게이트 통과 여부
            all_gates_passed = all([
                direction_gate, structure_gate, slippage_gate, volume_gate
            ])
            
            gate_results['all_passed'] = all_gates_passed
            gate_results['slippage_value'] = slippage
            gate_results['volume_ratio'] = volume_ratio if 'volume_ratio' in locals() else 0
            
            return all_gates_passed, gate_results
            
        except Exception as e:
            print(f"❌ Gate 확인 오류: {e}")
            return False, {}
    
    def calculate_score(self, df: pd.DataFrame, session_vwap: float,
                        opening_range: Dict[str, float], atr: float,
                        playbook: str, side: str, gate_results: Dict[str, Any], 
                        current_time: datetime, key_levels: Dict[str, float] = None) -> float:
        """Score(가중치 합산) 계산"""
        try:
            current_price = df['close'].iloc[-1]
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            
            # EMA 계산
            ema_fast = calculate_ema(df['close'], self.config.ema_fast)
            ema_slow = calculate_ema(df['close'], self.config.ema_slow)
            
            score = 0.0
            
            # === 방향 정렬 (0.25) ===
            if side == 'LONG':
                price_vwap_score = 0.13 if current_price > session_vwap else 0.0
                ema_score = 0.12 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else 0.0
            else:
                price_vwap_score = 0.13 if current_price < session_vwap else 0.0
                ema_score = 0.12 if ema_fast.iloc[-1] < ema_slow.iloc[-1] else 0.0
            
            score += price_vwap_score + ema_score
            
            # === 돌파/스윕 질 (0.20) ===
            if playbook == 'A' and opening_range and 'high' in opening_range and 'low' in opening_range:  # OR 돌파
                if side == 'LONG':
                    breakout_strength = (current_high - opening_range['high']) / atr if atr > 0 else 0
                else:
                    breakout_strength = (opening_range['low'] - current_low) / atr if atr > 0 else 0
                breakout_score = min(breakout_strength, 1.0) * self.config.weight_breakout_sweep
            elif playbook == 'B':  # 스윕
                # 스윕 깊이를 gate_results에서 가져와서 사용
                sweep_atr = float(gate_results.get('sweep_atr', 0.0))
                breakout_score = min(sweep_atr, 1.0) * self.config.weight_breakout_sweep
            else:  # Play C
                # VWAP 리버전 강도 계산 (VWAP에서의 거리 기반)
                if side == 'LONG':
                    # 롱: VWAP 아래에서의 거리 (음수일 때 더 강한 리버전)
                    vwap_distance = (current_price - session_vwap) / session_vwap if session_vwap > 0 else 0
                    # VWAP 아래에 있을 때 가점 (0.2% 이하)
                    if vwap_distance < -0.002:
                        breakout_score = min(abs(vwap_distance) * 50, 1.0) * self.config.weight_breakout_sweep
                    else:
                        breakout_score = 0.1 * self.config.weight_breakout_sweep
                else:
                    # 숏: VWAP 위에서의 거리 (양수일 때 더 강한 리버전)
                    vwap_distance = (current_price - session_vwap) / session_vwap if session_vwap > 0 else 0
                    # VWAP 위에 있을 때 가점 (0.2% 이상)
                    if vwap_distance > 0.002:
                        breakout_score = min(vwap_distance * 50, 1.0) * self.config.weight_breakout_sweep
                    else:
                        breakout_score = 0.1 * self.config.weight_breakout_sweep
            
            score += breakout_score
            
            # === 풀백 품질 (0.15) ===
            if playbook == 'A':
                # 풀백 깊이 계산 (실제 값 사용)
                if side == 'LONG' and 'high' in opening_range:
                    # 롱: OR 돌파 후 고점에서 풀백까지의 깊이
                    or_breakout_mask = df['high'] > opening_range['high']
                    if or_breakout_mask.any():
                        post_breakout_df = df[or_breakout_mask]
                        if not post_breakout_df.empty:
                            drive_high = post_breakout_df['high'].max()
                            drive_high_idx = post_breakout_df['high'].idxmax()
                            post_high_mask = df.index > drive_high_idx
                            if post_high_mask.any():
                                post_high_df = df[post_high_mask]
                                pullback_low = post_high_df['low'].min()
                                if not pd.isna(pullback_low):
                                    pullback_depth = (drive_high - pullback_low) / atr
                                else:
                                    pullback_depth = 0.8  # 기본값
                            else:
                                pullback_depth = 0.8  # 기본값
                        else:
                            pullback_depth = 0.8  # 기본값
                    else:
                        pullback_depth = 0.8  # 기본값
                elif side == 'SHORT' and 'low' in opening_range:
                    # 숏: OR 이탈 후 저점에서 풀백까지의 깊이
                    or_breakdown_mask = df['low'] < opening_range['low']
                    if or_breakdown_mask.any():
                        post_breakdown_df = df[or_breakdown_mask]
                        if not post_breakdown_df.empty:
                            drive_low = post_breakdown_df['low'].min()
                            drive_low_idx = post_breakdown_df['low'].idxmin()
                            post_low_mask = df.index > drive_low_idx
                            if post_low_mask.any():
                                post_low_df = df[post_low_mask]
                                pullback_high = post_low_df['high'].max()
                                if not pd.isna(pullback_high):
                                    pullback_depth = (pullback_high - drive_low) / atr
                                else:
                                    pullback_depth = 0.8  # 기본값
                            else:
                                pullback_depth = 0.8  # 기본값
                        else:
                            pullback_depth = 0.8  # 기본값
                    else:
                        pullback_depth = 0.8  # 기본값
                else:
                    pullback_depth = 0.8  # 기본값
                
                # 가우시안 스코어: 0.4~1.6×ATR 범위에 가까울수록 가점
                optimal_depth = 1.0
                depth_score = np.exp(-((pullback_depth - optimal_depth) ** 2) / 0.5)
                pullback_score = depth_score * self.config.weight_pullback
            else:
                pullback_score = 0.1 * self.config.weight_pullback  # 기본값
            
            score += pullback_score
            
            # === 기준선 근접/복귀 (0.10) ===
            baseline_score = 0.0
            
            if side == 'LONG':
                # 롱: 저가가 기준선에 근접하는지 확인
                ema_touch = abs(current_low - ema_slow.iloc[-1]) <= atr * 0.3
                vwap_touch = abs(current_low - session_vwap) <= atr * 0.3
                
                # 추가: 종가가 기준선 위에 있는지 확인
                ema_above = current_price > ema_slow.iloc[-1]
                vwap_above = current_price > session_vwap
                
                baseline_score = ((ema_touch or vwap_touch) and (ema_above or vwap_above)) * self.config.weight_baseline
            else:
                # 숏: 고가가 기준선에 근접하는지 확인
                ema_touch = abs(current_high - ema_slow.iloc[-1]) <= atr * 0.3
                vwap_touch = abs(current_high - session_vwap) <= atr * 0.3
                
                # 추가: 종가가 기준선 아래에 있는지 확인
                ema_below = current_price < ema_slow.iloc[-1]
                vwap_below = current_price < session_vwap
                
                baseline_score = ((ema_touch or vwap_touch) and (ema_below or vwap_below)) * self.config.weight_baseline
            
            score += baseline_score
            
            # === 세션 타이밍 (0.10) ===
            # 세션 시작 시간과의 거리로 계산
            if getattr(self, 'session_start_time', None):
                now_ts = current_time or (df.index[-1] if hasattr(df.index, 'tz') else self.time_manager.get_current_time())
                time_diff = abs((now_ts - self.session_start_time).total_seconds() / 60)  # 분 단위
                # 세션 시작 ±90분 내: 최고점, ±180분 내: 중간점, 그 외: 낮은 점수
                if time_diff <= 90:
                    timing_score = 1.0  # MID: +0.05 가점
                elif time_diff <= 180:
                    timing_score = 0.6  # 중간
                else:
                    timing_score = 0.2  # OPEN: -0.05 감점
            else:
                timing_score = 0.4  # 기본값
            
            # 세션 타이밍 가중치 캡 (안전 캡)
            timing_score = min(timing_score, 0.8)  # 안전 캡
            
            score += timing_score * self.config.weight_timing
            
            # === 오더플로우 (0.10) ===
            # 실제 거래량과 청산 데이터 기반 계산
            orderflow_score = 0.0
            
            # 거래량 급증 확인
            if len(df) >= 20:
                recent_volume = df['volume'].iloc[-1]
                avg_volume = df['volume'].iloc[-20:].mean()
                if avg_volume > 0:
                    volume_surge = recent_volume / avg_volume
                    if volume_surge >= 2.0:
                        orderflow_score += 0.1  # 거래량 급증
                    elif volume_surge >= 1.5:
                        orderflow_score += 0.05  # 거래량 증가
            
            # 청산 데이터 활용 (key_levels에서 가져오거나 기본값 사용)
            liquidation_data = self.bucket_aggregator.get_bucket()
            if liquidation_data:
                # 롱/숏 청산량 비율 계산
                long_vol = sum(x.get('size')*x.get('price') for x in liquidation_data if x.get('side') == 'SELL')
                short_vol = sum(x.get('size')*x.get('price') for x in liquidation_data if x.get('side') == 'BUY')
                total_vol = long_vol + short_vol
                
                if total_vol > 0 and side == 'LONG' and short_vol > long_vol:
                    # 롱 신호에서 숏 청산이 많으면 가점
                    orderflow_score += 0.1
                elif total_vol > 0 and side == 'SHORT' and long_vol > short_vol:
                    # 숏 신호에서 롱 청산이 많으면 가점
                    orderflow_score += 0.1
            else:
                # 청산 데이터가 없는 경우 기본값
                orderflow_score += 0.1
            
            orderflow_score = min(orderflow_score, 0.1)  # 최대 0.1 (0.2 → 0.1)
            score += orderflow_score
            
            # === 리스크 적정성 (0.10) ===
            # 실제 스탑 거리 계산
            stop_distance = atr * 1.0
            
            risk_score = 0.0
            if atr > 0:
                stop_atr_ratio = stop_distance / atr
                if 0.6 <= stop_atr_ratio <= 1.6:
                    risk_score = 1.0
                elif 0.4 <= stop_atr_ratio <= 2.0:
                    risk_score = 0.5
                else:
                    risk_score = 0.2
            
            risk_score *= self.config.weight_risk
            score += risk_score
            
            return min(score, 1.0)  # 최대 1.0
            
        except Exception as e:
            print(f"❌ Score 계산 오류: {e}")
            return 0.0
    
    def analyze_staged_signal(self, df: pd.DataFrame, session_vwap: float,
                                opening_range: Dict[str, float], atr: float,
                                playbook: str, side: str, key_levels: Dict[str, float] = None,
                                current_time: datetime = None) -> Optional[Dict[str, Any]]:
        """단계형 신호 분석: Gate → Score → 등급/행동"""
        try:
            # === Gate 확인 ===
            gates_passed, gate_results = self.check_gates(
                df, session_vwap, opening_range, atr, playbook, side, key_levels
            )
            
            if not gates_passed:
                return None
            
            # === Score 계산 ===
            score = self.calculate_score(
                df, session_vwap, opening_range, atr, playbook, side, gate_results, current_time, key_levels
            )
            
            # === 등급/행동 결정 ===
            signal_type = None
            action = None
            confidence = 0.0
            
            if score >= self.config.entry_thresh:
                signal_type = 'ENTRY'
                action = 'BUY' if side == 'LONG' else 'SELL'
                confidence = min(score, 0.95)
            elif score >= self.config.setup_thresh:
                signal_type = 'SETUP'
                action = 'OBSERVE'
                confidence = score
            elif score >= self.config.headsup_thresh:
                signal_type = 'HEADS_UP'
                action = 'ALERT'
                confidence = score
            else:
                return None
            
            # === 신호 정보 구성 ===
            signal = {
                'signal_type': f"{playbook}_{signal_type}_{side}",
                'action': action,
                'confidence': confidence,
                'score': score,
                'playbook': playbook,
                'side': side,
                'timestamp': datetime.now(),
                'timestamp': self.time_manager.get_current_time(),
                'gate_results': gate_results,
                'stage': signal_type
            }
            
            # Entry 신호인 경우 추가 정보
            if signal_type == 'ENTRY':
                current_price = df['close'].iloc[-1]
                current_high = df['high'].iloc[-1]
                current_low = df['low'].iloc[-1]
                
                if side == 'LONG':
                    entry_price = current_price
                    stop_loss = current_low - atr * 0.3
                    risk = entry_price - stop_loss
                    tp1 = entry_price + risk * self.config.tp1_R
                else:
                    entry_price = current_price
                    stop_loss = current_high + atr * 0.3
                    risk = stop_loss - entry_price
                    tp1 = entry_price - risk * self.config.tp1_R
                
                signal.update({
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit1': tp1,
                    'risk_reward': self.config.tp1_R,
                    'partial_1': 0.4,               # TP1에서 40% 청산
                    'trail_after_tp1_atr_mult': 1.0,# 남은 60% ATR*1.0 트레일
                    'hard_timeout_min': 240         # 4시간 초과 보유 금지(단타)
                })
            
            return signal
            
        except Exception as e:
            print(f"❌ 단계형 신호 분석 오류: {e}")
            return None
    
    def get_session_start_time(self, current_time) -> datetime:
        """가장 최근에 완성된 OR의 세션 시작 시간을 반환 (세션 매니저 사용)"""
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=pytz.UTC)
        
        try:
            session_start_tuple = self.time_manager.get_session_open_time()
            
            if session_start_tuple:
                # 현재 활성 세션의 시작 시간 반환 (튜플의 첫 번째 요소가 datetime)
                return session_start_tuple[0]
            
            # 활성 세션이 없으면 가장 최근 세션 시작 시간 반환
            session_history = self.time_manager.get_session_history()
            
            if session_history:
                # 가장 최근 세션 찾기
                latest_session = max(session_history.keys(), key=lambda k: session_history[k].get('session_open_time', ''))
                latest_session_info = session_history[latest_session]
                session_open_time_str = latest_session_info.get('session_open_time')
                if session_open_time_str:
                    try:
                        return datetime.fromisoformat(session_open_time_str.replace('Z', '+00:00'))
                    except:
                        pass
            
            # 폴백: 기본 세션 시작 시간 계산
            print(f"   ⚠️ 세션 매니저에서 세션 시작 시간을 가져올 수 없음. 기본값 사용")
            return current_time.replace(hour=13, minute=30, second=0, microsecond=0) - timedelta(days=1)
            
        except Exception as e:
            print(f"   ⚠️ 세션 매니저 사용 실패: {e}. 기본값 사용")
            # 폴백: 기본 세션 시작 시간 계산
            return current_time.replace(hour=13, minute=30, second=0, microsecond=0) - timedelta(days=1)
    
    def _get_session_type(self) -> str:
        """세션 시작 시간으로부터 세션 타입 식별 (세션 매니저 사용)"""
        # 세션 매니저에서 현재 세션 정보 가져오기
        session_status = self.time_manager.get_session_status()
        current_session = session_status.get('current_session', 'UNKNOWN')
        
        # 세션 이름을 한글로 변환
        session_name_map = {
            'EUROPE': '런던',
            'US': '뉴욕',
            'EUROPE_ACTIVE': '런던',
            'US_ACTIVE': '뉴욕'
        }
        
        return session_name_map.get(current_session, current_session)
    
    def analyze_playbook_a_opening_drive_pullback(self, df: pd.DataFrame, 
                                                    session_vwap: float,
                                                    opening_range: Dict[str, float],
                                                    atr: float) -> Optional[Dict]:
        """플레이북 A: 오프닝 드라이브 풀백 분석 (롱/숏)"""
        if len(df) < 50 or not opening_range:
            return None
        
        try:
            current_price = df['close'].iloc[-1]
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            
            # EMA 계산
            ema_fast = calculate_ema(df['close'], self.config.ema_fast)
            ema_slow = calculate_ema(df['close'], self.config.ema_slow)
            
            # === 롱 신호 분석 ===
            # OR 상단 돌파 확인
            or_breakout_long = current_high > opening_range['high']
            
            if or_breakout_long:
                # 추세 조건 확인 (롱)
                trend_bullish = (ema_fast.iloc[-1] > ema_slow.iloc[-1] and 
                                current_price > session_vwap)
                
                print(f"   🔍 롱 신호 분석: OR 돌파 ✅, 추세조건 {'✅' if trend_bullish else '❌'}")
                
                if trend_bullish:
                    # 롱 신호 로직
                    long_signal = self._analyze_long_pullback(df, session_vwap, opening_range, atr, 'high')
                    if long_signal:
                        return long_signal
                    else:
                        print(f"   🔍 롱 풀백 조건 불만족")
                else:
                    print(f"   🔍 롱 추세 조건 불만족")
            
            # === 숏 신호 분석 ===
            # OR 하단 이탈 확인
            or_breakdown_short = current_low < opening_range['low']
            
            if or_breakdown_short:
                # 추세 조건 확인 (숏)
                trend_bearish = (ema_fast.iloc[-1] < ema_slow.iloc[-1] and 
                                current_price < session_vwap)
                
                if trend_bearish:
                    # 숏 신호 로직
                    short_signal = self._analyze_short_pullback(df, session_vwap, opening_range, atr, 'low')
                    if short_signal:
                        return short_signal
            
            return None
            
        except Exception as e:
            print(f"❌ 오프닝 드라이브 풀백 분석 오류: {e}")
            return None
    
    def _analyze_long_pullback(self, df: pd.DataFrame, session_vwap: float, 
                               opening_range: Dict[str, float], atr: float, 
                               breakout_level: str) -> Optional[Dict]:
        """롱 풀백 분석"""
        try:
            current_price = df['close'].iloc[-1]
            current_high = df['high'].iloc[-1]
            
            # OR 돌파 이후의 고점 찾기 (정확한 구간 산정)
            or_breakout_mask = df['high'] > opening_range['high']
            if not or_breakout_mask.any():
                return None
            
            # OR 돌파 이후 데이터만 필터링
            post_breakout_df = df[or_breakout_mask]
            if post_breakout_df.empty:
                return None
            
            # OR 돌파 이후의 최고점
            drive_high = post_breakout_df['high'].max()
            drive_high_idx = post_breakout_df['high'].idxmax()
            
            # drive_R 재정의: ORH와 EMA/VWAP/ATR 기반
            orh = opening_range['high']
            ema_slow = calculate_ema(df['close'], self.config.ema_slow)
            current_ema = ema_slow.iloc[-1]
            current_vwap = session_vwap
            
            # ORH 대비 진행거리와 EMA/VWAP 대비 진행거리 중 큰 값 사용
            drive_return_orh = (drive_high - orh) / atr
            drive_return_ema = (drive_high - current_ema) / atr if drive_high > current_ema else 0
            drive_return_vwap = (drive_high - current_vwap) / atr if drive_high > current_vwap else 0
            
            drive_return = max(drive_return_orh, drive_return_ema, drive_return_vwap)
            
            # 최소 진행 확인
            print(f"      📊 진행거리: {drive_return:.2f}R (최소 {self.config.min_drive_return_R}R 필요)")
            if drive_return < self.config.min_drive_return_R:
                print(f"      ❌ 최소 진행거리 부족")
                return None
            
            # 풀백 확인: 고점 이후의 저점 찾기
            post_high_mask = df.index > drive_high_idx
            if not post_high_mask.any():
                return None
            
            post_high_df = df[post_high_mask]
            pullback_low = post_high_df['low'].min()
            
            if pd.isna(pullback_low):
                print(f"      ❌ 풀백 데이터 없음")
                return None
            
            pullback_depth = (drive_high - pullback_low) / atr
            print(f"      📊 풀백 깊이: {pullback_depth:.2f}R (허용범위: {self.config.pullback_depth_atr[0]}-{self.config.pullback_depth_atr[1]}R)")
            
            # 풀백 깊이 범위 확인
            if not (self.config.pullback_depth_atr[0] <= pullback_depth <= self.config.pullback_depth_atr[1]):
                print(f"      ❌ 풀백 깊이 범위 초과")
                return None
            
            # EMA20 또는 VWAP 터치 확인 (0.3×ATR 버퍼)
            ema_slow = calculate_ema(df['close'], self.config.ema_slow)
            touch_buffer = atr * 0.3  # 풀백 터치 버퍼: 0.3×ATR
            
            ema_touch = abs(pullback_low - ema_slow.iloc[-1]) <= touch_buffer
            vwap_touch = abs(pullback_low - session_vwap) <= touch_buffer
            
            print(f"      📊 풀백 저점: ${pullback_low:.2f}")
            print(f"      📊 EMA20: ${ema_slow.iloc[-1]:.2f}")
            print(f"      📊 VWAP: ${session_vwap:.2f}")
            print(f"      📊 EMA20 터치: {'✅' if ema_touch else '❌'}, VWAP 터치: {'✅' if vwap_touch else '❌'}")
            
            # SETUP: ±0.3 ATR "근접"도 인정, ENTRY: 터치/재진입 가점
            if not (ema_touch or vwap_touch):
                print(f"      ❌ EMA/VWAP 터치 조건 불만족")
                return None
            
            # 트리거 확인: 종가 복귀 또는 다음 봉 고점 돌파
            if self.config.trigger_type == "close_reject":
                # 종가가 EMA20 위로 회복
                trigger = (df['close'].iloc[-1] > ema_slow.iloc[-1] and 
                          df['close'].iloc[-1] > pullback_low)
            else:  # wick_touch
                # 저가가 EMA20 터치
                trigger = abs(pullback_low - ema_slow.iloc[-1]) <= atr * 0.1
            
            # 추가 트리거: 다음 봉 고점 돌파
            if len(df) >= 2:
                next_bar_high_breakout = df['high'].iloc[-1] > df['high'].iloc[-2]
                trigger = trigger or next_bar_high_breakout
            
            # 트리거 전: 직전 스윙 무효화 체크 (HH/LL 실패)
            if len(df) >= 4:
                recent_high = df['high'].iloc[-4:-1].max()
                swing_fail = df['high'].iloc[-1] > recent_high   # 고점 갱신으로 리버설 확인
                trigger = trigger and swing_fail
            
            if not trigger:
                return None
            
            # 슬리피지 계산
            slippage = abs(current_price - pullback_low) / current_price
            if slippage > self.config.max_slippage_pct:
                return None
            
            # 롱 신호 생성
            entry_price = current_price
            stop_loss = min(pullback_low, ema_slow.iloc[-1]) - atr * 0.3  # 0.3×ATR
            risk = entry_price - stop_loss
            tp1 = entry_price + risk * self.config.tp1_R
            
            # 2차 목표 계산
            if "OR_ext" in self.config.tp2_to_level:
                tp2 = opening_range['high'] + (opening_range['high'] - opening_range['low'])
            elif "PrevHigh" in self.config.tp2_to_level:
                tp2 = drive_high
            else:  # VWAP
                tp2 = session_vwap
            
            return {
                'signal_type': 'OPENING_DRIVE_PULLBACK_LONG',
                'action': 'BUY',
                'confidence': 0.85,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit1': tp1,
                'take_profit2': tp2,
                'risk_reward': self.config.tp1_R,
                'timestamp': datetime.now(),
                'timestamp': self.time_manager.get_current_time(),
                'reason': f"OR 상단 돌파 후 풀백 롱 | 진행: {drive_return:.1f}ATR, 풀백: {pullback_depth:.1f}ATR",
                'playbook': 'A',
                'partial_out': self.config.partial_out,
                'max_hold_min': self.config.max_hold_min,
                'partial_1': 0.4,               # TP1에서 40% 청산
                'trail_after_tp1_atr_mult': 1.0,# 남은 60% ATR*1.0 트레일
                'hard_timeout_min': 240         # 4시간 초과 보유 금지(단타)
            }
            
        except Exception as e:
            print(f"❌ 롱 풀백 분석 오류: {e}")
            return None
    
    def _analyze_short_pullback(self, df: pd.DataFrame, session_vwap: float, 
                                opening_range: Dict[str, float], atr: float, 
                                breakdown_level: str) -> Optional[Dict]:
        """숏 풀백 분석"""
        try:
            current_price = df['close'].iloc[-1]
            current_low = df['low'].iloc[-1]
            
            # OR 이탈 이후의 저점 찾기 (정확한 구간 산정)
            or_breakdown_mask = df['low'] < opening_range['low']
            if not or_breakdown_mask.any():
                return None
            
            # OR 이탈 이후 데이터만 필터링
            post_breakdown_df = df[or_breakdown_mask]
            if post_breakdown_df.empty:
                return None
            
            # OR 이탈 이후의 최저점
            drive_low = post_breakdown_df['low'].min()
            drive_low_idx = post_breakdown_df['low'].idxmin()
            
            drive_return = (opening_range['low'] - drive_low) / atr
            
            # 최소 진행 확인
            if drive_return < self.config.min_drive_return_R:
                return None
            
            # 되돌림 확인: 저점 이후의 고점 찾기
            post_low_mask = df.index > drive_low_idx
            if not post_low_mask.any():
                return None
            
            post_low_df = df[post_low_mask]
            pullback_high = post_low_df['high'].max()
            
            if pd.isna(pullback_high):
                return None
            
            pullback_depth = (pullback_high - drive_low) / atr
            
            # 되돌림 깊이 범위 확인
            if not (self.config.pullback_depth_atr[0] <= pullback_depth <= self.config.pullback_depth_atr[1]):
                return None
            
            # EMA20 또는 VWAP 터치 확인
            ema_slow = calculate_ema(df['close'], self.config.ema_slow)
            ema_touch = pullback_high >= ema_slow.iloc[-1]
            vwap_touch = abs(pullback_high - session_vwap) <= atr * 0.5
            
            if not (ema_touch or vwap_touch):
                return None
            
            # 트리거 확인: 종가 복귀 또는 다음 봉 저점 돌파
            if self.config.trigger_type == "close_reject":
                # 종가가 EMA20 아래로 회복
                trigger = (df['close'].iloc[-1] < ema_slow.iloc[-1] and 
                          df['close'].iloc[-1] < pullback_high)
            else:  # wick_touch
                # 고가가 EMA20 터치
                trigger = abs(pullback_high - ema_slow.iloc[-1]) <= atr * 0.1
            
            # 추가 트리거: 다음 봉 저점 돌파
            if len(df) >= 2:
                next_bar_low_breakdown = df['low'].iloc[-1] < df['low'].iloc[-2]
                trigger = trigger or next_bar_low_breakdown
            
            # 트리거 전: 직전 스윙 무효화 체크 (HH/LL 실패)
            if len(df) >= 4:
                recent_low = df['low'].iloc[-4:-1].min()
                swing_fail = df['low'].iloc[-1] < recent_low   # 저점 갱신으로 리버설 확인
                trigger = trigger and swing_fail
            
            if not trigger:
                return None
            
            # 슬리피지 계산
            slippage = abs(current_price - pullback_high) / current_price
            if slippage > self.config.max_slippage_pct:
                return None
            
            # 숏 신호 생성
            entry_price = current_price
            stop_loss = max(pullback_high, ema_slow.iloc[-1]) + atr * 0.3  # 0.3×ATR
            risk = stop_loss - entry_price
            tp1 = entry_price - risk * self.config.tp1_R
            
            # 2차 목표 계산
            if "OR_ext" in self.config.tp2_to_level:
                tp2 = opening_range['low'] - (opening_range['high'] - opening_range['low'])
            elif "PrevLow" in self.config.tp2_to_level:
                tp2 = drive_low
            else:  # VWAP
                tp2 = entry_price - risk * 2.5 # 기본값
            
            return {
                'signal_type': 'OPENING_DRIVE_PULLBACK_SHORT',
                'action': 'SELL',
                'confidence': 0.85,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit1': tp1,
                'take_profit2': tp2,
                'risk_reward': self.config.tp1_R,
                'timestamp': datetime.now(),
                'timestamp': self.time_manager.get_current_time(),
                'reason': f"OR 하단 이탈 후 되돌림 숏 | 진행: {drive_return:.1f}ATR, 되돌림: {pullback_depth:.1f}ATR",
                'playbook': 'A',
                'partial_out': self.config.partial_out,
                'max_hold_min': self.config.max_hold_min,
                'partial_1': 0.4,               # TP1에서 40% 청산
                'trail_after_tp1_atr_mult': 1.0,# 남은 60% ATR*1.0 트레일
                'hard_timeout_min': 240         # 4시간 초과 보유 금지(단타)
            }
            
        except Exception as e:
            print(f"❌ 숏 풀백 분석 오류: {e}")
            return None
    
    def analyze_playbook_b_liquidity_sweep_reclaim(self, df: pd.DataFrame,
                                                    key_levels: Dict[str, float],
                                                    atr: float) -> Optional[Dict]:
        """플레이북 B: 유동성 스윕 & 리클레임 분석 (롱/숏)"""
        if len(df) < 10:
            return None
        
        try:
            current_price = df['close'].iloc[-1]
            current_low = df['low'].iloc[-1]
            current_high = df['high'].iloc[-1]
            
            # === 롱 신호 분석 ===
            # 전일 저가 스윕 확인
            prev_day_low = key_levels.get('prev_day_low', 0)
            if prev_day_low > 0:
                # 스윕 확인
                sweep_long = current_low < prev_day_low
                sweep_depth_long = (prev_day_low - current_low) / atr
                
                if sweep_long and sweep_depth_long >= self.config.sweep_depth_atr_min:
                    # 리클레임 확인
                    reclaim_long = current_price > prev_day_low
                    
                    if reclaim_long:
                        # 리클레임 확증: 종가 재돌파=ENTRY 가점, 레벨 ±0.5 ATR "근접"=SETUP 허용
                        if len(df) >= 2:
                            next_bar_low = df['low'].iloc[-2]
                            next_bar_close = df['close'].iloc[-2]
                            # ENTRY: 종가 재돌파 확인
                            reclaim_confirmed = (next_bar_low >= prev_day_low and next_bar_close >= prev_day_low)
                            # SETUP: 레벨 ±0.5 ATR "근접" 허용
                            reclaim_setup = abs(next_bar_close - prev_day_low) <= atr * 0.5
                            
                            if not (reclaim_confirmed or reclaim_setup):
                                return None
                        
                        # 롱 신호 생성
                        entry_price = current_price
                        stop_loss = current_low - atr * 0.6  # 0.6×ATR
                        risk = entry_price - stop_loss
                        
                        # 1차 목표 (VWAP)
                        if self.config.tp1_to_b == "VWAP":
                            tp1 = self.session_vwap if self.session_vwap else entry_price + risk * 1.5
                        else:
                            tp1 = entry_price + risk * 1.5
                        
                        # 2차 목표
                        if self.config.tp2_to_b == "opposite_range_edge":
                            prev_day_high = key_levels.get('prev_day_high', 0)
                            if prev_day_high > 0:
                                tp2 = prev_day_high
                            else:
                                tp2 = entry_price + risk * 2.5
                        else:
                            tp2 = entry_price + risk * 2.5
                        
                        return {
                            'signal_type': 'LIQUIDITY_SWEEP_RECLAIM_LONG',
                            'action': 'BUY',
                            'confidence': 0.80,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit1': tp1,
                            'take_profit2': tp2,
                            'risk_reward': 1.5,
                            'timestamp': datetime.now(),
                            'timestamp': self.time_manager.get_current_time(),
                            'reason': f"전일저가 스윕 후 리클레임 롱 | 스윕깊이: {sweep_depth_long:.1f}ATR",
                            'playbook': 'B',
                            'partial_out': self.config.partial_out,
                            'max_hold_min': 45,
                            'partial_1': 0.4,               # TP1에서 40% 청산
                            'trail_after_tp1_atr_mult': 1.0,# 남은 60% ATR*1.0 트레일
                            'hard_timeout_min': 240         # 4시간 초과 보유 금지(단타)
                        }
            
            # === 숏 신호 분석 ===
            # 전일 고가 스윕 확인
            prev_day_high = key_levels.get('prev_day_high', 0)
            if prev_day_high > 0:
                # 스윕 확인
                sweep_short = current_high > prev_day_high
                sweep_depth_short = (current_high - prev_day_high) / atr
                
                if sweep_short and sweep_depth_short >= self.config.sweep_depth_atr_min:
                    # 리클레임 확인
                    reclaim_short = current_price < prev_day_high
                    
                    if reclaim_short:
                        # 리클레임 확증: 종가 재돌파=ENTRY 가점, 레벨 ±0.5 ATR "근접"=SETUP 허용
                        if len(df) >= 2:
                            next_bar_low = df['low'].iloc[-2]
                            next_bar_close = df['close'].iloc[-2]
                            # ENTRY: 종가 재돌파 확인
                            reclaim_confirmed = (next_bar_low <= prev_day_high and next_bar_close <= prev_day_high)
                            # SETUP: 레벨 ±0.5 ATR "근접" 허용
                            reclaim_setup = abs(next_bar_close - prev_day_high) <= atr * 0.5
                            
                            if not (reclaim_confirmed or reclaim_setup):
                                return None
                        
                        # 숏 신호 생성
                        entry_price = current_price
                        stop_loss = current_high + atr * 0.6  # 0.6×ATR
                        risk = stop_loss - entry_price
                        
                        # 1차 목표 (VWAP)
                        if self.config.tp1_to_b == "VWAP":
                            tp1 = self.session_vwap if self.session_vwap else entry_price - risk * 1.5
                        else:
                            tp1 = entry_price - risk * 1.5
                        
                        # 2차 목표
                        if self.config.tp2_to_b == "opposite_range_edge":
                            prev_day_low = key_levels.get('prev_day_low', 0)
                            if prev_day_low > 0:
                                tp2 = prev_day_low
                            else:
                                tp2 = entry_price - risk * 2.5
                        else:
                            tp2 = entry_price - risk * 2.5
                        
                        return {
                            'signal_type': 'LIQUIDITY_SWEEP_RECLAIM_SHORT',
                            'action': 'SELL',
                            'confidence': 0.80,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit1': tp1,
                            'take_profit2': tp2,
                            'risk_reward': 1.5,
                            'timestamp': datetime.now(),
                            'timestamp': self.time_manager.get_current_time(),
                            'reason': f"전일고가 스윕 후 리클레임 숏 | 스윕깊이: {sweep_depth_short:.1f}ATR",
                            'playbook': 'B',
                            'partial_out': self.config.partial_out,
                            'max_hold_min': 45,
                            'partial_1': 0.4,               # TP1에서 40% 청산
                            'trail_after_tp1_atr_mult': 1.0,# 남은 60% ATR*1.0 트레일
                            'hard_timeout_min': 240         # 4시간 초과 보유 금지(단타)
                        }
            
            return None
            
        except Exception as e:
            print(f"❌ 유동성 스윕 리클레임 분석 오류: {e}")
            return None
    
    def analyze_playbook_c_vwap_reversion_fade(self, df: pd.DataFrame,
                                                session_vwap: float,
                                                session_std: float,
                                                atr: float) -> Optional[Dict]:
        """플레이북 C: VWAP 리버전(평균회귀) 페이드 분석 (롱/숏)"""
        if len(df) < 3 or session_std == 0:
            return None
        
        try:
            # === 롱 신호 분석 ===
            # 추세 필터 확인
            if self.config.trend_filter_ma > 0:
                sma_trend = calculate_ema(df['close'], self.config.trend_filter_ma)
                if len(sma_trend) >= 2:
                    trend_slope = (sma_trend.iloc[-1] - sma_trend.iloc[-2]) / sma_trend.iloc[-2]
                    
                    # 강한 하락 추세일 때 롱 페이드 금지
                    if trend_slope < self.config.trend_filter_slope:
                        return None
                    
                    # 추세 기울기 절대값이 trend_filter_slope보다 크면 페이드 신호 자체 비활성화
                    if abs(trend_slope) > self.config.trend_filter_slope:
                        return None
            
            # t봉 종가가 VWAP-2σ 아래에서 마감
            t_bar_close = df['close'].iloc[-2]
            t_bar_low = df['low'].iloc[-2]
            
            oversold_trigger = t_bar_close < (session_vwap - self.config.sd_k_enter * session_std)
            
            if oversold_trigger:
                # t+1봉 종가가 -1.5σ 안쪽으로 재진입
                t_plus_1_close = df['close'].iloc[-1]
                reenter_trigger = t_plus_1_close > (session_vwap - self.config.sd_k_reenter * session_std)
                
                if reenter_trigger:
                    # 롱 신호 생성
                    entry_price = t_plus_1_close
                    stop_loss = session_vwap - self.config.stop_outside_sd_k * session_std
                    
                    # t봉 저점이 더 낮으면 그걸로 스탑
                    if t_bar_low < stop_loss:
                        stop_loss = t_bar_low
                    
                    risk = entry_price - stop_loss
                    
                    # 1차 목표: VWAP 터치
                    if self.config.tp1_to_c == "VWAP":
                        tp1 = session_vwap
                    else:
                        tp1 = entry_price + risk * 1.2  # 기본값
                    
                    # 2차 목표: 반대측 +0.5σ
                    tp2 = session_vwap + self.config.tp2_to_c * session_std
                    
                    return {
                        'signal_type': 'VWAP_REVERSION_FADE_LONG',
                        'action': 'BUY',
                        'confidence': 0.75,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit1': tp1,
                        'take_profit2': tp2,
                        'risk_reward': 1.2,
                        'timestamp': datetime.now(),
                        'timestamp': self.time_manager.get_current_time(),
                        'reason': f"VWAP 과매도 페이드 롱 | 진입: -{self.config.sd_k_enter}σ, 재진입: -{self.config.sd_k_reenter}σ",
                        'playbook': 'C',
                        'partial_out': self.config.partial_out,
                        'max_hold_min': 30,
                        'partial_1': 0.4,               # TP1에서 40% 청산
                        'trail_after_tp1_atr_mult': 1.0,# 남은 60% ATR*1.0 트레일
                        'hard_timeout_min': 240         # 4시간 초과 보유 금지(단타)
                    }
            
            # === 숏 신호 분석 ===
            # 추세 필터 확인 (숏)
            if self.config.trend_filter_ma > 0:
                sma_trend = calculate_ema(df['close'], self.config.trend_filter_ma)
                if len(sma_trend) >= 2:
                    trend_slope = (sma_trend.iloc[-1] - sma_trend.iloc[-2]) / sma_trend.iloc[-2]
                    
                    # 강한 상승 추세일 때 숏 페이드 금지
                    if trend_slope > -self.config.trend_filter_slope:
                        return None
                    
                    # 추세 기울기 절대값이 trend_filter_slope보다 크면 페이드 신호 자체 비활성화
                    if abs(trend_slope) > self.config.trend_filter_slope:
                        return None
            
            # t봉 종가가 VWAP+2σ 위에서 마감
            t_bar_close = df['close'].iloc[-2]
            t_bar_high = df['high'].iloc[-2]
            
            overbought_trigger = t_bar_close > (session_vwap + self.config.sd_k_enter * session_std)
            
            if overbought_trigger:
                # t+1봉 종가가 +1.5σ 안쪽으로 재진입
                t_plus_1_close = df['close'].iloc[-1]
                reenter_trigger = t_plus_1_close < (session_vwap + self.config.sd_k_reenter * session_std)
                
                if reenter_trigger:
                    # 숏 신호 생성
                    entry_price = t_plus_1_close
                    stop_loss = session_vwap + self.config.stop_outside_sd_k * session_std
                    
                    # t봉 고점이 더 높으면 그걸로 스탑
                    if t_bar_high > stop_loss:
                        stop_loss = t_bar_high
                    
                    risk = stop_loss - entry_price
                    
                    # 1차 목표: VWAP 터치
                    if self.config.tp1_to_c == "VWAP":
                        tp1 = session_vwap
                    else:
                        tp1 = entry_price - risk * 1.2  # 기본값
                    
                    # 2차 목표: 반대측 -0.5σ
                    tp2 = session_vwap - self.config.tp2_to_c * session_std
                    
                    return {
                        'signal_type': 'VWAP_REVERSION_FADE_SHORT',
                        'action': 'SELL',
                        'confidence': 0.75,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit1': tp1,
                        'take_profit2': tp2,
                        'risk_reward': 1.2,
                        'timestamp': datetime.now(),
                        'timestamp': self.time_manager.get_current_time(),
                        'reason': f"VWAP 과매수 페이드 숏 | 진입: +{self.config.sd_k_enter}σ, 재진입: +{self.config.sd_k_reenter}σ",
                        'playbook': 'C',
                        'partial_out': self.config.partial_out,
                        'max_hold_min': 30,
                        'partial_1': 0.4,               # TP1에서 40% 청산
                        'trail_after_tp1_atr_mult': 1.0,# 남은 60% ATR*1.0 트레일
                        'hard_timeout_min': 240         # 4시간 초과 보유 금지(단타)
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ VWAP 리버전 페이드 분석 오류: {e}")
            return None
    
    def analyze_session_strategy(self, df: pd.DataFrame, 
                                key_levels: Dict[str, float],
                                current_time: datetime) -> Optional[Dict]:
        """세션 기반 전략 통합 분석 (단계형 신호 적용)"""
            # current_time을 UTC timezone으로 변환
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=pytz.UTC)
        
        # 세션 시작 시간 확인 (이미 UTC tz-aware)
        session_start = self.get_session_start_time(current_time)
                
        session_vwap, session_std = get_vwap()
        or_info = get_opening_range()
        atr = get_atr()
        
        # 인스턴스 변수 업데이트
        self.session_vwap = session_vwap
        self.session_std = session_std
        self.opening_range = or_info
        self.session_start_time = session_start  # 세션 시작 시간 저장

        # --- 세션 정보 출력 (간단하게) ---
        if atr <= 0:
            return None
        
        # === 단계형 신호 분석 ===
        best_signal = None
        best_score = 0.0
        
        # 세션 데이터 슬라이스 (글로벌 지표 사용 시에도 필요)
        df_s = self._session_slice(df, session_start)
        
        # A: OR가 없거나(strict) 준비 안 됐으면 스킵 또는 티어 제한
        if or_info:
            for side in ["LONG","SHORT"]:
                sig = self.analyze_staged_signal(df_s, session_vwap, or_info, atr, 'A', side, key_levels, current_time)
                # 부분 OR이면 티어 캡 적용
                if sig and or_info.get("partial"):
                    tier_cap = self.config.partial_or_tier_cap.upper()
                    if tier_cap == "SETUP" and sig["stage"] == "ENTRY":
                        sig["stage"] = "SETUP"; sig["action"] = "OBSERVE"; sig["confidence"] *= 0.9
                    elif tier_cap == "HEADSUP" and sig["stage"] in ("ENTRY","SETUP"):
                        sig["stage"] = "HEADSUP"; sig["action"] = "ALERT"; sig["confidence"] *= 0.8
                if sig and sig["score"] > best_score:
                    best_signal, best_score = sig, sig["score"]
        else:
            print("⏭️ Play A 스킵")
        
        # B/C는 OR 없어도 정상 동작
        for side in ["LONG","SHORT"]:
            sig = self.analyze_staged_signal(df_s, session_vwap, or_info or {}, atr, 'B', side, key_levels, current_time)
            if sig and sig["score"] > best_score:
                best_signal, best_score = sig, sig["score"]

        if np.isfinite(session_vwap) and np.isfinite(session_std) and session_std > 0:
            for side in ["LONG","SHORT"]:
                sig = self.analyze_staged_signal(df_s, session_vwap, or_info or {}, atr, 'C', side, key_levels, current_time)
                if sig and sig["score"] > best_score:
                    best_signal, best_score = sig, sig["score"]
        
        # 최고 점수 신호 반환
        if best_signal:
            return best_signal
        
        return None

def make_session_trade_plan(df: pd.DataFrame, 
                            key_levels: Dict[str, float],
                            config: SessionConfig,
                            current_time: datetime) -> Optional[Dict]:
    """세션 기반 거래 계획 생성"""
    try:
        strategy = SessionBasedStrategy(config)
        signal = strategy.analyze_session_strategy(df, key_levels, current_time)
        
        if signal:
            # ENTRY에만 포지션 사이징 적용
            if signal.get("stage") == "ENTRY" and {"entry_price", "stop_loss"} <= signal.keys():
                risk_percent = 0.4   # 계좌 리스크 0.4%
                equity = 100000       # 예시 자본
                risk_dollar = equity * risk_percent / 100
                stop_distance = abs(signal["entry_price"] - signal["stop_loss"])
                position_size = risk_dollar / stop_distance if stop_distance > 0 else 0
                signal["position_size"] = position_size
                signal["risk_dollar"] = risk_dollar
            
            return signal
        
        return None
        
    except Exception as e:
        print(f"❌ 세션 거래 계획 생성 오류: {e}")
        return None
