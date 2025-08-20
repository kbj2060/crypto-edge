#!/usr/bin/env python3
"""
세션 기반 고급 전략 (Session-Based Advanced Strategy)
- 플레이북 A: 오프닝 드라이브 풀백 매수/매도
- 플레이북 B: 유동성 스윕 & 리클레임
- 플레이북 C: VWAP 리버전(평균회귀) 페이드
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from datetime import datetime, timedelta
import pytz
from indicators.vpvr import vpvr_key_levels
from indicators.moving_averages import calculate_ema
from indicators.atr import calculate_atr


@dataclass
class SessionConfig:
    """세션 기반 전략 설정"""
    # 기본 설정
    symbol: str = "ETHUSDT"
    timeframe: str = "1m"
    
    # 세션 설정
    ses_vwap_start_utc: str = "13:30 UTC"  # NY Open (KST 22:30, DST중)
    london_session_start_utc: str = "07:00 UTC"  # London Open (KST 16:00)
    or_minutes: int = 15  # 오프닝 레인지 분
    
    # 지표 설정
    ema_fast: int = 9
    ema_slow: int = 20
    atr_len: int = 14
    trend_filter_ma: int = 50
    
    # 플레이북 A: 오프닝 드라이브 풀백
    min_drive_return_R: float = 0.8  # OR 돌파 후 최소 0.8R 이상 진행 (ORH와 EMA/VWAP/ATR 기반)
    pullback_depth_atr: Tuple[float, float] = (0.6, 1.4)  # 풀백 깊이(ATR배) 허용 범위
    trigger_type: str = "close_reject"  # 'close_reject' 또는 'wick_touch'
    stop_atr_mult: float = 1.1  # 스탑 = 엔트리 기준 무효화/스윙 아래 + 1.1×ATR
    tp1_R: float = 1.5  # 1차 청산 R
    tp2_to_level: str = "OR_ext|PrevHigh|VWAP"  # 2차 목표 우선순위
    partial_out: float = 0.5  # 1차에서 절반 청산
    max_hold_min: int = 60  # 최대 보유시간(분)
    max_slippage_pct: float = 0.02  # 허용 슬리피지(%) 초과 시 신호 무효
    
    # 플레이북 B: 유동성 스윕 & 리클레임
    sweep_depth_atr_min: float = 0.3  # 레벨 하회/상회 최소 깊이(ATR배)
    reclaim_close_rule: str = "close_above_level"  # 롱: 레벨 위 종가 마감
    confirm_next_bar: bool = True  # 다음 봉이 레벨 위에서 지속 확인
    stop_buffer_atr: float = 0.6  # 스탑 버퍼
    tp1_to: str = "VWAP"  # 1차 목표
    tp2_to: str = "opposite_range_edge"  # 2차 목표
    
    # 플레이북 C: VWAP 리버전(평균회귀) 페이드
    sd_k_enter: float = 2.0  # 진입 트리거: 봉 종가가 ±2σ 밖에서 마감
    sd_k_reenter: float = 1.5  # 그 다음 봉 종가가 ±1.5σ 안쪽으로 재진입
    stop_outside_sd_k: float = 2.5  # 스탑: ±2.5σ 바깥
    tp1_to: str = "VWAP"  # 1차 목표: VWAP 터치
    tp2_to_band: float = 0.5  # 2차: 반대측 0.5σ
    trend_filter_slope: float = 0.0  # SMA50 기울기 > 0.0이면 숏페이드 보수적
    
    # 단계형 신호 설정
    entry_thresh: float = 0.70  # Entry 임계점
    setup_thresh: float = 0.50  # Setup 임계점
    headsup_thresh: float = 0.35  # Heads-up 임계점
    
    # Gate 설정
    min_sweep_depth_atr: float = 0.2  # 최소 스윕 깊이 (Play B)
    max_slippage_gate: float = 0.03  # 최대 허용 슬리피지 (Gate)
    min_volume_ratio: float = 0.7  # 최소 거래량 비율 (Gate)
    
    # Score 가중치
    weight_direction: float = 0.25  # 방향 정렬
    weight_breakout_sweep: float = 0.20  # 돌파/스윕 질
    weight_pullback: float = 0.15  # 풀백 품질
    weight_baseline: float = 0.10  # 기준선 근접/복귀
    weight_timing: float = 0.10  # 세션 타이밍
    weight_orderflow: float = 0.20  # 오더플로우
    weight_risk: float = 0.10  # 리스크 적정성
    
    # --- 설정 추가 ---
    strict_or: bool = True        # True면 OR 확정 전 Play A 완전 비활성
    min_or_bars: int = 15          # 부분 OR 최소 봉 수
    partial_or_tier_cap: str = "SETUP"  # 부분 OR일 때 최대 티어: "HEADSUP"|"SETUP"|"ENTRY"


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
        
    def calculate_session_vwap(
        self, df: pd.DataFrame, session_start: datetime, session_end: datetime
    ) -> Tuple[float, float]:
        """세션 구간 VWAP 및 표준편차 계산 (반개구간 [start, end), 누적 σ)"""
        if df.empty:
            return np.nan, np.nan
        # 안전장치: tz-aware & 정렬
        assert df.index.tzinfo is not None, "df.index must be tz-aware(UTC)"
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
        """세션 시작부터 현재까지의 데이터 슬라이스"""
        if df.empty:
            return df
        assert df.index.tzinfo is not None, "df.index must be tz-aware(UTC)"
        df = df.sort_index()
        
        # 세션 시작부터 현재까지의 데이터만 필터링
        mask = df.index >= session_start
        return df.loc[mask]
    
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
                if event.get('timestamp', current_time) >= one_hour_ago
            ]
            
            if not recent_events:
                return {}
            
            # SELL/BUY→롱/숏 청산 매핑 고정
            long_liquidations = [e for e in recent_events if e.get('side') == 'SELL']
            short_liquidations = [e for e in recent_events if e.get('side') == 'BUY']
            
            # 누적 청산량 계산
            long_volume = sum(e.get('size', 0) for e in long_liquidations)
            short_volume = sum(e.get('size', 0) for e in short_liquidations)
            
            # 청산 강도 계산 (LPI 기반)
            long_intensity = np.mean([e.get('lpi', 0) for e in long_liquidations]) if long_liquidations else 0
            short_intensity = np.mean([e.get('lpi', 0) for e in short_liquidations]) if short_liquidations else 0
            
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
    
    def check_gates(self, df: pd.DataFrame, session_vwap: float, 
                    opening_range: Dict[str, float], atr: float, 
                    playbook: str, side: str) -> Tuple[bool, Dict[str, Any]]:
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
                # opening_range가 유효한지 확인
                if not opening_range or 'high' not in opening_range or 'low' not in opening_range:
                    structure_gate = False
                else:
                    # 키 레벨 스윕 확인 (간단한 구현)
                    if side == 'LONG':
                        sweep_depth = (opening_range['low'] - current_low) / atr if atr > 0 else 0
                        structure_gate = sweep_depth >= self.config.min_sweep_depth_atr
                    else:
                        sweep_depth = (current_high - opening_range['high']) / atr if atr > 0 else 0
                        structure_gate = sweep_depth >= self.config.min_sweep_depth_atr
            elif playbook == 'C':  # VWAP 리버전 페이드
                if side == 'LONG':
                    structure_gate = current_price < (session_vwap - 2 * self.session_std)
                else:
                    structure_gate = current_price > (session_vwap + 2 * self.session_std)
            
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
                        playbook: str, side: str, gate_results: Dict[str, Any]) -> float:
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
                sweep_depth = gate_results.get('slippage_value', 0) * 100  # %를 ATR로 변환
                breakout_score = min(sweep_depth / atr, 1.0) * self.config.weight_breakout_sweep if atr > 0 else 0
            else:  # Play C
                breakout_score = 0.15 * self.config.weight_breakout_sweep  # 기본값
            
            score += breakout_score
            
            # === 풀백 품질 (0.15) ===
            if playbook == 'A':
                # 풀백 깊이 계산 (간단한 구현)
                pullback_depth = 0.8  # 기본값, 실제로는 계산 필요
                # 가우시안 스코어: 0.4~1.6×ATR 범위에 가까울수록 가점
                optimal_depth = 1.0
                depth_score = np.exp(-((pullback_depth - optimal_depth) ** 2) / 0.5)
                pullback_score = depth_score * self.config.weight_pullback
            else:
                pullback_score = 0.1 * self.config.weight_pullback  # 기본값
            
            score += pullback_score
            
            # === 기준선 근접/복귀 (0.10) ===
            if side == 'LONG':
                ema_touch = abs(current_low - ema_slow.iloc[-1]) <= atr * 0.3
                vwap_touch = abs(current_low - session_vwap) <= atr * 0.3
            else:
                ema_touch = abs(current_high - ema_slow.iloc[-1]) <= atr * 0.3
                vwap_touch = abs(current_high - session_vwap) <= atr * 0.3
            
            baseline_score = (ema_touch or vwap_touch) * self.config.weight_baseline
            score += baseline_score
            
            # === 세션 타이밍 (0.10) ===
            # 간단한 구현: 현재 시간이 세션 시작 ±90분 내인지 확인
            timing_score = 0.4  # 기본값, 실제로는 계산 필요
            score += timing_score * self.config.weight_timing
            
            # === 오더플로우 (0.20) ===
            # 간단한 구현: 기본값 사용
            orderflow_score = 0.15 * self.config.weight_orderflow
            score += orderflow_score
            
            # === 리스크 적정성 (0.10) ===
            # 스탑 거리 계산 (간단한 구현)
            stop_distance = atr * 1.0  # 기본값
            risk_score = 0.0
            if 0.6 <= stop_distance / atr <= 1.6:
                risk_score = 1.0
            elif 0.4 <= stop_distance / atr <= 2.0:
                risk_score = 0.5
            
            risk_score *= self.config.weight_risk
            score += risk_score
            
            return min(score, 1.0)  # 최대 1.0
            
        except Exception as e:
            print(f"❌ Score 계산 오류: {e}")
            return 0.0
    
    def analyze_staged_signal(self, df: pd.DataFrame, session_vwap: float,
                             opening_range: Dict[str, float], atr: float,
                             playbook: str, side: str) -> Optional[Dict[str, Any]]:
        """단계형 신호 분석: Gate → Score → 등급/행동"""
        try:
            # === Gate 확인 ===
            gates_passed, gate_results = self.check_gates(
                df, session_vwap, opening_range, atr, playbook, side
            )
            
            if not gates_passed:
                return None
            
            # === Score 계산 ===
            score = self.calculate_score(
                df, session_vwap, opening_range, atr, playbook, side, gate_results
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
                    'risk_reward': self.config.tp1_R
                })
            
            return signal
            
        except Exception as e:
            print(f"❌ 단계형 신호 분석 오류: {e}")
            return None
    
    def get_session_start_time(self, current_time) -> datetime:
        """가장 최근에 완성된 OR의 세션 시작 시간을 반환"""
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=pytz.UTC)
        
        # 현지시간 기준 세션 시작 시간 (DST 자동 처리)
        ny_tz = pytz.timezone('America/New_York')
        london_tz = pytz.timezone('Europe/London')
        
        # 현재 날짜
        current_date = current_time.date()
        
        # 현지시간 기준으로 세션 시작 시간 생성 (DST 자동 처리)
        ny_session_local = ny_tz.localize(datetime.combine(current_date, datetime.strptime('09:30', '%H:%M').time()))
        london_session_local = london_tz.localize(datetime.combine(current_date, datetime.strptime('08:00', '%H:%M').time()))
        
        # UTC로 변환
        today_ny = ny_session_local.astimezone(pytz.UTC)
        today_london = london_session_local.astimezone(pytz.UTC)
        
        # 어제 세션들
        yesterday = current_date - timedelta(days=1)
        yesterday_ny = ny_tz.localize(datetime.combine(yesterday, datetime.strptime('09:30', '%H:%M').time())).astimezone(pytz.UTC)
        yesterday_london = london_tz.localize(datetime.combine(yesterday, datetime.strptime('08:00', '%H:%M').time())).astimezone(pytz.UTC)
        
        # OR 완성 시간 계산 (15분 후)
        or_duration = timedelta(minutes=self.config.or_minutes)
        
        # 각 세션의 OR 완성 시간
        or_completion_times = [
            (yesterday_ny + or_duration, yesterday_ny, "어제 뉴욕"),
            (yesterday_london + or_duration, yesterday_london, "어제 런던"),
            (today_london + or_duration, today_london, "오늘 런던"),
            (today_ny + or_duration, today_ny, "오늘 뉴욕")
        ]
        
        # 현재 시간보다 이전에 완성된 OR들 중 가장 최근 것 찾기
        completed_ors = [(completion, start, name) for completion, start, name in or_completion_times 
                         if completion <= current_time]
        
        if not completed_ors:
            # 완성된 OR가 없으면 어제 뉴욕 세션 반환
            print(f" 세션 시작 시간: {yesterday_ny.strftime('%Y-%m-%d %H:%M:%S')} UTC (어제 뉴욕 - 기본값)")
            return yesterday_ny
        
        # 가장 최근에 완성된 OR의 세션 시작 시간 반환
        latest_completion, latest_start, latest_name = max(completed_ors, key=lambda x: x[0])
        
        print(f" 세션 시작 시간: {latest_start.strftime('%Y-%m-%d %H:%M:%S')} UTC ({latest_name})")
        print(f"   OR 완성 시간: {latest_completion.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        return latest_start
    
    def _get_session_type(self, session_start: datetime) -> str:
        """세션 시작 시간으로부터 세션 타입 식별"""
        ny_session_time = datetime.strptime('13:30', '%H:%M').time()
        london_session_time = datetime.strptime('07:00', '%H:%M').time()
        
        session_time = session_start.time()
        
        if session_time == ny_session_time:
            return "뉴욕"
        elif session_time == london_session_time:
            return "런던"
        else:
            return "알 수 없음"
    
    def calculate_opening_range(
        self, df: pd.DataFrame, session_start: datetime
    ) -> Dict[str, float]:
        """세션 구간 오프닝 레인지 계산 (반개구간, 정확히 OR 분만)"""
        if df.empty:
            return {}
        assert df.index.tzinfo is not None, "df.index must be tz-aware(UTC)"
        df = df.sort_index()

        or_end = session_start + timedelta(minutes=self.config.or_minutes)
        mask = (df.index >= session_start) & (df.index < or_end)
        head = df.loc[mask]
        bars = len(head)
        if bars == 0:
            return {}

        h = float(head["high"].max())
        l = float(head["low"].min())
        
        # 유효성 검사
        if pd.isna(h) or pd.isna(l) or h <= l:
            print(f"❌ OR 계산 오류: 유효하지 않은 high/low 값 - high: {h}, low: {l}")
            return {}
        
        ready = (bars >= self.config.or_minutes)     # 완전 OR 확보?
        partial = (not ready) and (bars >= self.config.min_or_bars)

        return {
            "high": h, "low": l, "center": (h + l) / 2.0, "range": h - l,
            "bars": bars, "ready": ready, "partial": partial
        }
    
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
            
            # 최근 고점 찾기 (OR 돌파 이후)
            recent_highs = df[df['high'] > opening_range['high']]['high']
            if recent_highs.empty:
                return None
            
            drive_high = recent_highs.max()
            
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
            
            # 풀백 확인
            pullback_low = df[df['high'] >= drive_high]['low'].min()
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
                'reason': f"OR 상단 돌파 후 풀백 롱 | 진행: {drive_return:.1f}ATR, 풀백: {pullback_depth:.1f}ATR",
                'playbook': 'A',
                'partial_out': self.config.partial_out,
                'max_hold_min': self.config.max_hold_min
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
            
            # 최근 저점 찾기 (OR 이탈 이후)
            recent_lows = df[df['low'] < opening_range['low']]['low']
            if recent_lows.empty:
                return None
            
            drive_low = recent_lows.min()
            drive_return = (opening_range['low'] - drive_low) / atr
            
            # 최소 진행 확인
            if drive_return < self.config.min_drive_return_R:
                return None
            
            # 되돌림 확인
            pullback_high = df[df['low'] <= drive_low]['high'].max()
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
                tp2 = session_vwap
            
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
                'reason': f"OR 하단 이탈 후 되돌림 숏 | 진행: {drive_return:.1f}ATR, 되돌림: {pullback_depth:.1f}ATR",
                'playbook': 'A',
                'partial_out': self.config.partial_out,
                'max_hold_min': self.config.max_hold_min
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
                        # 다음 봉 확인 (옵션)
                        if self.config.confirm_next_bar and len(df) >= 2:
                            next_bar_low = df['low'].iloc[-2]
                            next_bar_high = df['high'].iloc[-2]
                            confirm = (next_bar_low > prev_day_low and next_bar_high > prev_day_low)
                            
                            if not confirm:
                                return None
                        
                        # 롱 신호 생성
                        entry_price = current_price
                        stop_loss = current_low - atr * 0.6  # 0.6×ATR
                        risk = entry_price - stop_loss
                        
                        # 1차 목표 (VWAP)
                        if self.config.tp1_to == "VWAP":
                            tp1 = self.session_vwap if self.session_vwap else entry_price + risk * 1.5
                        else:
                            tp1 = entry_price + risk * 1.5
                        
                        # 2차 목표
                        if self.config.tp2_to == "opposite_range_edge":
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
                            'reason': f"전일저가 스윕 후 리클레임 롱 | 스윕깊이: {sweep_depth_long:.1f}ATR",
                            'playbook': 'B',
                            'partial_out': self.config.partial_out,
                            'max_hold_min': 45
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
                        # 다음 봉 확인 (옵션)
                        if self.config.confirm_next_bar and len(df) >= 2:
                            next_bar_low = df['low'].iloc[-2]
                            next_bar_high = df['high'].iloc[-2]
                            confirm = (next_bar_low < prev_day_high and next_bar_high < prev_day_high)
                            
                            if not confirm:
                                return None
                        
                        # 숏 신호 생성
                        entry_price = current_price
                        stop_loss = current_high + atr * 0.6  # 0.6×ATR
                        risk = stop_loss - entry_price
                        
                        # 1차 목표 (VWAP)
                        if self.config.tp1_to == "VWAP":
                            tp1 = self.session_vwap if self.session_vwap else entry_price - risk * 1.5
                        else:
                            tp1 = entry_price - risk * 1.5
                        
                        # 2차 목표
                        if self.config.tp2_to == "opposite_range_edge":
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
                            'reason': f"전일고가 스윕 후 리클레임 숏 | 스윕깊이: {sweep_depth_short:.1f}ATR",
                            'playbook': 'B',
                            'partial_out': self.config.partial_out,
                            'max_hold_min': 45
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
                    tp1 = session_vwap
                    
                    # 2차 목표: 반대측 +0.5σ
                    tp2 = session_vwap + self.config.tp2_to_band * session_std
                    
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
                        'reason': f"VWAP 과매도 페이드 롱 | 진입: -{self.config.sd_k_enter}σ, 재진입: -{self.config.sd_k_reenter}σ",
                        'playbook': 'C',
                        'partial_out': self.config.partial_out,
                        'max_hold_min': 30
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
                    tp1 = session_vwap
                    
                    # 2차 목표: 반대측 -0.5σ
                    tp2 = session_vwap - self.config.tp2_to_band * session_std
                    
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
                        'reason': f"VWAP 과매수 페이드 숏 | 진입: +{self.config.sd_k_enter}σ, 재진입: +{self.config.sd_k_reenter}σ",
                        'playbook': 'C',
                        'partial_out': self.config.partial_out,
                        'max_hold_min': 30
                    }
            
            return None
            
        except Exception as e:
            print(f"❌ VWAP 리버전 페이드 분석 오류: {e}")
            return None
    
    def analyze_session_strategy(self, df: pd.DataFrame, 
                                key_levels: Dict[str, float],
                                current_time: datetime) -> Optional[Dict]:
        """세션 기반 전략 통합 분석 (단계형 신호 적용)"""
        try:
            # current_time을 UTC timezone으로 변환
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=pytz.UTC)
            
            # 세션 시작 시간 확인 (이미 UTC tz-aware)
            session_start = self.get_session_start_time(current_time)
            
            # 세션 데이터 슬라이스 및 VWAP/OR 계산
            df_s = self._session_slice(df, session_start)
            self.session_vwap, self.session_std = self.calculate_session_vwap(df_s, session_start, current_time)
            or_info = self.calculate_opening_range(df_s, session_start)
            self.opening_range = or_info if or_info and (or_info.get("ready") or or_info.get("partial")) else None

            # --- OR 로그 ---
            if not or_info:
                print("ℹ️ OR 없음: 세션 시작 직후이거나 데이터 부족 → Play A 건너뜀, B/C만 평가")
            else:
                print(f"🎯 OR bars={or_info['bars']} ready={or_info['ready']} partial={or_info['partial']} "
                        f"range={or_info['range']:.2f}")
            
            # ATR 계산
            atr = calculate_atr(df_s, self.config.atr_len)
            if pd.isna(atr):
                return None
            
            # === 단계형 신호 분석 ===
            best_signal = None
            best_score = 0.0
            

            
            # A: OR가 없거나(strict) 준비 안 됐으면 스킵 또는 티어 제한
            if or_info and (or_info.get("ready") or (not self.config.strict_or and or_info.get("partial"))):
                for side in ["LONG","SHORT"]:
                    sig = self.analyze_staged_signal(df_s, self.session_vwap, or_info, atr, 'A', side)
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
                print("⏭️ Play A 스킵 (OR 미확정)")
            
            # B/C는 OR 없어도 정상 동작
            for side in ["LONG","SHORT"]:
                sig = self.analyze_staged_signal(df_s, self.session_vwap, or_info or {}, atr, 'B', side)
                if sig and sig["score"] > best_score:
                    best_signal, best_score = sig, sig["score"]

            if np.isfinite(self.session_vwap) and np.isfinite(self.session_std) and self.session_std > 0:
                for side in ["LONG","SHORT"]:
                    sig = self.analyze_staged_signal(df_s, self.session_vwap, or_info or {}, atr, 'C', side)
                    if sig and sig["score"] > best_score:
                        best_signal, best_score = sig, sig["score"]
            
            # 최고 점수 신호 반환
            if best_signal:
                print(f"�� 단계형 신호 생성: {best_signal['stage']} (점수: {best_signal['score']:.3f})")
                print(f"   �� 플레이북: {best_signal['playbook']}, 방향: {best_signal['side']}")
                print(f"   �� 액션: {best_signal['action']}, 신뢰도: {best_signal['confidence']:.1%}")
                
                # Gate 결과 출력
                gate_results = best_signal.get('gate_results', {})
                if gate_results:
                    print(f"   🔒 Gate 결과:")
                    print(f"      방향: {'✅' if gate_results.get('direction') else '❌'}")
                    print(f"      구조: {'✅' if gate_results.get('structure') else '❌'}")
                    print(f"      슬리피지: {'✅' if gate_results.get('slippage') else '❌'}")
                    print(f"      거래량: {'✅' if gate_results.get('volume') else '❌'}")
                
                return best_signal
            
            return None
            
        except Exception as e:
            print(f"❌ 세션 기반 전략 분석 오류: {e}")
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
            # 포지션 사이징 계산 (예시)
            risk_percent = 0.4  # 계좌 리스크 0.4%
            equity = 10000  # 예시 자본금
            risk_dollar = equity * risk_percent / 100
            
            stop_distance = abs(signal['entry_price'] - signal['stop_loss'])
            position_size = risk_dollar / stop_distance if stop_distance > 0 else 0
            
            signal['position_size'] = position_size
            signal['risk_dollar'] = risk_dollar
            
            return signal
        
        return None
        
    except Exception as e:
        print(f"❌ 세션 거래 계획 생성 오류: {e}")
        return None
