#!/usr/bin/env python3
"""
고급 청산 분석 전략 (Advanced Liquidation Analysis Strategy)
- 스파이크 판정 (Z점수 기반)
- LPI (Liquidation Pressure Index)
- 캐스케이드 조건 감지
- 3가지 실행형 전략: 스윕&리클레임, 스퀴즈 추세지속, 과열-소멸 페이드
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Literal
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections import deque
import pytz
from data.data_manager import get_data_manager
from utils.time_manager import get_time_manager
from indicators.global_indicators import get_global_indicator_manager

@dataclass
class AdvancedLiquidationConfig:
    """고급 청산 전략 설정"""
    # 기본 설정
    symbol: str = "ETHUSDT"
    
    # 청산 데이터 집계 설정
    bin_sec: int = 3  # 1초 → 3초 bin (노이즈 완화)
    agg_window_sec: int = 60  # 30초 → 60초 집계
    background_window_min: int = 180  # 60분 → 180분, 베이스라인 안정
    
    # 최소 워밍업 요구사항 (방향별)
    min_warmup_samples: int = 20  # ENTRY: 해당 방향 샘플 ≥20 (10 → 20)
    min_warmup_samples_setup: int = 10  # SETUP: 해당 방향 샘플 ≥10 (5 → 10)
    
    # 스파이크 판정 설정 (계층별 분리)
    z_spike: float = 0.8  # HEADS-UP 기준 (0.6 → 0.8)
    z_setup: float = 2.2  # SETUP 기준 (2.0 → 2.2)
    z_entry: float = 3.0  # ENTRY 기준 (2.5 → 3.0)
    z_strong: float = 2.2  # 강한 스파이크 임계값 (1.8 → 2.2)
    z_medium: float = 1.6  # 중간 스파이크 임계값 (1.2 → 1.6)
    lpi_bias: float = 0.15      # LPI 바이어스 임계값 (0.10 → 0.15)
    lpi_min: float = 0.6
    
    # 캐스케이드 설정 (지속성 강조)
    cascade_seconds: int = 30  # 지난 30초 안에 (20초 → 30초)
    cascade_count: int = 6  # 6회 이상 (5회 → 6회)
    cascade_z: float = 4.0  # z >= 4.0 (유지)
    
    # 쿨다운 설정 (재진입 남발 억제)
    cooldown_after_strong_sec: int = 20  # 강한 스파이크 후 20초 쿨다운 (8초 → 20초)
    cooldown_after_medium_sec: int = 8  # 중간 스파이크 후 8초 쿨다운 (3초 → 8초)
    
    # 리스크 설정 (단타 보유를 반영)
    risk_pct: float = 0.3  # 1트레이드 계좌대비 위험 (0.4% → 0.3%)
    slippage_max_pct: float = 0.02  # 최대 슬리피지 (3% → 2%)
    
    # 레벨 설정
    or_minutes: int = 30  # 오프닝 레인지 분
    atr_len: int = 14  # ATR 기간
    vwap_sd_enter: float = 2.2  # VWAP ±2.2σ 진입 (2.0 → 2.2)
    vwap_sd_enter_cascade: float = 2.0  # 캐스케이드 시 VWAP ±2.0σ 진입 (1.8 → 2.0)
    vwap_sd_stop: float = 3.0  # VWAP ±3.0σ 스탑 (2.5 → 3.0)
    
    # 전략 A: 스윕&리클레임
    sweep_buffer_atr: float = 0.25  # 스윕 버퍼 ATR (0.3 → 0.25)
    reclaim_atr_tolerance: float = 0.25  # 리클레임 ATR 허용치 (0.2~0.3 ATR)
    opposite_liquidation_boost: float = 0.1  # 반대측 청산 시 신뢰도 부스트
    tp1_R_a: float = 1.5  # 전략 A 1차 목표 R (1.2 → 1.5)
    tp2: str = "VWAP_or_range_edge"  # 2차 목표
    
    # 전략 B: 스퀴즈 추세지속
    retest_atr_tol: float = 0.55  # 리테스트 ATR 허용치 (0.4 → 0.55로 확대)
    retest_atr_tol_or_extension: float = 0.7  # OR 확장 시 리테스트 ATR 허용치 (추가 완화)
    tp1_R_b: float = 1.8  # 전략 B 1차 목표 R (1.5 → 1.8)
    or_extension: bool = True  # OR 확장 사용
    
    # 전략 C: 과열-소멸 페이드
    post_spike_decay_ratio: float = 0.9  # 스파이크 후 감소 비율 (0.8 → 0.9로 완화)
    post_spike_decay_ratio_cascade: float = 0.95  # 캐스케이드 시 감소 비율 (더 완화)
    z_extreme: float = 3.5  # 극단 스파이크 임계값 (다중 경로 트리거)
    lpi_extreme: float = 0.5  # LPI 극단 임계값
    vwap_sd_extreme: float = 1.8  # 극단 스파이크 시 VWAP ±1.8σ 진입
    vwap_sd_reenter: float = 1.5  # VWAP ±1.5σ 재진입 (SETUP 허용)
    stop_atr: float = 0.45  # 스탑 ATR (0.35 → 0.45)
    tp2_sigma: float = 0.6  # 2차 목표 시그마 (0.5 → 0.6)
    tp1_R_c: float = 1.5  # 전략 C 1차 목표 R (1.2 → 1.5)
    
    # 단계형 스코어링 설정
    # 가중치 구성 (합계 1.00) - 구조·트렌드 비중↑, 데이터 품질·오더플로우 비중↓
    weight_orderflow: float = 0.20  # 오더플로우(청산) (0.30 → 0.20)
    weight_structure: float = 0.25  # 구조 품질(플레이북별) (0.20 → 0.25)
    weight_decay_cascade: float = 0.15  # 소멸/연쇄 (유지)
    weight_trend_context: float = 0.15  # 추세/컨텍스트 (0.10 → 0.15)
    weight_location_baseline: float = 0.10  # 로케이션/기준선 (유지)
    weight_risk_appropriateness: float = 0.10  # 리스크 적정성 (유지)
    weight_data_quality: float = 0.05  # 데이터 품질 (유지)
    
    # Tier 임계값 (ENTRY 더 까다롭게)
    tier_entry_threshold: float = 0.62  # ENTRY ≥ 0.62 (0.55 → 0.62)
    tier_setup_threshold: float = 0.40  # SETUP ≥ 0.40 (0.35 → 0.40)
    tier_heads_up_threshold: float = 0.25  # HEADS-UP ≥ 0.25 유지
    
    # 동시양방향 충돌 회피 (더 완화)
    conflict_threshold: float = 0.02  # 점수 차 < 0.02면 관망 (0.01 → 0.02로 보수화)


class AdvancedLiquidationStrategy:
    """고급 청산 분석 전략"""
    
    def __init__(self, config: AdvancedLiquidationConfig):
        self.config = config
        
        # 글로벌 지표 매니저 초기화
        self.global_manager = get_global_indicator_manager()
        
        # TimeManager 초기화
        self.time_manager = get_time_manager()
        
        # 청산 데이터 저장소
        self.liquidation_bins = deque(maxlen=config.background_window_min * 60)  # 1분 = 60초
        self.long_bins = deque(maxlen=config.background_window_min * 60)
        self.short_bins = deque(maxlen=config.background_window_min * 60)
        
        # 백그라운드 통계
        self.mu_long = 0.0
        self.sigma_long = 1.0
        self.mu_short = 0.0
        self.sigma_short = 1.0
        
        # 상태 관리
        self.last_strong_spike_time = None
        self.cascade_detected = False
        self.cascade_start_time = None
        
        # 세션 필터 (감점형)
        self.session_active = False
        self.session_start_time = None
        self.session_score_penalty = 0.15  # 세션 외 신뢰도 감점
    
    """_summary_
    바이낸스 청산 이벤트 형식
    event = {'timestamp': datetime.datetime(2025, 8, 22, 1, 42, 47, 173880), 
        'symbol': 'ETHUSDT', 'side': 'BUY', 
        'quantity': 0.048, 'price': 4255.65, 'qty_usd': 204.2712, 'time': 1755794568097}
    """    
    def process_liquidation_event(self, event: Dict) -> None:
        """청산 이벤트 처리"""
        try:
            timestamp = event.get('timestamp', 0)
            side = event.get('side', 'unknown')
            qty_usd = event.get('qty_usd', 0.0)
            
            if qty_usd <= 0:
                return
            
            # UTC 시간으로 통일 (timezone-aware)
            current_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
            
            # 세션 상태 확인
            self._check_session_status(current_time)
            
            # 1초 bin에 추가
            bin_key = int(timestamp)
            
            # 청산 이벤트 side 매핑 (포지션 청산 방향)
            if side.lower() in ['long', 'sell']:
                # 롱 포지션 청산 → 롱 청산 데이터에 추가
                self._add_to_bin(self.long_bins, bin_key, qty_usd)
            elif side.lower() in ['short', 'buy']:
                # 숏 포지션 청산 → 숏 청산 데이터에 추가
                self._add_to_bin(self.short_bins, bin_key, qty_usd)
            else:
                print(f"⚠️ 알 수 없는 side: {side}, 이벤트 무시")
                return
            
            # 청산 bin에도 추가
            self._add_to_bin(self.liquidation_bins, bin_key, qty_usd)
            
            # 백그라운드 통계 업데이트
            self._update_background_stats()
            
        except Exception as e:
            print(f"❌ 청산 이벤트 처리 오류: {e}")
    
    def _add_to_bin(self, bin_deque: deque, bin_key: int, value: float) -> None:
        """bin에 값 추가"""
        # 기존 bin이 있으면 업데이트, 없으면 새로 생성
        bin_found = False
        for i, (key, val) in enumerate(bin_deque):
            if key == bin_key:
                bin_deque[i] = (key, val + value)
                bin_found = True
                break
        
        if not bin_found:
            bin_deque.append((bin_key, value))
    
    def _update_background_stats(self) -> None:
        """백그라운드 통계 업데이트"""
        try:
            # 롱 청산 통계
            long_values = [val for _, val in self.long_bins]
            if long_values:
                self.mu_long = np.mean(long_values)
                self.sigma_long = max(np.std(long_values), 1e-9)
            else:
                # 초기값 설정 (데이터가 없을 때)
                self.mu_long = 1000.0  # 기본 청산 금액
                self.sigma_long = 500.0  # 기본 표준편차
            
            # 숏 청산 통계
            short_values = [val for _, val in self.short_bins]
            if short_values:
                self.mu_short = np.mean(short_values)
                self.sigma_short = max(np.std(short_values), 1e-9)
            else:
                # 초기값 설정 (데이터가 없을 때)
                self.mu_short = 1000.0  # 기본 청산 금액
                self.sigma_short = 500.0  # 기본 표준편차
                
        except Exception as e:
            print(f"❌ 백그라운드 통계 업데이트 오류: {e}")
    
    # _cleanup_old_bins 메서드 제거 - deque maxlen이 자동으로 처리
    
    def _check_session_status(self, current_time: datetime) -> None:
        """세션 상태 확인 (DST 자동 반영)"""
        try:
            # UTC 시간을 각 시간대로 변환 (DST 자동 반영)
            london_tz = pytz.timezone('Europe/London')
            ny_tz = pytz.timezone('America/New_York')
            
            london_local = current_time.astimezone(london_tz)
            ny_local = current_time.astimezone(ny_tz)
            
            # 각 시간대의 오픈 시간 (현지 시간 기준)
            london_open = london_local.replace(hour=8, minute=0, second=0, microsecond=0)
            ny_open = ny_local.replace(hour=9, minute=30, second=0, microsecond=0)
            
            # ±90분 윈도우
            london_start = london_open - timedelta(minutes=90)
            london_end = london_open + timedelta(minutes=90)
            ny_start = ny_open - timedelta(minutes=90)
            ny_end = ny_open + timedelta(minutes=90)
            
            # 세션 활성 상태 확인
            self.session_active = (
                (london_start <= london_local <= london_end) or
                (ny_start <= ny_local <= ny_end)
            )
            
            # 세션 시작 시간 기록
            if self.session_active and not self.session_start_time:
                self.session_start_time = current_time
            elif not self.session_active:
                self.session_start_time = None
                
        except Exception as e:
            print(f"❌ 세션 상태 확인 오류: {e}")
    
    def get_session_score(self) -> float:
        """세션 점수 계산 (감점형)"""
        if self.session_active:
            return 1.0  # 세션 내: 만점
        else:
            return 1.0 - self.session_score_penalty  # 세션 외: 감점
    
    def get_session_core_status(self) -> Dict[str, Any]:
        """세션 코어 상태 확인 (거래량/변동성 급증 시간대)"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # 런던/뉴욕 세션 코어 시간 (오픈 후 ±30분)
            london_tz = pytz.timezone('Europe/London')
            ny_tz = pytz.timezone('America/New_York')
            
            london_local = current_time.astimezone(london_tz)
            ny_local = current_time.astimezone(ny_tz)
            
            # 런던 코어: 8:00-8:30 (현지시간)
            london_core_start = london_local.replace(hour=8, minute=0, second=0, microsecond=0)
            london_core_end = london_local.replace(hour=8, minute=30, second=0, microsecond=0)
            
            # 뉴욕 코어: 9:30-10:00 (현지시간)
            ny_core_start = ny_local.replace(hour=9, minute=30, second=0, microsecond=0)
            ny_core_end = ny_local.replace(hour=10, minute=0, second=0, microsecond=0)
            
            # 코어 세션 확인
            london_core = london_core_start <= london_local <= london_core_end
            ny_core = ny_core_start <= ny_local <= ny_core_end
            is_core_session = london_core or ny_core
            
            # 임계값 조정 계수 (백테스트용으로 1.0 고정)
            # threshold_multiplier = 1.1 if is_core_session else 0.95  # 코어: +10%, 한산: -5%
            threshold_multiplier = 1.0  # 백테스트용으로 세션 영향 제거 (디버깅용)
            
            return {
                'is_core_session': is_core_session,
                'london_core': london_core,
                'ny_core': ny_core,
                'threshold_multiplier': threshold_multiplier
            }
            
        except Exception as e:
            print(f"❌ 세션 코어 상태 확인 오류: {e}")
            return {
                'is_core_session': False,
                'london_core': False,
                'ny_core': False,
                'threshold_multiplier': 1.0
            }
    
    def check_slippage_and_risk(self, entry_price: float, stop_loss: float, 
                                current_price: float, atr: float) -> Dict[str, Any]:
        """슬리피지 및 리스크 체크"""
        try:
            # 슬리피지 계산
            slippage_pct = abs(entry_price - current_price) / current_price
            
            # 스탑 거리 계산
            stop_distance = abs(entry_price - stop_loss)
            stop_distance_atr = stop_distance / atr if atr > 0 else 0
            
            # 슬리피지 체크
            slippage_ok = slippage_pct <= self.config.slippage_max_pct
            
            # 스탑 거리 체크 (0.5~2.0R 허용)
            stop_distance_ok = 0.5 <= stop_distance_atr <= 2.0  # 0.02~5.0 → 0.5~2.0으로 수정
            
            # 전체 체크 결과
            all_checks_passed = slippage_ok and stop_distance_ok
            
            # 신호 등급 결정
            if all_checks_passed:
                signal_grade = 'ENTRY'
            elif slippage_ok and not stop_distance_ok:
                signal_grade = 'SETUP'  # 슬리피지는 OK, 스탑 거리 문제
            else:
                signal_grade = 'HEADS_UP'  # 슬리피지 초과
            
            return {
                'slippage_ok': slippage_ok,
                'stop_distance_ok': stop_distance_ok,
                'all_checks_passed': all_checks_passed,
                'signal_grade': signal_grade,
                'slippage_pct': slippage_pct,
                'stop_distance_atr': stop_distance_atr,
                'slippage_limit': self.config.slippage_max_pct
            }
            
        except Exception as e:
            print(f"❌ 슬리피지 및 리스크 체크 오류: {e}")
            return {
                'slippage_ok': False,
                'stop_distance_ok': False,
                'all_checks_passed': False,
                'signal_grade': 'HEADS_UP',
                'slippage_pct': 0.0,
                'stop_distance_atr': 0.0,
                'slippage_limit': self.config.slippage_max_pct
            }
    
    def check_gate_conditions(self, price_data: pd.DataFrame, atr: float, 
                                current_price: float, signal_side: str = None) -> Dict[str, Any]:
        """Gate(최소 위생 조건) 확인 (방향별 분리)"""
        try:
            # 1. 데이터 준비 확인 (방향별 워밍업)
            warmup_status = self.get_warmup_status()
            
            # 신호 방향에 따른 워밍업 확인
            if signal_side == 'BUY':  # 롱 신호
                data_ready = warmup_status['long_signal_warmup']  # 숏 청산 샘플 확인
            elif signal_side == 'SELL':  # 숏 신호
                data_ready = warmup_status['short_signal_warmup']  # 롱 청산 샘플 확인
            else:
                data_ready = warmup_status['basic_warmup']  # 기본 워밍업
            
            # 2. 실행 가능 조건 확인
            atr_valid = atr > 0
            price_valid = current_price > 0
            
            # 3. 기본 위생 조건
            basic_hygiene = (data_ready and atr_valid and price_valid)
            
            # 4. 하드 블록 조건 (완전 차단)
            hard_blocked = False
            block_reason = None
            
            # 워밍업 부족 (방향별)
            if not data_ready:
                if signal_side == 'BUY':
                    block_reason = f"롱 신호 워밍업 부족 (숏 청산 샘플: {warmup_status['short_samples']}개)"
                elif signal_side == 'SELL':
                    block_reason = f"숏 신호 워밍업 부족 (롱 청산 샘플: {warmup_status['long_samples']}개)"
                else:
                    block_reason = f"워밍업 부족 (전체 샘플: {warmup_status['total_samples']}개)"
                hard_blocked = True
                print(f"🚪 Gate 블록: {block_reason}")
            # ATR 무효
            elif not atr_valid:
                hard_blocked = True
                block_reason = f"ATR 무효 (ATR={atr:.2f})"
                print(f"🚪 Gate 블록: {block_reason}")
            # 가격 무효
            elif not price_valid:
                hard_blocked = True
                block_reason = f"가격 무효 (가격={current_price:.2f})"
                print(f"🚪 Gate 블록: {block_reason}")
            
            return {
                'gate_passed': basic_hygiene and not hard_blocked,
                'basic_hygiene': basic_hygiene,
                'hard_blocked': hard_blocked,
                'block_reason': block_reason,
                'warmup_status': warmup_status,
                'atr_valid': atr_valid,
                'price_valid': price_valid,
                'signal_side': signal_side
            }
            
        except Exception as e:
            print(f"❌ Gate 조건 확인 오류: {e}")
            return {
                'gate_passed': False,
                'basic_hygiene': False,
                'hard_blocked': True,
                'block_reason': f"오류: {e}",
                'warmup_status': {},
                'atr_valid': False,
                'price_valid': False,
                'signal_side': signal_side
            }
    
    def calculate_orderflow_score(self, metrics: Dict[str, Any], signal_side: str) -> float:
        """오더플로우 점수 계산 (청산 지분·강도) - SETUP/ENTRY 분리"""
        try:
            z_long = metrics.get('z_long', 0)
            z_short = metrics.get('z_short', 0)
            lpi = metrics.get('lpi', 0)
            
            # 롱 신호: '숏 청산' 지분·강도↑ 가점 / '롱 청산'↑ 감점
            if signal_side == 'BUY':
                # 숏 청산 스파이크 가점 (계층별 임계값 분리)
                if z_short >= self.config.z_entry:  # ENTRY 기준 (2.5)
                    short_liquidation_bonus = min(z_short / 3.0, 1.0)
                elif z_short >= self.config.z_setup:  # SETUP 기준 (2.0)
                    short_liquidation_bonus = min(z_short / 3.0, 0.7)
                elif z_short >= self.config.z_spike:  # HEADS-UP 기준 (0.6)
                    short_liquidation_bonus = min(z_short / 3.0, 0.4)
                else:
                    short_liquidation_bonus = 0.0
                
                # 롱 청산 스파이크 감점 (z_long ≥ 1.0) - 임계값 완화
                long_liquidation_penalty = min(z_long / 3.0, 0.5) if z_long >= 1.0 else 0.0
                
                # LPI 바이어스 (숏 청산 편향)
                lpi_bonus = max(lpi, 0) if lpi > 0 else 0.0
                
                score = short_liquidation_bonus + lpi_bonus - long_liquidation_penalty
                
            # 숏 신호: '롱 청산' 지분·강도↑ 가점 / '숏 청산'↑ 감점
            else:  # SELL
                # 롱 청산 스파이크 가점 (계층별 임계값 분리)
                if z_long >= self.config.z_entry:  # ENTRY 기준 (2.5)
                    long_liquidation_bonus = min(z_long / 3.0, 1.0)
                elif z_long >= self.config.z_setup:  # SETUP 기준 (2.0)
                    long_liquidation_bonus = min(z_long / 3.0, 0.7)
                elif z_long >= self.config.z_spike:  # HEADS-UP 기준 (0.6)
                    long_liquidation_bonus = min(z_long / 3.0, 0.4)
                else:
                    long_liquidation_bonus = 0.0
                
                # 숏 청산 스파이크 감점 (z_short ≥ 1.0) - 임계값 완화
                short_liquidation_penalty = min(z_short / 3.0, 0.5) if z_short >= 1.0 else 0.0
                
                # LPI 바이어스 (롱 청산 편향)
                lpi_bonus = max(-lpi, 0) if lpi < 0 else 0.0
                
                score = long_liquidation_bonus + lpi_bonus - short_liquidation_penalty
            
            # 점수 정규화 (0.0 ~ 1.0)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"❌ 오더플로우 점수 계산 오류: {e}")
            return 0.0
    
    def calculate_structure_score(self, strategy_name: str, price_data: pd.DataFrame, 
                                key_levels: Dict[str, float], opening_range: Dict[str, float],
                                atr: float, metrics: Dict[str, Any]) -> float:
        """구조 품질 점수 계산 (플레이북별)"""
        try:
            if strategy_name == 'A':  # 스윕&리클레임
                return self._calculate_strategy_a_structure_score(price_data, key_levels, atr)
            elif strategy_name == 'B':  # 스퀴즈 지속
                return self._calculate_strategy_b_structure_score(price_data, opening_range, atr, metrics)
            elif strategy_name == 'C':  # VWAP 페이드
                return self._calculate_strategy_c_structure_score(price_data, key_levels, atr, metrics)
            else:
                return 0.0
                
        except Exception as e:
            print(f"❌ 구조 품질 점수 계산 오류: {e}")
            return 0.0
    
    def _calculate_strategy_a_structure_score(self, price_data: pd.DataFrame, 
                                            key_levels: Dict[str, float], atr: float) -> float:
        """전략 A 구조 품질 점수"""
        try:
            if len(price_data) < 3:
                return 0.0
            
            score = 0.05  # 바닥 점수
            current_price = price_data['close'].iloc[-1]
            prev_day_low = key_levels.get('prev_day_low', 0)
            prev_day_high = key_levels.get('prev_day_high', 0)
            
            # 스윕 깊이 (ATR 기준)
            if prev_day_low > 0 and current_price < prev_day_low:
                sweep_depth = (prev_day_low - current_price) / atr if atr > 0 else 0
                if sweep_depth >= 0.25:  # min_sweep_atr
                    score += 0.4
            
            # 리클레임 품질 (완화: 종가 리클로즈 + 레벨±0.5ATR 근접도 SETUP 인정)
            if len(price_data) >= 2:
                prev_close = price_data['close'].iloc[-2]
                current_close = price_data['close'].iloc[-1]
                
                # 종가 리클로즈 (완전 복귀) - 가점 증가
                if prev_day_low > 0 and prev_close < prev_day_low and current_close > prev_day_low:
                    score += 0.5  # 0.4 → 0.5로 증가
                # 근접 리클레임 (±0.5ATR) - SETUP 인정
                elif prev_day_low > 0 and prev_close < prev_day_low:
                    atr_buffer = atr * 0.5
                    if current_close > prev_day_low - atr_buffer:
                        score += 0.3  # 0.2 → 0.3으로 증가
                
                # 상단 스윕&리클레임도 동일하게
                if prev_day_high > 0 and prev_close > prev_day_high and current_close < prev_day_high:
                    score += 0.5  # 0.4 → 0.5로 증가
                elif prev_day_high > 0 and prev_close > prev_day_high:
                    atr_buffer = atr * 0.5
                    if current_close < prev_day_high + atr_buffer:
                        score += 0.3  # 0.2 → 0.3으로 증가
            
            # 스윕 최근성 (≤15봉)
            bars_since_sweep = 0
            for i in range(1, min(16, len(price_data))):
                if prev_day_low > 0 and price_data['low'].iloc[-i] < prev_day_low:
                    bars_since_sweep = i
                    break
                if prev_day_high > 0 and price_data['high'].iloc[-i] > prev_day_high:
                    bars_since_sweep = i
                    break
            
            if bars_since_sweep <= 15:
                score += 0.2
            elif bars_since_sweep > 15:
                score -= 0.1  # 패널티
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"❌ 전략 A 구조 점수 계산 오류: {e}")
            return 0.0
    
    def _calculate_strategy_b_structure_score(self, price_data: pd.DataFrame, 
                                            opening_range: Dict[str, float], atr: float,
                                            metrics: Dict[str, Any]) -> float:
        """전략 B 구조 품질 점수"""
        try:
            if len(price_data) < 4:
                return 0.0
            
            score = 0.05  # 바닥 점수
            current_price = price_data['close'].iloc[-1]
            or_high = opening_range.get('high', 0)
            or_low = opening_range.get('low', 0)
            
            # 브레이크 확인
            if or_high > 0 and current_price > or_high:  # 상단 돌파
                score += 0.3
            elif or_low > 0 and current_price < or_low:  # 하단 이탈
                score += 0.3
            
            # 리테스트 거리 (≤0.5~0.6ATR 가점)
            retest_distance = 0.0
            if or_high > 0 and current_price > or_high:  # 롱 신호
                for i in range(1, min(5, len(price_data))):
                    low_price = price_data['low'].iloc[-i]
                    if low_price < or_high:
                        retest_distance = (or_high - low_price) / atr if atr > 0 else 0
                        break
            elif or_low > 0 and current_price < or_low:  # 숏 신호
                for i in range(1, min(5, len(price_data))):
                    high_price = price_data['high'].iloc[-i]
                    if high_price > or_low:
                        retest_distance = (high_price - or_low) / atr if atr > 0 else 0
                        break
            
            if retest_distance <= 0.6:  # 0.6 → 0.6 유지
                score += 0.4
            elif retest_distance <= 1.0:
                score += 0.2
            
            # 추가 반대편 청산 확인 가점
            if self._check_additional_long_liquidation():
                score += 0.3
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"❌ 전략 B 구조 점수 계산 오류: {e}")
            return 0.0
    
    def _calculate_strategy_c_structure_score(self, price_data: pd.DataFrame, 
                                            key_levels: Dict[str, float], atr: float,
                                            metrics: Dict[str, Any]) -> float:
        """전략 C 구조 품질 점수"""
        try:
            score = 0.0
            current_price = price_data['close'].iloc[-1]
            vwap = key_levels.get('vwap')
            vwap_std = key_levels.get('vwap_std')
            
            # VWAP와 표준편차가 없으면 직접 계산
            if not vwap or not vwap_std or vwap_std <= 0:
                vwap, vwap_std = self._fallback_vwap_std(price_data)
            
            # ±σ 이탈 정도 (2σ 기준)
            vwap_distance = abs(current_price - vwap) / vwap_std if vwap_std > 0 else 0
            
            if vwap_distance >= 2.0:
                score += 0.5
            elif vwap_distance >= 1.8:
                score += 0.3
            
            # 극단 스파이크 확인
            z_long = metrics.get('z_long', 0)
            z_short = metrics.get('z_short', 0)
            lpi = metrics.get('lpi', 0)
            
            extreme_spike = (max(z_long, z_short) >= self.config.z_extreme and 
                           abs(lpi) >= self.config.lpi_extreme)
            
            if extreme_spike:
                score += 0.3
            
            # 재진입 확인 (±1.5σ)
            if vwap_distance <= 1.5:
                score += 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"❌ 전략 C 구조 점수 계산 오류: {e}")
            return 0.0
    
    def calculate_decay_cascade_score(self, metrics: Dict[str, Any], side: str) -> float:
        """소멸/연쇄 점수 계산 (peak→decay & cascade) - SETUP/ENTRY 분리"""
        try:
            score = 0.0
            
            # 1. 감소 확인 (SETUP/ENTRY 임계값 분리)
            decay_ok_entry = self._check_post_spike_decay(metrics, side, for_entry=True)
            decay_ok_setup = self._check_post_spike_decay(metrics, side, for_entry=False)
            
            if decay_ok_entry:
                score += 0.6  # ENTRY 기준 만족
            elif decay_ok_setup:
                score += 0.3  # SETUP 기준만 만족
            # ENTRY 승급 조건으로만 사용, 불만족시 강한 감점은 제거
            
            # 2. 같은 방향 캐스케이드 감점 (Play C만 강차단, A/B는 감점/강등)
            is_cascade = metrics.get('is_cascade', False)
            if is_cascade:
                # 캐스케이드 지분 확인 (최근 20~30초)
                current_time = datetime.now(timezone.utc)
                window_start = int(current_time.timestamp()) - 25  # 25초 윈도우
                
                if side == 'long':
                    cascade_liquidation = sum(val for ts, val in self.long_bins if ts >= window_start)
                    total_liquidation = sum(val for ts, val in self.liquidation_bins if ts >= window_start)
                else:  # short
                    cascade_liquidation = sum(val for ts, val in self.short_bins if ts >= window_start)
                    total_liquidation = sum(val for ts, val in self.liquidation_bins if ts >= window_start)
                
                cascade_ratio = cascade_liquidation / (total_liquidation + 1e-9)
                
                # 지분 ≥ 0.85 & 이벤트 ≥ 2회면 감점 (Play C만 강차단)
                if cascade_ratio >= 0.85:
                    score -= 0.2  # 강차단에서 감점으로 완화
                elif cascade_ratio >= 0.7:
                    score -= 0.1  # 감점 유지
            
            # 3. 쿨다운 감점 적용
            cooldown_info = metrics.get('cooldown_info', {})
            if cooldown_info.get('active', False):
                score -= cooldown_info.get('penalty', 0.0)
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"❌ 소멸/연쇄 점수 계산 오류: {e}")
            return 0.0
    
    def calculate_trend_context_score(self, price_data: pd.DataFrame, 
                                    key_levels: Dict[str, float], metrics: Dict[str, Any]) -> float:
        """추세/컨텍스트 점수 계산"""
        try:
            score = 0.0
            current_price = price_data['close'].iloc[-1]
            
            # 1. 가격↔VWAP 관계
            vwap = key_levels.get('vwap', current_price)
            if vwap > 0:
                vwap_distance = abs(current_price - vwap) / current_price
                if vwap_distance <= 0.01:  # ±1% 이내
                    score += 0.3
                elif vwap_distance <= 0.02:  # ±2% 이내
                    score += 0.1
            
            # 2. 세션 위상 보정
            session_core = self.get_session_core_status()
            if session_core['is_core_session']:
                score += 0.05  # MID +0.05
            else:
                score -= 0.05  # OPEN -0.05
            
            # 3. EMA 정렬 (간단한 추세 확인)
            if len(price_data) >= 20:
                ema_20 = price_data['close'].rolling(20).mean().iloc[-1]
                ema_10 = price_data['close'].rolling(10).mean().iloc[-1]
                
                if current_price > ema_10 > ema_20:  # 상승 추세
                    score += 0.2
                elif current_price < ema_10 < ema_20:  # 하락 추세
                    score += 0.2
                else:  # 혼조
                    score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"❌ 추세/컨텍스트 점수 계산 오류: {e}")
            return 0.0
    
    def calculate_location_baseline_score(self, price_data: pd.DataFrame, 
                                        key_levels: Dict[str, float], 
                                        opening_range: Dict[str, float], atr: float) -> float:
        """로케이션/기준선 점수 계산"""
        try:
            score = 0.0
            current_price = price_data['close'].iloc[-1]
            
            # 1. 키레벨 근접/복귀 (허용대역 상향)
            prev_day_low = key_levels.get('prev_day_low', 0)
            prev_day_high = key_levels.get('prev_day_high', 0)
            
            if prev_day_low > 0:
                low_distance = abs(current_price - prev_day_low) / atr if atr > 0 else 0
                if low_distance <= 0.5:
                    score += 0.3
                elif low_distance <= 1.5:
                    score += 0.1
            
            if prev_day_high > 0:
                high_distance = abs(current_price - prev_day_high) / atr if atr > 0 else 0
                if high_distance <= 0.5:
                    score += 0.3
                elif high_distance <= 1.5:
                    score += 0.1
            
            # 2. OR 확장 여지
            or_high = opening_range.get('high', 0)
            or_low = opening_range.get('low', 0)
            
            if or_high > 0 and or_low > 0:
                or_range = or_high - or_low
                current_range = max(price_data['high'].iloc[-20:]) - min(price_data['low'].iloc[-20:])
                
                if current_range < or_range * 1.5:  # 확장 여지 있음
                    score += 0.2
            
            # 3. VWAP 근접 (허용대역 상향)
            vwap = key_levels.get('vwap', current_price)
            if vwap > 0:
                vwap_distance = abs(current_price - vwap) / current_price
                if vwap_distance <= 0.015:  # ±1.5% 이내
                    score += 0.2
                elif vwap_distance <= 0.02:  # ±2% 이내
                    score += 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"❌ 로케이션/기준선 점수 계산 오류: {e}")
            return 0.0
    
    def calculate_risk_appropriateness_score(self, entry_price: float, stop_loss: float, 
                                           take_profit1: float, atr: float) -> float:
        """리스크 적정성 점수 계산"""
        try:
            score = 0.0
            
            # 1. 스탑 거리/ATR 적합 (0.5~2.0R 가점)
            stop_distance = abs(entry_price - stop_loss)
            stop_distance_atr = stop_distance / atr if atr > 0 else 0
            
            if 0.5 <= stop_distance_atr <= 2.0:  # 0.6~1.6 → 0.5~2.0으로 확대
                score += 0.5
            elif 0.3 <= stop_distance_atr <= 2.5:  # 0.4~2.0 → 0.3~2.5로 확대
                score += 0.3
            elif stop_distance_atr < 0.2 or stop_distance_atr > 3.0:
                score -= 0.3  # 패널티
            
            # 2. R-multiple 목표 가능성
            risk = stop_distance
            if risk > 0:
                tp1_distance = abs(take_profit1 - entry_price)
                r_multiple = tp1_distance / risk
                
                if r_multiple >= 1.7:
                    score += 0.3
                elif r_multiple >= 1.3:
                    score += 0.2
                elif r_multiple < 1.0:
                    score -= 0.2  # 패널티
            
            # 3. 슬리피지 체크
            slippage_check = self.check_slippage_and_risk(entry_price, stop_loss, entry_price, atr)
            if slippage_check['slippage_ok']:
                score += 0.2
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"❌ 리스크 적정성 점수 계산 오류: {e}")
            return 0.0
    
    def calculate_data_quality_score(self) -> float:
        """데이터 품질 점수 계산"""
        try:
            score = 0.0
            
            # 1. 결측/빈 bin 비율
            total_bins = len(self.liquidation_bins)
            if total_bins > 0:
                # 최근 60초 내 빈 bin 확인
                current_time = datetime.now(timezone.utc)
                window_start = int(current_time.timestamp()) - 60
                
                filled_bins = sum(1 for ts, val in self.liquidation_bins if ts >= window_start and val > 0)
                total_recent_bins = sum(1 for ts, _ in self.liquidation_bins if ts >= window_start)
                
                if total_recent_bins > 0:
                    fill_ratio = filled_bins / total_recent_bins
                    if fill_ratio >= 0.8:
                        score += 0.3
                    elif fill_ratio >= 0.6:
                        score += 0.2
                    elif fill_ratio < 0.4:
                        score -= 0.2
            
            # 2. 이벤트 밀도
            if total_bins >= 120:  # 최소 워밍업
                score += 0.4
            elif total_bins >= 60:
                score += 0.2
            
            # 3. μ·σ 안정성
            if self.sigma_long > 0 and self.sigma_short > 0:
                score += 0.3
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            print(f"❌ 데이터 품질 점수 계산 오류: {e}")
            return 0.0
    
    def calculate_total_score(self, strategy_name: str, signal_side: str, 
                            price_data: pd.DataFrame, key_levels: Dict[str, float],
                            opening_range: Dict[str, float], atr: float, 
                            entry_price: float, stop_loss: float, take_profit1: float,
                            metrics: Dict[str, Any]) -> Dict[str, Any]:
        """통합 점수 계산 (가중 합산)"""
        try:
            # 각 구성 요소 점수 계산
            orderflow_score = self.calculate_orderflow_score(metrics, signal_side)
            structure_score = self.calculate_structure_score(strategy_name, price_data, key_levels, opening_range, atr, metrics)
            decay_cascade_score = self.calculate_decay_cascade_score(metrics, 'long' if signal_side == 'BUY' else 'short')
            trend_context_score = self.calculate_trend_context_score(price_data, key_levels, metrics)
            location_baseline_score = self.calculate_location_baseline_score(price_data, key_levels, opening_range, atr)
            risk_appropriateness_score = self.calculate_risk_appropriateness_score(entry_price, stop_loss, take_profit1, atr)
            data_quality_score = self.calculate_data_quality_score()
            
            # 가중 합산
            total_score = (
                orderflow_score * self.config.weight_orderflow +
                structure_score * self.config.weight_structure +
                decay_cascade_score * self.config.weight_decay_cascade +
                trend_context_score * self.config.weight_trend_context +
                location_baseline_score * self.config.weight_location_baseline +
                risk_appropriateness_score * self.config.weight_risk_appropriateness +
                data_quality_score * self.config.weight_data_quality
            )
            
            # 점수 정규화 (0.0 ~ 1.0)
            total_score = max(0.0, min(1.0, total_score))
            
            return {
                'total_score': total_score,
                'component_scores': {
                    'orderflow': orderflow_score,
                    'structure': structure_score,
                    'decay_cascade': decay_cascade_score,
                    'trend_context': trend_context_score,
                    'location_baseline': location_baseline_score,
                    'risk_appropriateness': risk_appropriateness_score,
                    'data_quality': data_quality_score
                },
                'weighted_contributions': {
                    'orderflow': orderflow_score * self.config.weight_orderflow,
                    'structure': structure_score * self.config.weight_structure,
                    'decay_cascade': decay_cascade_score * self.config.weight_decay_cascade,
                    'trend_context': trend_context_score * self.config.weight_trend_context,
                    'location_baseline': location_baseline_score * self.config.weight_location_baseline,
                    'risk_appropriateness': risk_appropriateness_score * self.config.weight_risk_appropriateness,
                    'data_quality': data_quality_score * self.config.weight_data_quality
                }
            }
            
        except Exception as e:
            print(f"❌ 통합 점수 계산 오류: {e}")
            return {
                'total_score': 0.0,
                'component_scores': {},
                'weighted_contributions': {}
            }
    
    def determine_signal_tier(self, total_score: float, strategy_name: str,
                            metrics: Dict[str, Any], atr: float) -> Dict[str, Any]:
        """신호 Tier 결정 (ENTRY/SETUP/HEADS-UP)"""
        try:
            # 현재 전략 추적
            self.current_strategy = strategy_name
            
            # 기본 Tier 결정
            if total_score >= self.config.tier_entry_threshold:
                base_tier = 'ENTRY'
            elif total_score >= self.config.tier_setup_threshold:
                base_tier = 'SETUP'
            elif total_score >= self.config.tier_heads_up_threshold:
                base_tier = 'HEADS_UP'
            else:
                base_tier = 'REJECT'
            
            # 전략별 특수 규칙 적용
            final_tier = base_tier
            tier_modification = None
            
            # 전략 C: 감소 미확인 시 ENTRY → SETUP 강등 (디버깅용으로 완화)
            if strategy_name == 'C' and base_tier == 'ENTRY':
                # 감소 확인 필요
                decay_ok = self._check_post_spike_decay(metrics, 'long')  # 기본값
                if not decay_ok:
                    # final_tier = 'SETUP'  # 디버깅용으로 강등 비활성화
                    # tier_modification = "감소 미확인으로 강등"
                    pass
            
            # 캐스케이드 감지 시: ENTRY → SETUP 강등 (디버깅용으로 완화)
            if base_tier == 'ENTRY' and metrics.get('is_cascade', False):
                # Play C(페이드)만 강차단, A/B는 강등
                strategy_name = getattr(self, 'current_strategy', 'UNKNOWN')
                if strategy_name == 'C':
                    final_tier = 'REJECT'  # Play C는 강차단
                    tier_modification = "캐스케이드 감지로 강차단 (Play C)"
                else:
                    final_tier = 'SETUP'  # Play A/B는 강등
                    tier_modification = "캐스케이드 감지로 강등 (Play A/B)"
            # 슬리피지 초과 강등 로직은 신호 생성 단계에서 처리(여기서는 미적용)
            
            return {
                'base_tier': base_tier,
                'final_tier': final_tier,
                'tier_modification': tier_modification,
                'total_score': total_score,
                'thresholds': {
                    'entry': self.config.tier_entry_threshold,
                    'setup': self.config.tier_setup_threshold,
                    'heads_up': self.config.tier_heads_up_threshold
                }
            }
            
        except Exception as e:
            print(f"❌ Tier 결정 오류: {e}")
            return {
                'base_tier': 'REJECT',
                'final_tier': 'REJECT',
                'tier_modification': f"오류: {e}",
                'total_score': total_score,
                'thresholds': {}
            }
    
    def check_conflict_resolution(self, long_signal: Dict[str, Any], 
                                short_signal: Dict[str, Any]) -> Dict[str, Any]:
        """동시양방향 충돌 회피 확인"""
        try:
            if not long_signal or not short_signal:
                return {'conflict': False, 'resolution': '단일 신호'}
            
            long_score = long_signal.get('total_score', 0)
            short_score = short_signal.get('total_score', 0)
            
            score_diff = abs(long_score - short_score)
            
            if score_diff < self.config.conflict_threshold:
                return {
                    'conflict': True,
                    'resolution': '관망 (점수 차 < 0.05)',
                    'long_score': long_score,
                    'short_score': short_score,
                    'score_diff': score_diff
                }
            else:
                # 높은 점수 신호 선택
                winner = 'LONG' if long_score > short_score else 'SHORT'
                return {
                    'conflict': False,
                    'resolution': f'{winner} 신호 선택',
                    'winner_score': max(long_score, short_score),
                    'loser_score': min(long_score, short_score),
                    'score_diff': score_diff
                }
                
        except Exception as e:
            print(f"❌ 충돌 해결 확인 오류: {e}")
            return {'conflict': True, 'resolution': f'오류: {e}'}
    
    def log_strategy_diagnosis(self, strategy_name: str, metrics: Dict[str, Any], 
                                reason: str = None) -> None:
        """전략별 진단 로그 - 디버깅 비활성화"""
        # 디버깅 출력 제거
        pass
    
    def log_scoring_results(self, strategy_name: str, signal_side: str, 
                            scoring_result: Dict[str, Any], tier_result: Dict[str, Any]) -> None:
        """스코어링 결과 로그 - 디버깅 비활성화"""
        # 디버깅 출력 제거
        pass
    
    def log_candidate_details(self, strategy_name: str, signal_side: str, 
                            metrics: Dict[str, Any], price_data: pd.DataFrame,
                            key_levels: Dict[str, float], atr: float) -> None:
        """후보별 상세 로그 - 디버깅 비활성화"""
        # 디버깅 출력 제거
        pass
    
    def get_warmup_status(self) -> Dict[str, Any]:
        """워밍업 상태 확인 (방향별 분리)"""
        # 현재 시간 (UTC)
        now = datetime.now(timezone.utc)
        
        # 롱/숏 청산 이벤트 수집
        long_samples = len(self.long_bins)
        short_samples = len(self.short_bins)
        total_samples = long_samples + short_samples
        
        # 기본 워밍업 확인 (전체 샘플)
        basic_warmup = total_samples >= 1
        
        # 방향별 워밍업 확인 (신호 방향에 따라 해당 방향만 확인)
        # LONG 신호: 숏 청산 샘플만 확인 (숏 청산이 많으면 롱 신호)
        # SHORT 신호: 롱 청산 샘플만 확인 (롱 청산이 많으면 숏 신호)
        long_signal_warmup = short_samples >= self.config.min_warmup_samples_setup  # 롱 신호를 위한 숏 청산 샘플 (SETUP 기준)
        short_signal_warmup = long_samples >= self.config.min_warmup_samples_setup  # 숏 신호를 위한 롱 청산 샘플 (SETUP 기준)
        
        # ENTRY 레벨 워밍업 확인
        long_signal_entry_warmup = short_samples >= self.config.min_warmup_samples  # ENTRY 기준
        short_signal_entry_warmup = long_samples >= self.config.min_warmup_samples  # ENTRY 기준
        
        # μ·σ 안정성 확인
        mu_long_valid = self.mu_long > 0
        mu_short_valid = self.mu_short > 0
        sigma_long_valid = self.sigma_long > 0
        sigma_short_valid = self.sigma_short > 0
        
        mu_stable = (sigma_long_valid and sigma_short_valid and
                    mu_long_valid and mu_short_valid)
        
        # 워밍업 상태 로깅 제거 (디버깅 출력)
        
        # 방향별 실행 가능 여부
        can_long_setup = basic_warmup and long_signal_warmup and mu_stable
        can_long_entry = basic_warmup and long_signal_entry_warmup and mu_stable
        can_short_setup = basic_warmup and short_signal_warmup and mu_stable
        can_short_entry = basic_warmup and short_signal_entry_warmup and mu_stable
        
        # 워밍업 상태 요약
        warmup_summary = {
            'basic_warmup': basic_warmup,
            'long_signal_warmup': long_signal_warmup,
            'short_signal_warmup': short_signal_warmup,
            'long_signal_entry_warmup': long_signal_entry_warmup,
            'short_signal_entry_warmup': short_signal_entry_warmup,
            'mu_stable': mu_stable,
            'can_long_setup': can_long_setup,
            'can_long_entry': can_long_entry,
            'can_short_setup': can_short_setup,
            'can_short_entry': can_short_entry,
            'total_samples': total_samples,
            'long_samples': long_samples,
            'short_samples': short_samples,
            'mu_long_valid': mu_long_valid,
            'mu_short_valid': mu_short_valid,
            'sigma_long_valid': sigma_long_valid,
            'sigma_short_valid': sigma_short_valid
        }
        
        return warmup_summary
    
    def get_current_liquidation_metrics(self) -> Dict[str, Any]:
        """현재 청산 지표 계산"""
        try:
            # UTC 시간으로 통일
            current_time = datetime.now(timezone.utc)
            current_timestamp = int(current_time.timestamp())
            
            # 30초 윈도우 계산
            window_start = current_timestamp - self.config.agg_window_sec
            
            # 롱 청산 30초 합계
            l_long_30s = sum(val for ts, val in self.long_bins if ts >= window_start)
            
            # 숏 청산 30초 합계
            l_short_30s = sum(val for ts, val in self.short_bins if ts >= window_start)
            
            # Z점수 계산 - 30초 합계에 맞게 스케일링
            # 30초 합계의 경우: μ → 30×μ, σ → √30×σ
            scale_factor = self.config.agg_window_sec  # 30
            scale_sqrt = np.sqrt(scale_factor)  # √30
            
            mu_long_scaled = self.mu_long * scale_factor
            sigma_long_scaled = self.sigma_long * scale_sqrt
            mu_short_scaled = self.mu_short * scale_factor
            sigma_short_scaled = self.sigma_short * scale_sqrt
            
            # 상대적 Z-score 계산 (백그라운드 대비 변화율 기반)
            # 절단 제거: |z|<1.0 → 0 처리 제거하고, 절대값만 적용
            if mu_long_scaled > 0:
                z_long_raw = (l_long_30s - mu_long_scaled) / max(sigma_long_scaled, 1e-9)
                z_long = abs(z_long_raw)
            else:
                z_long = 0.0
                
            if mu_short_scaled > 0:
                z_short_raw = (l_short_30s - mu_short_scaled) / max(sigma_short_scaled, 1e-9)
                z_short = abs(z_short_raw)
            else:
                z_short = 0.0
            
            # LPI 계산
            total_liquidation = l_long_30s + l_short_30s
            lpi = (l_short_30s - l_long_30s) / (total_liquidation + 1e-9)
            
            # 캐스케이드 감지
            cascade_info = self._detect_cascade(current_timestamp)
            is_cascade = cascade_info['total_cascade']
            
            # 쿨다운 상태 확인 (방향별)
            cooldown_info = self._is_cooldown_active(current_time)
            
            # 상세한 Z-score 스케일링 로깅 제거 (디버깅 출력)
            
            return {
                'timestamp': current_time,
                'l_long_30s': l_long_30s,
                'l_short_30s': l_short_30s,
                'z_long': z_long,
                'z_short': z_short,
                'lpi': lpi,
                'is_cascade': is_cascade,
                'cooldown_info': cooldown_info,
                'session_active': self.session_active,
                'background_stats': {
                    'mu_long': self.mu_long,
                    'sigma_long': self.sigma_long,
                    'mu_short': self.mu_short,
                    'sigma_short': self.sigma_short,
                    'mu_long_scaled': mu_long_scaled,
                    'sigma_long_scaled': sigma_long_scaled,
                    'mu_short_scaled': mu_short_scaled,
                    'sigma_short_scaled': sigma_short_scaled
                }
            }
            
        except Exception as e:
            print(f"❌ 청산 지표 계산 오류: {e}")
            return {}
    
    def _detect_cascade(self, current_timestamp: int) -> Dict[str, bool]:
        """캐스케이드 조건 감지 (방향별 분리)"""
        try:
            cascade_start = current_timestamp - self.config.cascade_seconds
            long_cascade_count = 0
            short_cascade_count = 0
            
            # 롱 청산 캐스케이드 확인
            for ts, val in self.long_bins:
                if ts >= cascade_start:
                    # 1초 bin 값과 1초 스케일 통계 비교
                    z_score = (val - self.mu_long) / max(self.sigma_long, 1e-9)
                    if z_score >= self.config.cascade_z:
                        long_cascade_count += 1
            
            # 숏 청산 캐스케이드 확인
            for ts, val in self.short_bins:
                if ts >= cascade_start:
                    # 1초 bin 값과 1초 스케일 통계 비교
                    z_score = (val - self.mu_short) / max(self.sigma_short, 1e-9)
                    if z_score >= self.config.cascade_z:
                        short_cascade_count += 1
            
            # 방향별 캐스케이드 상태 업데이트 (20~30초 한쪽 지분 ≥0.85 & 이벤트 ≥2)
            long_cascade = long_cascade_count >= self.config.cascade_count
            short_cascade = short_cascade_count >= self.config.cascade_count
            
            # 전체 캐스케이드 상태 (하위 호환성)
            if long_cascade or short_cascade:
                if not self.cascade_detected:
                    self.cascade_detected = True
                    self.cascade_start_time = datetime.now(timezone.utc)
            else:
                # 캐스케이드 종료 확인 (30초 후)
                if (self.cascade_detected and self.cascade_start_time and 
                    (datetime.now(timezone.utc) - self.cascade_start_time).total_seconds() > 30):
                    self.cascade_detected = False
                    self.cascade_start_time = None
            
            return {
                'long_cascade': long_cascade,
                'short_cascade': short_cascade,
                'total_cascade': long_cascade or short_cascade
            }
                
        except Exception as e:
            print(f"❌ 캐스케이드 감지 오류: {e}")
            return {'long_cascade': False, 'short_cascade': False, 'total_cascade': False}
    
    def _is_cooldown_active(self, current_time: datetime, signal_side: str = None) -> Dict[str, Any]:
        """쿨다운 상태 확인 (방향별 감점/강등)"""
        if not self.last_strong_spike_time:
            return {'active': False, 'penalty': 0.0, 'reason': None}
        
        time_since_spike = (current_time - self.last_strong_spike_time).total_seconds()
        
        # 방향별 쿨다운 확인
        cooldown_active = False
        penalty = 0.0
        reason = None
        
        # 강한 스파이크 (z >= 3.5) 쿨다운: ENTRY 제한/SETUP 허용
        if hasattr(self, 'last_spike_strength') and self.last_spike_strength >= 3.5:
            if time_since_spike < self.config.cooldown_after_strong_sec:
                cooldown_active = True
                penalty = 0.3  # 강한 스파이크 후 ENTRY 제한
                reason = f"강한 스파이크 쿨다운 - ENTRY 제한/SETUP 허용 ({time_since_spike:.1f}s)"
        
        # 중간 스파이크 (z >= 3.0) 쿨다운
        elif hasattr(self, 'last_spike_strength') and self.last_spike_strength >= 3.0:
            if time_since_spike < self.config.cooldown_after_medium_sec:
                cooldown_active = True
                penalty = 0.1  # 중간 스파이크 후 감점
                reason = f"중간 스파이크 쿨다운 ({time_since_spike:.1f}s)"
        
        # 기본 쿨다운 (하위 호환성)
        elif time_since_spike < self.config.cooldown_after_strong_sec:
            cooldown_active = True
            penalty = 0.15
            reason = f"기본 쿨다운 ({time_since_spike:.1f}s)"
        
        return {
            'active': cooldown_active,
            'penalty': penalty,
            'reason': reason,
            'time_since_spike': time_since_spike
        }
    
    def analyze_strategy_a_sweep_reclaim(self, 
                                        metrics: Dict[str, Any],
                                        price_data: pd.DataFrame,
                                        key_levels: Dict[str, float],
                                        atr: float) -> Optional[Dict]:
        """전략 A: 스윕&리클레임 분석 (스코어링 방식)"""
        try:
            current_price = price_data['close'].iloc[-1]
            prev_day_low = key_levels.get('prev_day_low')
            prev_day_high = key_levels.get('prev_day_high')
            
            signals = []
            
            # === 롱 신호 후보 생성 ===
            # 최근 N봉 내 레벨 스윕 + 재진입/재정착 확인
            swept_recently = False
            N = 20
            for i in range(1, min(N + 1, len(price_data))):
                low_price = price_data['low'].iloc[-i]
                if prev_day_low > 0 and low_price < prev_day_low:
                    swept_recently = True
                    break
            reentered = False
            if prev_day_low > 0 and len(price_data) >= 2:
                prev_close = price_data['close'].iloc[-2]
                curr_close = price_data['close'].iloc[-1]
                atr_buffer = atr * 0.5
                reentered = (
                    (prev_close < prev_day_low and curr_close > prev_day_low) or
                    (prev_close < prev_day_low and curr_close > prev_day_low - atr_buffer)
                )
            if prev_day_low > 0 and swept_recently and reentered:
                # 1. Gate 조건 확인 (롱 신호용)
                gate_conditions = self.check_gate_conditions(price_data, atr, current_price, 'BUY')
                if not gate_conditions['gate_passed']:
                    self.log_strategy_diagnosis('A', metrics, f"롱 신호 Gate 실패: {gate_conditions.get('block_reason', '알 수 없음')}")
                else:
                    # 기본 조건 확인
                    z_long = metrics.get('z_long', 0)
                    lpi = metrics.get('lpi', 0)
                    
                    if z_long >= self.config.z_spike and lpi <= -self.config.lpi_bias:  # HEADS-UP 기준 (0.6)
                        # 신호 생성
                        entry_price = current_price
                        stop_loss = min(prev_day_low, current_price) - atr * 0.3
                        backup_stop = current_price * 0.9992
                        stop_loss = min(stop_loss, backup_stop)
                        
                        risk = entry_price - stop_loss
                        tp1 = entry_price + risk * self.config.tp1_R_a
                        
                        if "VWAP" in self.config.tp2:
                            tp2 = key_levels.get('vwap', entry_price + risk * 2.0)
                        else:
                            tp2 = entry_price + risk * 2.0
                        
                        # 스코어링 및 Tier 결정
                        scoring_result = self.calculate_total_score(
                            'A', 'BUY', price_data, key_levels, {}, atr, 
                            entry_price, stop_loss, tp1, metrics
                        )
                        
                        tier_result = self.determine_signal_tier(
                            scoring_result['total_score'], 'A', metrics, atr
                        )
                        
                        # 후보 상세 로그
                        self.log_candidate_details('A', 'BUY', metrics, price_data, key_levels, atr)
                        
                        # 스코어링 결과 로그
                        self.log_scoring_results('A', 'BUY', scoring_result, tier_result)
                        
                        # 신호 생성
                        signal = {
                            'signal_type': 'SWEEP_RECLAIM_LONG',
                            'action': 'BUY',
                            'confidence': scoring_result['total_score'],
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit1': tp1,
                            'take_profit2': tp2,
                            'risk_reward': self.config.tp1_R_a,
                            'timestamp': datetime.now(timezone.utc),
                            'reason': f"하단 스윕 + 롱청산스파이크 | Z:{z_long:.1f}, LPI:{lpi:.2f}",
                            'playbook': 'A',
                            'liquidation_metrics': metrics,
                            'total_score': scoring_result['total_score'],
                            'tier': tier_result['final_tier'],
                            'component_scores': scoring_result['component_scores']
                        }
                        
                        signals.append(signal)
            
            # === 숏 신호 후보 생성 ===
            # 최근 N봉 내 레벨 스윕 + 재진입/재정착 확인
            swept_recently_h = False
            for i in range(1, min(N + 1, len(price_data))):
                high_price = price_data['high'].iloc[-i]
                if prev_day_high > 0 and high_price > prev_day_high:
                    swept_recently_h = True
                    break
            reentered_h = False
            if prev_day_high > 0 and len(price_data) >= 2:
                prev_close = price_data['close'].iloc[-2]
                curr_close = price_data['close'].iloc[-1]
                atr_buffer = atr * 0.5
                reentered_h = (
                    (prev_close > prev_day_high and curr_close < prev_day_high) or
                    (prev_close > prev_day_high and curr_close < prev_day_high + atr_buffer)
                )
            if prev_day_high > 0 and swept_recently_h and reentered_h:
                # 1. Gate 조건 확인 (숏 신호용)
                gate_conditions = self.check_gate_conditions(price_data, atr, current_price, 'SELL')
                if not gate_conditions['gate_passed']:
                    self.log_strategy_diagnosis('A', metrics, f"숏 신호 Gate 실패: {gate_conditions.get('block_reason', '알 수 없음')}")
                else:
                    # 기본 조건 확인
                    z_short = metrics.get('z_short', 0)
                    lpi = metrics.get('lpi', 0)
                    
                    if z_short >= self.config.z_spike and lpi >= self.config.lpi_bias:  # HEADS-UP 기준 (0.6)
                        # 신호 생성
                        entry_price = current_price
                        stop_loss = max(prev_day_high, current_price) + atr * 0.3
                        backup_stop = current_price * 1.0008
                        stop_loss = max(stop_loss, backup_stop)
                        
                        risk = stop_loss - entry_price
                        tp1 = entry_price - risk * self.config.tp1_R_a
                        
                        if "VWAP" in self.config.tp2:
                            tp2 = key_levels.get('vwap', entry_price - risk * 2.0)
                        else:
                            tp2 = entry_price - risk * 2.0
                        
                        # 스코어링 및 Tier 결정
                        scoring_result = self.calculate_total_score(
                            'A', 'SELL', price_data, key_levels, {}, atr, 
                            entry_price, stop_loss, tp1, metrics
                        )
                        
                        tier_result = self.determine_signal_tier(
                            scoring_result['total_score'], 'A', metrics, atr
                        )
                        
                        # 후보 상세 로그
                        self.log_candidate_details('A', 'SELL', metrics, price_data, key_levels, atr)
                        
                        # 스코어링 결과 로그
                        self.log_scoring_results('A', 'SELL', scoring_result, tier_result)
                        
                        # 신호 생성
                        signal = {
                            'signal_type': 'SWEEP_RECLAIM_SHORT',
                            'action': 'SELL',
                            'confidence': scoring_result['total_score'],
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit1': tp1,
                            'take_profit2': tp2,
                            'risk_reward': self.config.tp1_R_a,
                            'timestamp': datetime.now(timezone.utc),
                            'reason': f"상단 스윕 + 숏청산스파이크 | Z:{z_short:.1f}, LPI:{lpi:.2f}",
                            'playbook': 'A',
                            'liquidation_metrics': metrics,
                            'total_score': scoring_result['total_score'],
                            'tier': tier_result['final_tier'],
                            'component_scores': scoring_result['component_scores']
                        }
                        
                        signals.append(signal)
            
            # 3. 신호 선택 (가장 높은 점수)
            if signals:
                best_signal = max(signals, key=lambda x: x['total_score'])
                print(f"🎯 전략 A 최종 신호: {best_signal['action']} (점수: {best_signal['total_score']:.3f}, Tier: {best_signal['tier']})")
                return best_signal
            
            # HEADS-UP 강제 출력 경로
            z_long = metrics.get('z_long', 0)
            z_short = metrics.get('z_short', 0)
            lpi = metrics.get('lpi', 0)
            if max(z_long, z_short) >= self.config.z_spike or abs(lpi) >= self.config.lpi_bias or metrics.get('is_cascade', False):  # HEADS-UP 기준 (0.6)
                self.log_strategy_diagnosis('A', metrics, "레벨 스윕/리클레임 미충족이나 약한 스파이크/LPI/캐스케이드 감지 → HEADS_UP")
                return {
                    'signal_type': 'SWEEP_RECLAIM_HEADS_UP',
                    'action': 'BUY' if z_short >= z_long else 'SELL',
                    'confidence': 0.1,
                    'entry_price': current_price,
                    'stop_loss': current_price,
                    'take_profit1': current_price,
                    'take_profit2': current_price,
                    'risk_reward': 0.0,
                    'timestamp': datetime.now(timezone.utc),
                    'reason': '관찰 필요: 약한 스파이크/LPI/캐스케이드',
                    'playbook': 'A',
                    'liquidation_metrics': metrics,
                    'total_score': 0.12,
                    'tier': 'HEADS_UP',
                    'component_scores': {}
                }
            return None
            
        except Exception as e:
            print(f"❌ 스윕&리클레임 분석 오류: {e}")
            return None
    
    def analyze_strategy_b_squeeze_trend_continuation(self,
                                                    metrics: Dict[str, Any],
                                                    price_data: pd.DataFrame,
                                                    opening_range: Dict[str, float],
                                                    atr: float) -> Optional[Dict]:
        """전략 B: 스퀴즈 추세지속 분석 (스코어링 방식)"""
        try:
            # 오프닝 레인지 필요: 없으면 준OR(최근 60분 rolling range)로 폴백
            if not opening_range or opening_range.get('high', 0) == 0 or opening_range.get('low', 0) == 0:
                if len(price_data) >= 60:
                    recent_high = price_data['high'].iloc[-60:].max()
                    recent_low = price_data['low'].iloc[-60:].min()
                    opening_range = {'high': float(recent_high), 'low': float(recent_low)}
                    self.log_strategy_diagnosis('B', metrics, f"준OR 사용: high={recent_high:.2f}, low={recent_low:.2f}")
                else:
                    self.log_strategy_diagnosis('B', metrics, "OR/준OR 데이터 부족")
                    return None
            
            current_price = price_data['close'].iloc[-1]
            or_high = opening_range.get('high', 0)
            or_low = opening_range.get('low', 0)
            
            signals = []
            
            # === 롱 신호 후보 생성 ===
            if or_high > 0 and current_price > or_high:  # 상단 돌파
                # 1. Gate 조건 확인 (롱 신호용)
                gate_conditions = self.check_gate_conditions(price_data, atr, current_price, 'BUY')
                if not gate_conditions['gate_passed']:
                    self.log_strategy_diagnosis('B', metrics, f"롱 신호 Gate 실패: {gate_conditions.get('block_reason', '알 수 없음')}")
                else:
                    # 기본 조건 확인
                    z_short = metrics.get('z_short', 0)
                    lpi = metrics.get('lpi', 0)
                    
                    if z_short >= self.config.z_spike and lpi >= self.config.lpi_bias:  # HEADS-UP 기준 (0.6)
                        # 리테스트 확인
                        retest_found = False
                        retest_low = current_price
                        
                        if len(price_data) >= 4:
                            retest_tolerance = (self.config.retest_atr_tol_or_extension if self.config.or_extension 
                                              else self.config.retest_atr_tol)
                            
                            for i in range(1, min(11, len(price_data))):
                                low_price = price_data['low'].iloc[-i]
                                if low_price < or_high and low_price >= or_high - atr * retest_tolerance:
                                    retest_found = True
                                    retest_low = min(retest_low, low_price)
                                    break
                        
                        if retest_found:
                            # 신호 생성
                            entry_price = current_price
                            stop_loss = retest_low - atr * 0.5
                            
                            risk = entry_price - stop_loss
                            tp1 = entry_price + risk * self.config.tp1_R_b
                            
                            if self.config.or_extension:
                                or_range = or_high - or_low
                                tp2 = or_high + or_range
                            else:
                                tp2 = entry_price + risk * 2.5
                            
                            # 스코어링 및 Tier 결정
                            scoring_result = self.calculate_total_score(
                                'B', 'BUY', price_data, {}, opening_range, atr, 
                                entry_price, stop_loss, tp1, metrics
                            )
                            
                            tier_result = self.determine_signal_tier(
                                scoring_result['total_score'], 'B', metrics, atr
                            )
                            
                            # 후보 상세 로그
                            self.log_candidate_details('B', 'BUY', metrics, price_data, {}, atr)
                            
                            # 스코어링 결과 로그
                            self.log_scoring_results('B', 'BUY', scoring_result, tier_result)
                            
                            # 신호 생성
                            signal = {
                                'signal_type': 'SQUEEZE_TREND_CONTINUATION_LONG',
                                'action': 'BUY',
                                'confidence': scoring_result['total_score'],
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'take_profit1': tp1,
                                'take_profit2': tp2,
                                'risk_reward': self.config.tp1_R_b,
                                'timestamp': datetime.now(timezone.utc),
                                'reason': f"상단 돌파 + 숏청산스파이크 + 리테스트 | Z:{z_short:.1f}, LPI:{lpi:.2f}",
                                'playbook': 'B',
                                'liquidation_metrics': metrics,
                                'total_score': scoring_result['total_score'],
                                'tier': tier_result['final_tier'],
                                'component_scores': scoring_result['component_scores']
                            }
                            
                            signals.append(signal)
            
            # === 숏 신호 후보 생성 ===
            if or_low > 0 and current_price < or_low:  # 하단 이탈
                # 1. Gate 조건 확인 (숏 신호용)
                gate_conditions = self.check_gate_conditions(price_data, atr, current_price, 'SELL')
                if not gate_conditions['gate_passed']:
                    self.log_strategy_diagnosis('B', metrics, f"숏 신호 Gate 실패: {gate_conditions.get('block_reason', '알 수 없음')}")
                else:
                    # 기본 조건 확인
                    z_long = metrics.get('z_long', 0)
                    lpi = metrics.get('lpi', 0)
                    
                    if z_long >= self.config.z_spike and lpi <= -self.config.lpi_bias:  # HEADS-UP 기준 (0.6)
                        # 리테스트 확인
                        retest_found = False
                        retest_high = current_price
                        
                        if len(price_data) >= 4:
                            retest_tolerance = (self.config.retest_atr_tol_or_extension if self.config.or_extension 
                                              else self.config.retest_atr_tol)
                            
                            for i in range(1, min(11, len(price_data))):
                                high_price = price_data['high'].iloc[-i]
                                if high_price > or_low and high_price <= or_low + atr * retest_tolerance:
                                    retest_found = True
                                    retest_high = max(retest_high, high_price)
                                    break
                        
                        if retest_found:
                            # 신호 생성
                            entry_price = current_price
                            stop_loss = retest_high + atr * 0.5
                            
                            risk = stop_loss - entry_price
                            tp1 = entry_price - risk * self.config.tp1_R_b
                            
                            if self.config.or_extension:
                                or_range = or_high - or_low
                                tp2 = or_low - or_range
                            else:
                                tp2 = entry_price - risk * 2.5
                            
                            # 스코어링 및 Tier 결정
                            scoring_result = self.calculate_total_score(
                                'B', 'SELL', price_data, {}, opening_range, atr, 
                                entry_price, stop_loss, tp1, metrics
                            )
                            
                            tier_result = self.determine_signal_tier(
                                scoring_result['total_score'], 'B', metrics, atr
                            )
                            
                            # 후보 상세 로그
                            self.log_candidate_details('B', 'SELL', metrics, price_data, {}, atr)
                            
                            # 스코어링 결과 로그
                            self.log_scoring_results('B', 'SELL', scoring_result, tier_result)
                            
                            # 신호 생성
                            signal = {
                                'signal_type': 'SQUEEZE_TREND_CONTINUATION_SHORT',
                                'action': 'SELL',
                                'confidence': scoring_result['total_score'],
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'take_profit1': tp1,
                                'take_profit2': tp2,
                                'risk_reward': self.config.tp1_R_b,
                                'timestamp': datetime.now(timezone.utc),
                                'reason': f"하단 이탈 + 롱청산스파이크 + 리테스트 | Z:{z_long:.1f}, LPI:{lpi:.2f}",
                                'playbook': 'B',
                                'liquidation_metrics': metrics,
                                'total_score': scoring_result['total_score'],
                                'tier': tier_result['final_tier'],
                                'component_scores': scoring_result['component_scores']
                            }
                            
                            signals.append(signal)
            
            # 3. 신호 선택 (가장 높은 점수)
            if signals:
                best_signal = max(signals, key=lambda x: x['total_score'])
                print(f"🎯 전략 B 최종 신호: {best_signal['action']} (점수: {best_signal['total_score']:.3f}, Tier: {best_signal['tier']})")
                return best_signal
            
            # HEADS-UP 강제 출력 경로
            z_long = metrics.get('z_long', 0)
            z_short = metrics.get('z_short', 0)
            lpi = metrics.get('lpi', 0)
            if max(z_long, z_short) >= self.config.z_spike or abs(lpi) >= self.config.lpi_bias or metrics.get('is_cascade', False):  # HEADS-UP 기준 (0.6)
                self.log_strategy_diagnosis('B', metrics, "OR 돌파/리테스트 미충족이나 약한 스파이크/LPI/캐스케이드 감지 → HEADS_UP")
                return {
                    'signal_type': 'SQUEEZE_TREND_HEADS_UP',
                    'action': 'BUY' if z_short >= z_long else 'SELL',
                    'confidence': 0.1,
                    'entry_price': current_price,
                    'stop_loss': current_price,
                    'take_profit1': current_price,
                    'take_profit2': current_price,
                    'risk_reward': 0.0,
                    'timestamp': datetime.now(timezone.utc),
                    'reason': '관찰 필요: 약한 스파이크/LPI/캐스케이드',
                    'playbook': 'B',
                    'liquidation_metrics': metrics,
                    'total_score': 0.12,
                    'tier': 'HEADS_UP',
                    'component_scores': {}
                }
            return None
            
        except Exception as e:
            print(f"❌ 스퀴즈 추세지속 분석 오류: {e}")
            return None
    
    def analyze_strategy_c_overheat_extinction_fade(self,
                                                    metrics: Dict[str, Any],
                                                    price_data: pd.DataFrame,
                                                    vwap: float,
                                                    vwap_std: float,
                                                    atr: float) -> Optional[Dict]:
        """전략 C: 과열-소멸 페이드 분석 (스코어링 방식)"""
        try:
            current_price = price_data['close'].iloc[-1]
            
            # VWAP σ 임계 결정 (다중 경로: 극단 스파이크, 캐스케이드, 기본)
            is_cascade = metrics.get('is_cascade', False)
            z_long = metrics.get('z_long', 0)
            z_short = metrics.get('z_short', 0)
            lpi = metrics.get('lpi', 0)
            
            # 극단 스파이크 확인 (z≥3.5 & LPI 극단)
            extreme_spike = (max(z_long, z_short) >= self.config.z_extreme and 
                           abs(lpi) >= self.config.lpi_extreme)
            
            # σ 임계 결정 (우선순위: 극단 > 캐스케이드 > 기본)
            if extreme_spike:
                vwap_sd_threshold = self.config.vwap_sd_extreme
            elif is_cascade:
                vwap_sd_threshold = self.config.vwap_sd_enter_cascade
            else:
                vwap_sd_threshold = self.config.vwap_sd_enter
            
            # VWAP ±σ 바깥 확인
            vwap_lower = vwap - vwap_sd_threshold * vwap_std
            vwap_upper = vwap + vwap_sd_threshold * vwap_std
            
            price_outside_vwap = current_price < vwap_lower or current_price > vwap_upper
            
            if not price_outside_vwap:
                # 다단계: 내부면 후보 생성 안 함
                return None
            
            signals = []
            
            # === 롱 페이드 후보 생성 ===
            if current_price < vwap_lower:  # 하락 과열
                # 1. Gate 조건 확인 (롱 신호용)
                gate_conditions = self.check_gate_conditions(price_data, atr, current_price, 'BUY')
                if not gate_conditions['gate_passed']:
                    self.log_strategy_diagnosis('C', metrics, f"롱 신호 Gate 실패: {gate_conditions.get('block_reason', '알 수 없음')}")
                else:
                    # 다단계: z_long 임계별 HEADS-UP/SETUP/ENTRY
                    tier_hint = None
                    if z_long >= self.config.z_entry or is_cascade:  # ENTRY 기준 (2.5)
                        tier_hint = 'ENTRY'
                    elif z_long >= self.config.z_setup:  # SETUP 기준 (2.0)
                        tier_hint = 'SETUP'
                    elif z_long >= self.config.z_spike:  # HEADS-UP 기준 (0.6)
                        tier_hint = 'HEADS_UP'
                    # 기본: 1.8σ에서 SETUP 허용
                    elif abs(current_price - vwap) >= 1.8 * vwap_std:
                        tier_hint = 'SETUP'
                    if tier_hint:
                        # 신호 생성
                        entry_price = current_price
                        stop_loss = max(
                            current_price - atr * self.config.stop_atr,
                            vwap - self.config.vwap_sd_stop * vwap_std
                        )
                        
                        risk = entry_price - stop_loss
                        tp1 = vwap  # VWAP 터치
                        tp2 = vwap + self.config.tp2_sigma * vwap_std
                        
                        # 스코어링 및 Tier 결정
                        scoring_result = self.calculate_total_score(
                            'C', 'BUY', price_data, {'vwap': vwap, 'vwap_std': vwap_std}, {}, atr, 
                            entry_price, stop_loss, tp1, metrics
                        )
                        
                        tier_result = self.determine_signal_tier(
                            scoring_result['total_score'], 'C', metrics, atr
                        )
                        # 힌트에 따른 최소 Tier 보정
                        if tier_hint == 'HEADS_UP' and tier_result['final_tier'] == 'REJECT':
                            tier_result['final_tier'] = 'HEADS_UP'
                        if tier_hint == 'SETUP' and tier_result['final_tier'] in ['REJECT', 'HEADS_UP']:
                            tier_result['final_tier'] = 'SETUP'
                        
                        # 후보 상세 로그
                        self.log_candidate_details('C', 'BUY', metrics, price_data, {'vwap': vwap, 'vwap_std': vwap_std}, atr)
                        
                        # 스코어링 결과 로그
                        self.log_scoring_results('C', 'BUY', scoring_result, tier_result)
                        
                        # 신호 생성
                        signal = {
                            'signal_type': 'OVERHEAT_EXTINCTION_FADE_LONG',
                            'action': 'BUY',
                            'confidence': scoring_result['total_score'],
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit1': tp1,
                            'take_profit2': tp2,
                            'risk_reward': self.config.tp1_R_c,
                            'timestamp': datetime.now(timezone.utc),
                            'reason': f"VWAP -{vwap_sd_threshold:.1f}σ + 롱청산스파이크 | Z:{z_long:.1f}",
                            'playbook': 'C',
                            'liquidation_metrics': metrics,
                            'total_score': scoring_result['total_score'],
                            'tier': tier_result['final_tier'],
                            'component_scores': scoring_result['component_scores']
                        }
                        
                        signals.append(signal)
            
            # === 숏 페이드 후보 생성 ===
            elif current_price > vwap_upper:  # 상승 과열
                # 1. Gate 조건 확인 (숏 신호용)
                gate_conditions = self.check_gate_conditions(price_data, atr, current_price, 'SELL')
                if not gate_conditions['gate_passed']:
                    self.log_strategy_diagnosis('C', metrics, f"숏 신호 Gate 실패: {gate_conditions.get('block_reason', '알 수 없음')}")
                else:
                    # 다단계: z_short 임계별 HEADS-UP/SETUP/ENTRY
                    tier_hint = None
                    if z_short >= self.config.z_entry or is_cascade:  # ENTRY 기준 (2.5)
                        tier_hint = 'ENTRY'
                    elif z_short >= self.config.z_setup:  # SETUP 기준 (2.0)
                        tier_hint = 'SETUP'
                    elif z_short >= self.config.z_spike:  # HEADS-UP 기준 (0.6)
                        tier_hint = 'HEADS_UP'
                    # 기본: 1.8σ에서 SETUP 허용
                    elif abs(current_price - vwap) >= 1.8 * vwap_std:
                        tier_hint = 'SETUP'
                    if tier_hint:
                        # 신호 생성
                        entry_price = current_price
                        stop_loss = min(
                            current_price + atr * self.config.stop_atr,
                            vwap + self.config.vwap_sd_stop * vwap_std
                        )
                        
                        risk = stop_loss - entry_price
                        tp1 = vwap  # VWAP 터치
                        tp2 = vwap - self.config.tp2_sigma * vwap_std
                        
                        # 스코어링 및 Tier 결정
                        scoring_result = self.calculate_total_score(
                            'C', 'SELL', price_data, {'vwap': vwap, 'vwap_std': vwap_std}, {}, atr, 
                            entry_price, stop_loss, tp1, metrics
                        )
                        
                        tier_result = self.determine_signal_tier(
                            scoring_result['total_score'], 'C', metrics, atr
                        )
                        # 힌트에 따른 최소 Tier 보정
                        if tier_hint == 'HEADS_UP' and tier_result['final_tier'] == 'REJECT':
                            tier_result['final_tier'] = 'HEADS_UP'
                        if tier_hint == 'SETUP' and tier_result['final_tier'] in ['REJECT', 'HEADS_UP']:
                            tier_result['final_tier'] = 'SETUP'
                        
                        # 후보 상세 로그
                        self.log_candidate_details('C', 'SELL', metrics, price_data, {'vwap': vwap, 'vwap_std': vwap_std}, atr)
                        
                        # 스코어링 결과 로그
                        self.log_scoring_results('C', 'SELL', scoring_result, tier_result)
                        
                        # 신호 생성
                        signal = {
                            'signal_type': 'OVERHEAT_EXTINCTION_FADE_SHORT',
                            'action': 'SELL',
                            'confidence': scoring_result['total_score'],
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit1': tp1,
                            'take_profit2': tp2,
                            'risk_reward': self.config.tp1_R_c,
                            'timestamp': datetime.now(timezone.utc),
                            'reason': f"VWAP +{vwap_sd_threshold:.1f}σ + 숏청산스파이크 | Z:{z_short:.1f}",
                            'playbook': 'C',
                            'liquidation_metrics': metrics,
                            'total_score': scoring_result['total_score'],
                            'tier': tier_result['final_tier'],
                            'component_scores': scoring_result['component_scores']
                        }
                        
                        signals.append(signal)
            
            # 3. 신호 선택 (가장 높은 점수)
            if signals:
                best_signal = max(signals, key=lambda x: x['total_score'])
                print(f"🎯 전략 C 최종 신호: {best_signal['action']} (점수: {best_signal['total_score']:.3f}, Tier: {best_signal['tier']})")
                return best_signal
            
            # HEADS-UP 강제 출력 경로
            z_long = metrics.get('z_long', 0)
            z_short = metrics.get('z_short', 0)
            lpi = metrics.get('lpi', 0)
            if max(z_long, z_short) >= self.config.z_spike or abs(lpi) >= self.config.lpi_bias or metrics.get('is_cascade', False):  # HEADS-UP 기준 (0.6)
                self.log_strategy_diagnosis('C', metrics, "VWAP 과열 미충족이나 약한 스파이크/LPI/캐스케이드 감지 → HEADS_UP")
                return {
                    'signal_type': 'VWAP_FADE_HEADS_UP',
                    'action': 'BUY' if z_short >= z_long else 'SELL',
                    'confidence': 0.1,
                    'entry_price': current_price,
                    'stop_loss': current_price,
                    'take_profit1': current_price,
                    'take_profit2': current_price,
                    'risk_reward': 0.0,
                    'timestamp': datetime.now(timezone.utc),
                    'reason': '관찰 필요: 약한 스파이크/LPI/캐스케이드',
                    'playbook': 'C',
                    'liquidation_metrics': metrics,
                    'total_score': 0.12,
                    'tier': 'HEADS_UP',
                    'component_scores': {}
                }
            return None
            
        except Exception as e:
            print(f"❌ 과열-소멸 페이드 분석 오류: {e}")
            return None
    
    def _analyze_long_fade(self, 
                          metrics: Dict[str, Any],
                          price_data: pd.DataFrame,
                          vwap: float,
                          vwap_std: float,
                          atr: float) -> Optional[Dict]:
        """롱 페이드 분석 (하락 과열)"""
        try:
            # 롱 청산 스파이크 확인
            z_long = metrics.get('z_long', 0)
            if z_long < self.config.z_strong:  # 강한 스파이크 필요
                return None
            
            # 스파이크 후 감소 확인
            if not self._check_post_spike_decay(metrics, 'long'):
                return None
            
            # 가격 구조 반전 확인
            if len(price_data) < 3:
                return None
            
            # 저점 갱신 실패 & 고점 돌파
            recent_low = min(price_data['low'].iloc[-3:])
            recent_high = max(price_data['high'].iloc[-3:])
            
            current_price = price_data['close'].iloc[-1]
            current_high = price_data['high'].iloc[-1]
            
            low_failure = current_price > recent_low
            high_breakout = current_high > recent_high
            
            if not (low_failure and high_breakout):
                return None
            
            # 신호 생성
            entry_price = current_price
            stop_loss = max(
                recent_low - atr * self.config.stop_atr,
                vwap - self.config.vwap_sd_stop * vwap_std
            )
            
            risk = entry_price - stop_loss
            tp1 = vwap  # VWAP 터치
            tp2 = vwap + self.config.tp2_sigma * vwap_std
            
            return {
                'signal_type': 'OVERHEAT_EXTINCTION_FADE_LONG',
                'action': 'BUY',
                'confidence': 0.75,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit1': tp1,
                'take_profit2': tp2,
                'risk_reward': self.config.tp1_R_c,
                'timestamp': datetime.now(timezone.utc),
                'reason': f"VWAP -2σ + 롱청산스파이크 + 감소 페이드 롱 | Z:{z_long:.1f}",
                'playbook': 'C',
                'liquidation_metrics': metrics
            }
            
        except Exception as e:
            print(f"❌ 롱 페이드 분석 오류: {e}")
            return None
    
    def _analyze_short_fade(self, 
                           metrics: Dict[str, Any],
                           price_data: pd.DataFrame,
                           vwap: float,
                           vwap_std: float,
                           atr: float) -> Optional[Dict]:
        """숏 페이드 분석 (상승 과열)"""
        try:
            # 숏 청산 스파이크 확인
            z_short = metrics.get('z_short', 0)
            if z_short < self.config.z_strong:  # 강한 스파이크 필요
                return None
            
            # 스파이크 후 감소 확인
            if not self._check_post_spike_decay(metrics, 'short'):
                return None
            
            # 가격 구조 반전 확인
            if len(price_data) < 3:
                return None
            
            # 고점 갱신 실패 & 저점 돌파
            recent_high = max(price_data['high'].iloc[-3:])
            recent_low = min(price_data['low'].iloc[-3:])
            
            current_price = price_data['close'].iloc[-1]
            current_low = price_data['low'].iloc[-1]
            
            high_failure = current_price < recent_high
            low_breakout = current_low < recent_low
            
            if not (high_failure and low_breakout):
                return None
            
            # 신호 생성
            entry_price = current_price
            stop_loss = min(
                recent_high + atr * self.config.stop_atr,
                vwap + self.config.vwap_sd_stop * vwap_std
            )
            
            risk = stop_loss - entry_price
            tp1 = vwap  # VWAP 터치
            tp2 = vwap - self.config.tp2_sigma * vwap_std
            
            return {
                'signal_type': 'OVERHEAT_EXTINCTION_FADE_SHORT',
                'action': 'SELL',
                'confidence': 0.75,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit1': tp1,
                'take_profit2': tp2,
                'risk_reward': self.config.tp1_R_c,
                'timestamp': datetime.now(timezone.utc),
                'reason': f"VWAP +2σ + 숏청산스파이크 + 감소 페이드 숏 | Z:{z_short:.1f}",
                'playbook': 'C',
                'liquidation_metrics': metrics
            }
            
        except Exception as e:
            print(f"❌ 숏 페이드 분석 오류: {e}")
            return None
    
    def _check_post_spike_decay(self, metrics: Dict[str, Any], side: str, for_entry: bool = True) -> bool:
        """스파이크 후 감소 확인 (SETUP/ENTRY 임계값 분리)"""
        try:
            # 10초 평균 청산 계산
            current_time = datetime.now(timezone.utc)
            window_start = int(current_time.timestamp()) - 10
            
            if side == 'long':
                current_10s = sum(val for ts, val in self.long_bins if ts >= window_start)
                # 10초 기준으로 스케일링
                mu_10s = self.mu_long * 10  # 1초 평균 × 10초
            else:  # short
                current_10s = sum(val for ts, val in self.short_bins if ts >= window_start)
                # 10초 기준으로 스케일링
                mu_10s = self.mu_short * 10  # 1초 평균 × 10초
            
            # SETUP/ENTRY 임계값 분리
            if for_entry:
                # ENTRY: 더 엄격한 기준 (0.80)
                base_threshold = 0.80
            else:
                # SETUP: 더 완화된 기준 (0.85)
                base_threshold = 0.85
            
            # 캐스케이드 상태에 따른 감소 기준 적용
            is_cascade = metrics.get('is_cascade', False)
            if is_cascade:
                decay_threshold = base_threshold + 0.05  # 캐스케이드 시 +0.05 완화
            else:
                decay_threshold = base_threshold
            
            # 스파이크 후 감소 확인
            decay_ratio = current_10s / (mu_10s + 1e-9)
            return decay_ratio < decay_threshold
            
        except Exception as e:
            print(f"❌ 스파이크 후 감소 확인 오류: {e}")
            return False
    
    def _check_additional_long_liquidation(self) -> bool:
        """추가 롱 청산 확인 (10초 누적)"""
        try:
            current_time = datetime.now(timezone.utc)
            window_start = int(current_time.timestamp()) - 10
            
            # 10초 누적 롱 청산
            long_10s = sum(val for ts, val in self.long_bins if ts >= window_start)
            
            # 10초 누적에 맞게 스케일링: μ → 10×μ, σ → √10×σ
            scale_factor = 10
            scale_sqrt = np.sqrt(scale_factor)
            mu_scaled = self.mu_long * scale_factor
            sigma_scaled = self.sigma_long * scale_sqrt
            
            # 기본선 + 2σ 확인 (스케일링된 값으로)
            threshold = mu_scaled + 2 * sigma_scaled
            
            return long_10s >= threshold
            
        except Exception as e:
            print(f"❌ 추가 롱 청산 확인 오류: {e}")
            return False
    
    def _check_additional_short_liquidation(self) -> bool:
        """추가 숏 청산 확인 (10초 누적)"""
        try:
            current_time = datetime.now(timezone.utc)
            window_start = int(current_time.timestamp()) - 10
            
            # 10초 누적 숏 청산
            short_10s = sum(val for ts, val in self.short_bins if ts >= window_start)
            
            # 10초 누적에 맞게 스케일링: μ → 10×μ, σ → √10×σ
            scale_factor = 10
            scale_sqrt = np.sqrt(scale_factor)
            mu_scaled = self.mu_short * scale_factor
            sigma_scaled = self.sigma_short * scale_sqrt
            
            # 기본선 + 2σ 확인 (스케일링된 값으로)
            threshold = mu_scaled + 2 * sigma_scaled
            
            return short_10s > threshold
            
        except Exception as e:
            print(f"❌ 추가 숏 청산 확인 오류: {e}")
            return False
    
    def analyze_all_strategies(self,
                                price_data: pd.DataFrame,
                                key_levels: Dict[str, float],
                                opening_range: Dict[str, float],
                                vwap: float,
                                vwap_std: float,
                                atr: float) -> Optional[Dict]:
        """모든 전략 분석 (스코어링 + 충돌 해결) - post-Gate 요약"""
        try:
            # 현재 청산 지표 가져오기
            metrics = self.get_current_liquidation_metrics()
            if not metrics:
                return None
            
            # 강한 스파이크 감지 시 쿨다운 시작
            z_long = metrics.get('z_long', 0)
            z_short = metrics.get('z_short', 0)
            max_z = max(z_long, z_short)
            
            if max_z >= self.config.z_medium:
                self.last_strong_spike_time = datetime.now(timezone.utc)
                self.last_spike_strength = max_z  # 스파이크 강도 기록
            
            # 모든 전략에서 신호 후보 수집 (pre-Gate)
            all_candidates = []
            
            # 전략 A: 스윕&리클레임
            signal_a = self.analyze_strategy_a_sweep_reclaim(
                metrics, price_data, key_levels, atr
            )
            if signal_a:
                all_candidates.append(signal_a)
            
            # 전략 B: 스퀴즈 추세지속
            signal_b = self.analyze_strategy_b_squeeze_trend_continuation(
                metrics, price_data, opening_range, atr
            )
            if signal_b:
                all_candidates.append(signal_b)
            
            # 전략 C: 과열-소멸 페이드
            signal_c = self.analyze_strategy_c_overheat_extinction_fade(
                metrics, price_data, vwap, vwap_std, atr
            )
            if signal_c:
                all_candidates.append(signal_c)
            
            # 신호가 없으면 중립 반환
            if not all_candidates:
                return {
                    'action': 'NEUTRAL',
                    'playbook': 'NO_SIGNAL',
                    'tier': 'NEUTRAL',
                    'total_score': 0.0,
                    'reason': '모든 전략에서 신호 없음',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            
            # post-Gate 신호들만 필터링 (Gate 통과한 신호들)
            post_gate_signals = []
            for candidate in all_candidates:
                if candidate.get('tier') in ['ENTRY', 'SETUP', 'HEADS_UP']:
                    post_gate_signals.append(candidate)
            
            # post-Gate 신호가 없으면 중립 반환
            if not post_gate_signals:
                return {
                    'action': 'NEUTRAL',
                    'playbook': 'GATE_BLOCKED',
                    'tier': 'NEUTRAL',
                    'total_score': 0.0,
                    'reason': '모든 신호가 Gate에서 차단됨',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'candidates': all_candidates
                }
            
            # 동시양방향 충돌 해결 (post-Gate 기준)
            long_signals = [s for s in post_gate_signals if s['action'] == 'BUY']
            short_signals = [s for s in post_gate_signals if s['action'] == 'SELL']
            
            if long_signals and short_signals:
                # 충돌 해결
                best_long = max(long_signals, key=lambda x: x['total_score'])
                best_short = max(short_signals, key=lambda x: x['total_score'])
                
                conflict_result = self.check_conflict_resolution(best_long, best_short)
                
                if conflict_result['conflict']:
                    return {
                        'action': 'NEUTRAL',
                        'playbook': 'CONFLICT_RESOLUTION',
                        'tier': 'NEUTRAL',
                        'total_score': 0.0,
                        'reason': f"동시양방향 충돌: {conflict_result['resolution']}",
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'long_signal': best_long,
                        'short_signal': best_short,
                        'candidates': all_candidates
                    }
                else:
                    # 승자 신호 반환
                    winner = best_long if best_long['total_score'] > best_short['total_score'] else best_short
                    winner['candidates'] = all_candidates  # 후보 정보 추가
                    return winner
            
            # 단일 방향 신호들 중 최고 점수 선택 (post-Gate)
            if post_gate_signals:
                best_signal = max(post_gate_signals, key=lambda x: x['total_score'])
                best_signal['candidates'] = all_candidates  # 후보 정보 추가
                return best_signal
            
            return None
            
        except Exception as e:
            print(f"❌ 전체 전략 분석 오류: {e}")
            return None
    
    def _fallback_vwap_std(self, df: pd.DataFrame, lookback: int = 120) -> Tuple[float, float]:
        """VWAP와 표준편차를 직접 계산하여 기본값 교체"""
        try:
            if len(df) < lookback:
                lookback = len(df)
            
            # 세션 앵커드 VWAP 계산
            pv = (df['close'] * df['volume']).cumsum()
            v = df['volume'].cumsum().replace(0, np.nan)
            vwap_series = pv / v
            
            # VWAP 대비 편차 계산
            dev = df['close'] - vwap_series
            
            # 최근 lookback 기간의 표준편차 계산
            recent_dev = dev.tail(lookback).dropna()
            if len(recent_dev) == 0:
                return df['close'].iloc[-1], df['close'].iloc[-1] * 0.005  # 기본값 0.5%
            
            std = float(recent_dev.std(ddof=0))
            vwap = float(vwap_series.iloc[-1])
            
            # 너무 작을 때 최소 바닥(0.1%) 부여
            min_std = df['close'].iloc[-1] * 0.001
            final_std = max(std, min_std)
            
            return vwap, final_std
            
        except Exception as e:
            print(f"❌ VWAP 표준편차 계산 오류: {e}")
            # 기본값 반환
            return df['close'].iloc[-1], df['close'].iloc[-1] * 0.005
    
    
    def analyze_bucket_liquidations(self, bucket_data: List[Dict]) -> Optional[Dict]:
            """60초 버킷 데이터 분석
            - 기본: 버킷 기반 오더플로우 메트릭만으로 HEADS_UP/SETUP을 생성
            - 확장: context(price_data, key_levels, opening_range, vwap, vwap_std, atr)가 주어지면
                    정식 분석 루틴(analyze_all_strategies)으로 위임하여 ENTRY까지 평가
            """
            if bucket_data:
                # 버킷 데이터로 메트릭 계산
                metrics = self._calculate_bucket_metrics(bucket_data)

                # 워밍업 체크
                if not self._check_basic_warmup(metrics):
                    return None

                # Z점수 및 LPI 계산 (USD 노션널 기반, 60초 스케일)
                z_long, z_short, lpi = self._calculate_z_and_lpi(bucket_data)
                metrics.update({
                    'z_long': z_long,
                    'z_short': z_short,
                    'lpi': lpi
                })

                # 캐스케이드/쿨다운 체크
                is_cascade = self._check_cascade_condition(bucket_data)
                metrics['is_cascade'] = is_cascade
                cooldown_info = self._check_cooldown_condition(metrics)
                metrics['cooldown_info'] = cooldown_info
                
                print(f"🔍 버킷 분석: 이벤트 {len(bucket_data)}개, Z_L:{z_long:.2f}, Z_S:{z_short:.2f}, LPI:{lpi:.3f}, cascade={is_cascade}")
                
                # 🚫 고급청산전략 차단 조건 체크
                if self._should_block_strategy(cooldown_info, z_long, z_short, lpi, is_cascade):
                    print(f"🚫 고급청산전략 차단됨 - 차단 조건 충족")
                    return None

            # 내부에서 지표 데이터 가져오기 (1줄로 간소화)
            results = self.global_manager.get_all_indicators()
            
            vpvr_obj = results.get('vpvr')
            vpvr = vpvr_obj.get_status()

            # 각 지표 객체에서 실제 데이터 가져오기
            key_levels_obj = results.get('daily_levels')  # ✅ 'daily_levels'로 수정
            key_levels = key_levels_obj.get_status()
            key_levels.update({
                'prev_day_high': key_levels.get('prev_day_high'),
                'prev_day_low': key_levels.get('prev_day_low'),
                'poc': vpvr.get('poc'),
                'hvn': vpvr.get('hvn'),
                'lvn': vpvr.get('lvn')
            })
            opening_range_obj = results.get('opening_range')
            opening_range = opening_range_obj.get_status()
                            
            vwap_obj = results.get('vwap').get_status()
            vwap = vwap_obj.get('vwap')
            vwap_std = vwap_obj.get('vwap_std')
            
            atr_obj = results.get('atr')
            atr = atr_obj.get_status().get('current_atr')
            
            data_manager = get_data_manager()
            data = data_manager.get_latest_data(count=200)

            return self.analyze_all_strategies(
                price_data=data,
                key_levels=key_levels,
                opening_range=opening_range,
                vwap=vwap,
                vwap_std=vwap_std,
                atr=atr
                )


    def _calculate_bucket_metrics(self, bucket_data: List[Dict]) -> Dict[str, Any]:
        """버킷 데이터로 기본 메트릭 계산"""
        try:
            total_count = len(bucket_data)
            # side 매핑: SELL(롱 청산) → long, BUY(숏 청산) → short
            long_count = sum(1 for item in bucket_data if item.get('side') == 'SELL')
            short_count = sum(1 for item in bucket_data if item.get('side') == 'BUY')
            
            total_value = sum(item.get('qty_usd', 0) for item in bucket_data)
            long_value = sum(item.get('qty_usd', 0) for item in bucket_data if item.get('side') == 'SELL')
            short_value = sum(item.get('qty_usd', 0) for item in bucket_data if item.get('side') == 'BUY')
        
            return {
                        'total_count': total_count,
                        'long_count': long_count,
                        'short_count': short_count,
                        'total_value': total_value,
                        'long_value': long_value,
                        'short_value': short_value,
                        'long_ratio': long_count / total_count if total_count > 0 else 0,
                        'short_ratio': short_count / total_count if total_count > 0 else 0
                    }
            
        except Exception as e:
            print(f"❌ 버킷 메트릭 계산 오류: {e}")
            return {}
    
    def _calculate_z_and_lpi(self, bucket_data: List[Dict]) -> Tuple[float, float, float]:
        """Z점수와 LPI 계산"""
        try:
            if not bucket_data:
                return 0.0, 0.0, 0.0
            
            # 최근 60초 데이터로 Z점수 계산
            time_manager = get_time_manager()
            current_time = time_manager.get_current_timestamp_int()
            window_start = current_time - 60
            
            # 청산 데이터 side 매핑: SELL(롱 청산) → long, BUY(숏 청산) → short
            recent_long = [item for item in bucket_data if time_manager.get_timestamp_int(item.get('timestamp', 0)) >= window_start and item.get('side') == 'SELL']
            recent_short = [item for item in bucket_data if time_manager.get_timestamp_int(item.get('timestamp', 0)) >= window_start and item.get('side') == 'BUY']
            
            # Z점수 계산 (최근 60초 vs 이전 60초)
            if len(recent_long) > 0 and len(recent_short) > 0:
                z_long = len(recent_long) / max(len(recent_short), 1)
                z_short = len(recent_short) / max(len(recent_long), 1)
            else:
                z_long = len(recent_long) / 10.0  # 기본값
                z_short = len(recent_short) / 10.0  # 기본값
            
            # LPI 계산
            total_recent = len(recent_long) + len(recent_short)
            if total_recent > 0:
                lpi = (len(recent_long) - len(recent_short)) / total_recent
            else:
                lpi = 0.0
            
            return z_long, z_short, lpi
            
        except Exception as e:
            print(f"❌ Z점수/LPI 계산 오류: {e}")
            return 0.0, 0.0, 0.0
    
    def _check_basic_warmup(self, metrics: Dict[str, Any]) -> bool:
        """기본 워밍업 조건 체크"""
        try:
            total_count = metrics.get('total_count', 0)
            return total_count >= 5  # 최소 5개 이벤트 필요
            
        except Exception as e:
            print(f"❌ 워밍업 체크 오류: {e}")
            return False
    
    def _check_cascade_condition(self, bucket_data: List[Dict]) -> bool:
        """캐스케이드 조건 체크"""
        try:
            if len(bucket_data) < 3:
                return False
            
            # 최근 30초 내 같은 방향 청산이 연속으로 발생하는지 체크
            time_manager = get_time_manager()
            current_time = time_manager.get_current_timestamp_int()
            window_start = current_time - 30
            
            recent_data = [item for item in bucket_data if time_manager.get_timestamp_int(item.get('timestamp', 0)) >= window_start]
            
            if len(recent_data) < 3:
                return False
            
            # 같은 방향 청산이 연속으로 발생하는지 확인
            sides = [item.get('side') for item in recent_data]
            if len(sides) >= 3:
                # 최근 3개가 모두 같은 방향인지 체크 (SELL=롱청산, BUY=숏청산)
                if all(side == 'SELL' for side in sides[-3:]) or all(side == 'BUY' for side in sides[-3:]):
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ 캐스케이드 조건 체크 오류: {e}")
            return False
    
    def _should_block_strategy(self, cooldown_info: Dict[str, Any], z_long: float, z_short: float, lpi: float, is_cascade: bool) -> bool:
        """
        고급청산전략을 차단할지 여부 결정
        
        Args:
            cooldown_info: 쿨다운 정보
            z_long: 롱 청산 Z점수
            z_short: 숏 청산 Z점수
            lpi: Liquidation Pressure Index
            is_cascade: 캐스케이드 여부
            
        Returns:
            True: 전략 차단, False: 전략 실행
        """
        # 1. 쿨다운 차단 체크
        if cooldown_info.get('blocked', False):
            print(f"   🚫 쿨다운 차단: {cooldown_info.get('reason', '알 수 없는 이유')}")
            return True
        
        # 2. Z점수 설정값 미달 체크 (z_setup = 1.0)
        z_setup = 1.0
        max_z = max(z_long, z_short)
        if max_z < z_setup:
            print(f"   🚫 Z점수 부족: 최대 Z점수 {max_z:.2f} < 설정값 {z_setup}")
            return True
        
        # 3. LPI 최소값 미달 체크
        lpi_min = self.config.lpi_min  # config에서 가져오기
        if lpi < lpi_min:
            print(f"   🚫 LPI 부족: LPI {lpi:.3f} < 최소값 {lpi_min}")
            return True
        
        # 4. 캐스케이드 차단 체크
        if is_cascade:
            print(f"   🚫 캐스케이드 감지: 전략 차단")
            return True
        
        # 모든 차단 조건을 통과
        return False
    
    def _check_cooldown_condition(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """쿨다운 조건 체크"""
        try:
            cooldown_info = {
                'active': False,
                'penalty': 0.0,
                'reason': ''
            }
            
            # 강한 신호 후 쿨다운
            z_long = metrics.get('z_long', 0)
            z_short = metrics.get('z_short', 0)
            max_z = max(z_long, z_short)
            
            if max_z >= self.config.z_strong:
                cooldown_info.update({
                    'active': True,
                    'penalty': 0.3,
                    'reason': '강한 신호 후 쿨다운'
                })
            elif max_z >= self.config.z_medium:
                cooldown_info.update({
                    'active': True,
                    'penalty': 0.15,
                    'reason': '중간 신호 후 쿨다운'
                })
            
            return cooldown_info
            
        except Exception as e:
            print(f"❌ 쿨다운 조건 체크 오류: {e}")
            return {'active': False, 'penalty': 0.0, 'reason': ''}

