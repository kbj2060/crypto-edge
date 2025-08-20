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
from datetime import datetime, timedelta
from collections import deque
import pytz
from indicators.moving_averages import calculate_ema
from indicators.atr import calculate_atr


@dataclass
class AdvancedLiquidationConfig:
    """고급 청산 전략 설정"""
    # 기본 설정
    symbol: str = "ETHUSDT"
    
    # 청산 데이터 집계 설정
    bin_sec: int = 1  # 1초 bin
    agg_window_sec: int = 30  # 30초 집계 윈도우
    background_window_min: int = 60  # 백그라운드 평균 윈도우 (분)
    
    # 스파이크 판정 설정
    z_spike: float = 3.0  # 기본 스파이크 임계값
    z_strong: float = 4.0  # 강한 스파이크 임계값
    lpi_bias: float = 0.4  # LPI 바이어스 임계값
    
    # 캐스케이드 설정
    cascade_seconds: int = 10  # 지난 10초 안에
    cascade_count: int = 3  # 3회 이상
    cascade_z: float = 3.0  # z >= 3
    
    # 쿨다운 설정
    cooldown_after_strong_sec: int = 30  # 강한 스파이크 후 30초 쿨다운
    
    # 리스크 설정
    risk_pct: float = 0.4  # 1트레이드 계좌대비 위험
    slippage_max_pct: float = 0.02  # 최대 슬리피지
    
    # 레벨 설정
    or_minutes: int = 15  # 오프닝 레인지 분
    atr_len: int = 14  # ATR 기간
    vwap_sd_enter: float = 2.0  # VWAP ±2σ 진입
    vwap_sd_stop: float = 2.5  # VWAP ±2.5σ 스탑
    
    # 전략 A: 스윕&리클레임
    sweep_buffer_atr: float = 0.3  # 스윕 버퍼 ATR
    tp1_R: float = 1.2  # 1차 목표 R
    tp2: str = "VWAP_or_range_edge"  # 2차 목표
    
    # 전략 B: 스퀴즈 추세지속
    retest_atr_tol: float = 0.4  # 리테스트 ATR 허용치
    tp1_R: float = 1.5  # 1차 목표 R
    or_extension: bool = True  # OR 확장 사용
    
    # 전략 C: 과열-소멸 페이드
    post_spike_decay_ratio: float = 0.8  # 스파이크 후 감소 비율
    stop_atr: float = 0.35  # 스탑 ATR
    tp2_sigma: float = 0.5  # 2차 목표 시그마


class AdvancedLiquidationStrategy:
    """고급 청산 분석 전략"""
    
    def __init__(self, config: AdvancedLiquidationConfig):
        self.config = config
        
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
        
        # 세션 필터
        self.session_active = False
        self.session_start_time = None
        
    def process_liquidation_event(self, event: Dict) -> None:
        """청산 이벤트 처리"""
        try:
            timestamp = event.get('ts', 0)
            side = event.get('side', 'unknown')
            qty_usd = event.get('qty_usd', 0.0)
            
            if qty_usd <= 0:
                return
            
            # 현재 시간 계산
            current_time = datetime.fromtimestamp(timestamp)
            
            # 세션 상태 확인
            self._check_session_status(current_time)
            
            # 1초 bin에 추가
            bin_key = int(timestamp)
            
            if side.lower() == 'long':
                self._add_to_bin(self.long_bins, bin_key, qty_usd)
            elif side.lower() == 'short':
                self._add_to_bin(self.short_bins, bin_key, qty_usd)
            
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
            
            # 숏 청산 통계
            short_values = [val for _, val in self.short_bins]
            if short_values:
                self.mu_short = np.mean(short_values)
                self.sigma_short = max(np.std(short_values), 1e-9)
                
        except Exception as e:
            print(f"❌ 백그라운드 통계 업데이트 오류: {e}")
    
    def _check_session_status(self, current_time: datetime) -> None:
        """세션 상태 확인 (런던/NYSE 오픈 ±90분)"""
        try:
            # 런던 오픈 (UTC 08:00)
            london_open = current_time.replace(hour=8, minute=0, second=0, microsecond=0)
            london_start = london_open - timedelta(minutes=90)
            london_end = london_open + timedelta(minutes=90)
            
            # NYSE 오픈 (UTC 13:30)
            nyse_open = current_time.replace(hour=13, minute=30, second=0, microsecond=0)
            nyse_start = nyse_open - timedelta(minutes=90)
            nyse_end = nyse_open + timedelta(minutes=90)
            
            # 세션 활성 상태 확인
            self.session_active = (
                (london_start <= current_time <= london_end) or
                (nyse_start <= current_time <= nyse_end)
            )
            
            # 세션 시작 시간 기록
            if self.session_active and not self.session_start_time:
                self.session_start_time = current_time
            elif not self.session_active:
                self.session_start_time = None
                
        except Exception as e:
            print(f"❌ 세션 상태 확인 오류: {e}")
    
    def get_current_liquidation_metrics(self) -> Dict[str, Any]:
        """현재 청산 지표 계산"""
        try:
            current_time = datetime.now()
            current_timestamp = int(current_time.timestamp())
            
            # 30초 윈도우 계산
            window_start = current_timestamp - self.config.agg_window_sec
            
            # 롱 청산 30초 합계
            l_long_30s = sum(val for ts, val in self.long_bins if ts >= window_start)
            
            # 숏 청산 30초 합계
            l_short_30s = sum(val for ts, val in self.short_bins if ts >= window_start)
            
            # Z점수 계산
            z_long = (l_long_30s - self.mu_long) / max(self.sigma_long, 1e-9)
            z_short = (l_short_30s - self.mu_short) / max(self.sigma_short, 1e-9)
            
            # LPI 계산
            total_liquidation = l_long_30s + l_short_30s
            lpi = (l_short_30s - l_long_30s) / (total_liquidation + 1e-9)
            
            # 캐스케이드 감지
            is_cascade = self._detect_cascade(current_timestamp)
            
            # 쿨다운 상태 확인
            cooldown_active = self._is_cooldown_active(current_time)
            
            return {
                'timestamp': current_time,
                'l_long_30s': l_long_30s,
                'l_short_30s': l_short_30s,
                'z_long': z_long,
                'z_short': z_short,
                'lpi': lpi,
                'is_cascade': is_cascade,
                'cooldown_active': cooldown_active,
                'session_active': self.session_active,
                'background_stats': {
                    'mu_long': self.mu_long,
                    'sigma_long': self.sigma_long,
                    'mu_short': self.mu_short,
                    'sigma_short': self.sigma_short
                }
            }
            
        except Exception as e:
            print(f"❌ 청산 지표 계산 오류: {e}")
            return {}
    
    def _detect_cascade(self, current_timestamp: int) -> bool:
        """캐스케이드 조건 감지"""
        try:
            cascade_start = current_timestamp - self.config.cascade_seconds
            cascade_count = 0
            
            # 롱 청산 캐스케이드 확인
            for ts, val in self.long_bins:
                if ts >= cascade_start:
                    z_score = (val - self.mu_long) / max(self.sigma_long, 1e-9)
                    if z_score >= self.config.cascade_z:
                        cascade_count += 1
            
            # 숏 청산 캐스케이드 확인
            for ts, val in self.short_bins:
                if ts >= cascade_start:
                    z_score = (val - self.mu_short) / max(self.sigma_short, 1e-9)
                    if z_score >= self.config.cascade_z:
                        cascade_count += 1
            
            # 캐스케이드 상태 업데이트
            if cascade_count >= self.config.cascade_count:
                if not self.cascade_detected:
                    self.cascade_detected = True
                    self.cascade_start_time = datetime.now()
                return True
            else:
                # 캐스케이드 종료 확인 (30초 후)
                if (self.cascade_detected and self.cascade_start_time and 
                    (datetime.now() - self.cascade_start_time).total_seconds() > 30):
                    self.cascade_detected = False
                    self.cascade_start_time = None
                return False
                
        except Exception as e:
            print(f"❌ 캐스케이드 감지 오류: {e}")
            return False
    
    def _is_cooldown_active(self, current_time: datetime) -> bool:
        """쿨다운 상태 확인"""
        if not self.last_strong_spike_time:
            return False
        
        time_since_spike = (current_time - self.last_strong_spike_time).total_seconds()
        return time_since_spike < self.config.cooldown_after_strong_sec
    
    def analyze_strategy_a_sweep_reclaim(self, 
                                       metrics: Dict[str, Any],
                                       price_data: pd.DataFrame,
                                       key_levels: Dict[str, float],
                                       atr: float) -> Optional[Dict]:
        """전략 A: 스윕&리클레임 분석 (롱/숏)"""
        try:
            if not self.session_active or metrics.get('cooldown_active', False):
                return None
            
            # 캐스케이드 중이면 페이드 금지
            if metrics.get('is_cascade', False):
                return None
            
            current_price = price_data['close'].iloc[-1]
            prev_day_low = key_levels.get('prev_day_low', 0)
            prev_day_high = key_levels.get('prev_day_high', 0)
            
            # === 롱 신호 분석 ===
            if prev_day_low > 0:
                # 롱 진입 조건 확인
                z_long = metrics.get('z_long', 0)
                lpi = metrics.get('lpi', 0)
                
                # 가격이 prev_day_low 하회하고 롱 청산 스파이크
                price_below_level = current_price < prev_day_low
                long_liquidation_spike = (z_long >= self.config.z_spike and 
                                        lpi <= -self.config.lpi_bias)
                
                if price_below_level and long_liquidation_spike:
                    # 레벨 재진입 확인 (1분 캔들)
                    if len(price_data) >= 2:
                        # 이전 봉이 레벨 아래, 현재 봉이 레벨 위
                        prev_close = price_data['close'].iloc[-2]
                        current_close = price_data['close'].iloc[-1]
                        
                        level_reentry = (prev_close < prev_day_low and 
                                       current_close > prev_day_low)
                        
                        if level_reentry:
                            # 롱 신호 생성
                            entry_price = current_close
                            stop_loss = min(prev_day_low, current_price) - atr * 0.3  # 0.3×ATR
                            
                            # 백업 스탑로스 (0.08%)
                            backup_stop = current_price * 0.9992
                            stop_loss = max(stop_loss, backup_stop)
                            
                            risk = entry_price - stop_loss
                            tp1 = entry_price + risk * self.config.tp1_R
                            
                            # 2차 목표
                            if "VWAP" in self.config.tp2:
                                tp2 = key_levels.get('vwap', entry_price + risk * 2.0)
                            else:
                                tp2 = entry_price + risk * 2.0
                            
                            return {
                                'signal_type': 'SWEEP_RECLAIM_LONG',
                                'action': 'BUY',
                                'confidence': 0.80,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'take_profit1': tp1,
                                'take_profit2': tp2,
                                'risk_reward': self.config.tp1_R,
                                'timestamp': datetime.now(),
                                'reason': f"하단 스윕 + 롱청산스파이크 + 리클레임 | Z:{z_long:.1f}, LPI:{lpi:.2f}",
                                'playbook': 'A',
                                'liquidation_metrics': metrics
                            }
            
            # === 숏 신호 분석 ===
            if prev_day_high > 0:
                # 숏 진입 조건 확인
                z_short = metrics.get('z_short', 0)
                lpi = metrics.get('lpi', 0)
                
                # 가격이 prev_day_high 상회하고 숏 청산 스파이크
                price_above_level = current_price > prev_day_high
                short_liquidation_spike = (z_short >= self.config.z_spike and 
                                         lpi >= self.config.lpi_bias)
                
                if price_above_level and short_liquidation_spike:
                    # 레벨 재진입 확인 (1분 캔들)
                    if len(price_data) >= 2:
                        # 이전 봉이 레벨 위, 현재 봉이 레벨 아래
                        prev_close = price_data['close'].iloc[-2]
                        current_close = price_data['close'].iloc[-1]
                        
                        level_reentry = (prev_close > prev_day_high and 
                                       current_close < prev_day_high)
                        
                        if level_reentry:
                            # 숏 신호 생성
                            entry_price = current_close
                            stop_loss = max(prev_day_high, current_price) + atr * 0.3  # 0.3×ATR
                            
                            # 백업 스탑로스 (0.08%)
                            backup_stop = current_price * 1.0008
                            stop_loss = min(stop_loss, backup_stop)
                            
                            risk = stop_loss - entry_price
                            tp1 = entry_price - risk * self.config.tp1_R
                            
                            # 2차 목표
                            if "VWAP" in self.config.tp2:
                                tp2 = key_levels.get('vwap', entry_price - risk * 2.0)
                            else:
                                tp2 = entry_price - risk * 2.0
                            
                            return {
                                'signal_type': 'SWEEP_RECLAIM_SHORT',
                                'action': 'SELL',
                                'confidence': 0.80,
                                'entry_price': entry_price,
                                'stop_loss': stop_loss,
                                'take_profit1': tp1,
                                'take_profit2': tp2,
                                'risk_reward': self.config.tp1_R,
                                'timestamp': datetime.now(),
                                'reason': f"상단 스윕 + 숏청산스파이크 + 리클레임 | Z:{z_short:.1f}, LPI:{lpi:.2f}",
                                'playbook': 'A',
                                'liquidation_metrics': metrics
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
        """전략 B: 스퀴즈 추세지속 분석 (롱/숏)"""
        try:
            if not self.session_active or metrics.get('cooldown_active', False):
                return None
            
            # 오프닝 레인지 필요
            if not opening_range:
                return None
            
            current_price = price_data['close'].iloc[-1]
            or_high = opening_range.get('high', 0)
            or_low = opening_range.get('low', 0)
            
            # === 롱 신호 분석 ===
            if or_high > 0:
                # 숏 청산 스파이크 확인
                z_short = metrics.get('z_short', 0)
                lpi = metrics.get('lpi', 0)
                
                short_squeeze_spike = (z_short >= self.config.z_spike and 
                                     lpi >= self.config.lpi_bias)
                
                if short_squeeze_spike:
                    # 돌파 확인 (OR 상단 돌파)
                    breakout = current_price > or_high
                    
                    if breakout:
                        # 리테스트 확인 (15-60초 내 되돌림)
                        if len(price_data) >= 4:  # 최소 4봉 필요
                            # 최근 4봉에서 리테스트 패턴 찾기
                            retest_found = False
                            retest_low = current_price
                            
                            for i in range(1, min(5, len(price_data))):
                                low_price = price_data['low'].iloc[-i]
                                if low_price < or_high and low_price >= or_high - atr * self.config.retest_atr_tol:
                                    retest_found = True
                                    retest_low = min(retest_low, low_price)
                                    break
                            
                            if retest_found:
                                # 추가 숏 청산 확인 (10초 누적)
                                additional_short_liquidation = self._check_additional_short_liquidation()
                                
                                # 롱 신호 생성
                                entry_price = current_price
                                stop_loss = retest_low - atr * 0.5  # 0.5×ATR
                                
                                risk = entry_price - stop_loss
                                tp1 = entry_price + risk * self.config.tp1_R
                                
                                # 2차 목표 (OR 확장)
                                if self.config.or_extension:
                                    or_range = or_high - or_low
                                    tp2 = or_high + or_range
                                else:
                                    tp2 = entry_price + risk * 2.5
                                
                                return {
                                    'signal_type': 'SQUEEZE_TREND_CONTINUATION_LONG',
                                    'action': 'BUY',
                                    'confidence': 0.85,
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'take_profit1': tp1,
                                    'take_profit2': tp2,
                                    'risk_reward': self.config.tp1_R,
                                    'timestamp': datetime.now(),
                                    'reason': f"상단 돌파 + 숏청산스파이크 + 리테스트 롱 | Z:{z_short:.1f}, LPI:{lpi:.2f}",
                                    'playbook': 'B',
                                    'liquidation_metrics': metrics
                                }
            
            # === 숏 신호 분석 ===
            if or_low > 0:
                # 롱 청산 스파이크 확인
                z_long = metrics.get('z_long', 0)
                lpi = metrics.get('lpi', 0)
                
                long_squeeze_spike = (z_long >= self.config.z_spike and 
                                    lpi <= -self.config.lpi_bias)
                
                if long_squeeze_spike:
                    # 이탈 확인 (OR 하단 이탈)
                    breakdown = current_price < or_low
                    
                    if breakdown:
                        # 리테스트 확인 (15-60초 내 되돌림)
                        if len(price_data) >= 4:  # 최소 4봉 필요
                            # 최근 4봉에서 리테스트 패턴 찾기
                            retest_found = False
                            retest_high = current_price
                            
                            for i in range(1, min(5, len(price_data))):
                                high_price = price_data['high'].iloc[-i]
                                if high_price > or_low and high_price <= or_low + atr * self.config.retest_atr_tol:
                                    retest_found = True
                                    retest_high = max(retest_high, high_price)
                                    break
                            
                            if retest_found:
                                # 추가 롱 청산 확인 (10초 누적)
                                additional_long_liquidation = self._check_additional_long_liquidation()
                                
                                # 숏 신호 생성
                                entry_price = current_price
                                stop_loss = retest_high + atr * 0.5  # 0.5×ATR
                                
                                risk = stop_loss - entry_price
                                tp1 = entry_price - risk * self.config.tp1_R
                                
                                # 2차 목표 (OR 확장 아래)
                                if self.config.or_extension:
                                    or_range = or_high - or_low
                                    tp2 = or_low - or_range
                                else:
                                    tp2 = entry_price - risk * 2.5
                                
                                return {
                                    'signal_type': 'SQUEEZE_TREND_CONTINUATION_SHORT',
                                    'action': 'SELL',
                                    'confidence': 0.85,
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'take_profit1': tp1,
                                    'take_profit2': tp2,
                                    'risk_reward': self.config.tp1_R,
                                    'timestamp': datetime.now(),
                                    'reason': f"하단 이탈 + 롱청산스파이크 + 리테스트 숏 | Z:{z_long:.1f}, LPI:{lpi:.2f}",
                                    'playbook': 'B',
                                    'liquidation_metrics': metrics
                                }
            
            return None
            
        except Exception as e:
            print(f"❌ 스퀴즈 추세지속 분석 오류: {e}")
            return None
    
    def analyze_strategy_c_overheat_extinction_fade(self,
                                                  metrics: Dict[str, Any],
                                                  price_data: pd.DataFrame,
                                                  vwap: float,
                                                  vwap_std: float) -> Optional[Dict]:
        """전략 C: 과열-소멸 페이드 분석"""
        try:
            if not self.session_active or metrics.get('cooldown_active', False):
                return None
            
            # 캐스케이드 중이면 페이드 금지
            if metrics.get('is_cascade', False):
                return None
            
            current_price = price_data['close'].iloc[-1]
            
            # VWAP ±2σ 바깥 확인
            vwap_lower = vwap - self.config.vwap_sd_enter * vwap_std
            vwap_upper = vwap + self.config.vwap_sd_enter * vwap_std
            
            price_outside_vwap = current_price < vwap_lower or current_price > vwap_upper
            
            if not price_outside_vwap:
                return None
            
            # 롱 페이드 (하락 과열)
            if current_price < vwap_lower:
                return self._analyze_long_fade(metrics, price_data, vwap, vwap_std)
            
            # 숏 페이드 (상승 과열)
            elif current_price > vwap_upper:
                return self._analyze_short_fade(metrics, price_data, vwap, vwap_std)
            
            return None
            
        except Exception as e:
            print(f"❌ 과열-소멸 페이드 분석 오류: {e}")
            return None
    
    def _analyze_long_fade(self, 
                          metrics: Dict[str, Any],
                          price_data: pd.DataFrame,
                          vwap: float,
                          vwap_std: float) -> Optional[Dict]:
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
                'risk_reward': 1.2,
                'timestamp': datetime.now(),
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
                           vwap_std: float) -> Optional[Dict]:
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
                'risk_reward': 1.2,
                'timestamp': datetime.now(),
                'reason': f"VWAP +2σ + 숏청산스파이크 + 감소 페이드 숏 | Z:{z_short:.1f}",
                'playbook': 'C',
                'liquidation_metrics': metrics
            }
            
        except Exception as e:
            print(f"❌ 숏 페이드 분석 오류: {e}")
            return None
    
    def _check_post_spike_decay(self, metrics: Dict[str, Any], side: str) -> bool:
        """스파이크 후 감소 확인"""
        try:
            # 10초 평균 청산 계산
            current_time = datetime.now()
            window_start = int(current_time.timestamp()) - 10
            
            if side == 'long':
                current_10s = sum(val for ts, val in self.long_bins if ts >= window_start)
                mu_10s = self.mu_long
            else:  # short
                current_10s = sum(val for ts, val in self.short_bins if ts >= window_start)
                mu_10s = self.mu_short
            
            # 스파이크 후 감소 확인
            decay_ratio = current_10s / (mu_10s + 1e-9)
            return decay_ratio < self.config.post_spike_decay_ratio
            
        except Exception as e:
            print(f"❌ 스파이크 후 감소 확인 오류: {e}")
            return False
    
    def _check_additional_long_liquidation(self) -> bool:
        """추가 롱 청산 확인 (10초 누적)"""
        try:
            current_time = datetime.now()
            window_start = int(current_time.timestamp()) - 10
            
            # 10초 누적 롱 청산
            long_10s = sum(val for ts, val in self.long_bins if ts >= window_start)
            
            # 기본선 + 2σ 확인
            threshold = self.mu_long + 2 * self.sigma_long
            
            return long_10s >= threshold
            
        except Exception as e:
            print(f"❌ 추가 롱 청산 확인 오류: {e}")
            return False
    
    def _check_additional_short_liquidation(self) -> bool:
        """추가 숏 청산 확인 (10초 누적)"""
        try:
            current_time = datetime.now()
            window_start = int(current_time.timestamp()) - 10
            
            # 10초 누적 숏 청산
            short_10s = sum(val for ts, val in self.short_bins if ts >= window_start)
            
            # 기본선 + 2σ 확인
            threshold = self.mu_short + 2 * self.sigma_short
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
        """모든 전략 분석"""
        try:
            # 현재 청산 지표 가져오기
            metrics = self.get_current_liquidation_metrics()
            if not metrics:
                return None
            
            # 강한 스파이크 감지 시 쿨다운 시작
            z_long = metrics.get('z_long', 0)
            z_short = metrics.get('z_short', 0)
            
            if max(z_long, z_short) >= self.config.z_strong:
                self.last_strong_spike_time = datetime.now()
            
            # 전략 A: 스윕&리클레임
            signal_a = self.analyze_strategy_a_sweep_reclaim(
                metrics, price_data, key_levels, atr
            )
            if signal_a:
                return signal_a
            
            # 전략 B: 스퀴즈 추세지속
            signal_b = self.analyze_strategy_b_squeeze_trend_continuation(
                metrics, price_data, opening_range, atr
            )
            if signal_b:
                return signal_b
            
            # 전략 C: 과열-소멸 페이드
            signal_c = self.analyze_strategy_c_overheat_extinction_fade(
                metrics, price_data, vwap, vwap_std
            )
            if signal_c:
                return signal_c
            
            return None
            
        except Exception as e:
            print(f"❌ 전체 전략 분석 오류: {e}")
            return None
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """전략 요약 정보"""
        return {
            'session_active': self.session_active,
            'cascade_detected': self.cascade_detected,
            'cooldown_active': self._is_cooldown_active(datetime.now()),
            'background_stats': {
                'mu_long': self.mu_long,
                'sigma_long': self.sigma_long,
                'mu_short': self.mu_short,
                'sigma_short': self.sigma_short
            },
            'data_points': {
                'long_bins': len(self.long_bins),
                'short_bins': len(self.short_bins),
                'total_bins': len(self.liquidation_bins)
            }
        }


def make_advanced_liquidation_plan(df: pd.DataFrame,
                                  liquidation_events: List[Dict],
                                  config: AdvancedLiquidationConfig,
                                  key_levels: Dict[str, float],
                                  opening_range: Dict[str, float],
                                  vwap: float,
                                  vwap_std: float) -> Optional[Dict]:
    """고급 청산 거래 계획 생성"""
    try:
        strategy = AdvancedLiquidationStrategy(config)
        
        # 청산 이벤트 처리
        for event in liquidation_events:
            strategy.process_liquidation_event(event)
        
        # ATR 계산
        atr = calculate_atr(df, config.atr_len)
        if pd.isna(atr):
            atr = df['close'].iloc[-1] * 0.02  # 기본값
        
        # 모든 전략 분석
        signal = strategy.analyze_all_strategies(
            df, key_levels, opening_range, vwap, vwap_std, atr
        )
        
        if signal:
            # 포지션 사이징 계산
            risk_percent = config.risk_pct
            equity = 10000  # 예시 자본금
            risk_dollar = equity * risk_percent / 100
            
            stop_distance = abs(signal['entry_price'] - signal['stop_loss'])
            position_size = risk_dollar / stop_distance if stop_distance > 0 else 0
            
            signal['position_size'] = position_size
            signal['risk_dollar'] = risk_dollar
            
            return signal
        
        return None
        
    except Exception as e:
        print(f"❌ 고급 청산 거래 계획 생성 오류: {e}")
        return None
