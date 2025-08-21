#!/usr/bin/env python3
"""
통합 스마트 자동 트레이더 (리팩토링 버전)
실시간 청산 전략 + 세션 기반 전략 + 고급 청산 전략을 활용합니다.
"""

import time
import datetime
import threading
from typing import Dict, Any, Optional, List, Tuple
from core.trader_core import TraderCore
from analyzers.liquidation_analyzer import LiquidationAnalyzer
from analyzers.technical_analyzer import TechnicalAnalyzer
from handlers.websocket_handler import WebSocketHandler
from handlers.display_handler import DisplayHandler
from utils.trader_utils import get_next_5min_candle_time, format_time_delta
from config.integrated_config import IntegratedConfig
import pandas as pd
import numpy as np


class IntegratedSmartTrader:
    """통합 스마트 자동 트레이더 (리팩토링 버전)"""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self.running = False
        
        # 핵심 컴포넌트 초기화
        self.core = TraderCore(config)
        
        # 분석 엔진 초기화
        self.liquidation_analyzer = LiquidationAnalyzer(self.core.get_websocket())
        self.technical_analyzer = TechnicalAnalyzer(config)
        
        # 핸들러 초기화
        self.websocket_handler = WebSocketHandler(self.core.get_websocket())
        self.display_handler = DisplayHandler(self.core.get_websocket())
        
        # 상태 관리
        self.running = False
        self.last_analysis_time = None
        self.last_liquidation_analysis = None
        
        # 상태 및 통계 초기화
        self._init_state_and_stats()
        
        # 콜백 설정
        self._setup_callbacks()
    
    def _init_state_and_stats(self):
        """상태 및 통계 초기화"""
        # 거래량 급증 집계
        self.volume_spike_buffer = []
        self.last_volume_summary = None
        self.volume_summary_cooldown = 30
    
    def _setup_callbacks(self):
        """웹소켓 콜백 설정"""
        callbacks = {
            'liquidation': lambda data: self._handle_liquidation_event(data),
            'volume': lambda data: self._handle_volume_spike(data),
            'price': lambda data: self.websocket_handler.on_price_update(
                data, 
                self._analyze_realtime_liquidation  # 청산 분석만 실행
            ),
            'kline': lambda data: self.websocket_handler.on_kline(
                data, 
                self._analyze_realtime_liquidation  # 청산 분석만 실행
            )
        }
        self.websocket_handler.setup_callbacks(callbacks)
        
        # 실제 바이낸스 청산 스트림 연결 활성화
        self._enable_real_liquidation_stream()
        
    def _enable_real_liquidation_stream(self):
        """실제 바이낸스 청산 스트림 연결 활성화"""
        try:
            websocket = self.core.get_websocket()
            if websocket:
                # 청산 스트림 시작
                websocket.start_liquidation_stream()
                print(f"✅ 바이낸스 청산 스트림 연결됨: {self.config.symbol}")
            else:
                print(f"❌ 웹소켓 연결 실패")
        except Exception as e:
            print(f"❌ 청산 스트림 연결 오류: {e}")
    
    def _handle_liquidation_event(self, data: Dict):
        """청산 이벤트 처리 및 AdvancedLiquidationStrategy에 전달"""
        try:
            # 기본 청산 분석 실행
            self._analyze_realtime_liquidation(data)
            
            # AdvancedLiquidationStrategy에 청산 이벤트 전달
            if not hasattr(self, '_adv_liquidation_strategy'):
                # 새로 생성
                from signals.advanced_liquidation_strategy import AdvancedLiquidationStrategy, AdvancedLiquidationConfig
                adv_config = AdvancedLiquidationConfig()
                self._adv_liquidation_strategy = AdvancedLiquidationStrategy(adv_config)
            
            strategy = self._adv_liquidation_strategy
            
            # 바이낸스 청산 데이터 형식에 맞게 처리
            if 'side' in data and 'qty_usd' in data:
                # 바이낸스 청산 데이터 형식: BUY=숏청산, SELL=롱청산
                # BUY: 숏 포지션이 강제 청산됨 (숏 청산)
                # SELL: 롱 포지션이 강제 청산됨 (롱 청산)
                side = 'short' if data['side'] == 'BUY' else 'long'
                
                # 청산 이벤트를 딕셔너리로 구성
                liquidation_event = {
                    'ts': int(data.get('timestamp', datetime.datetime.now(datetime.timezone.utc)).timestamp()),
                    'side': side,
                    'qty_usd': data['qty_usd']
                }
                
                strategy.process_liquidation_event(liquidation_event)
                
                # 실시간 청산 정보 출력 (더 명확하게)
                if data['side'] == 'BUY':
                    print(f"🔥 실시간 청산: SHORT ${data['qty_usd']:,.0f} @ ${data.get('price', 0):.2f} (숏 포지션 강제 청산)")
                else:
                    print(f"🔥 실시간 청산: LONG ${data['qty_usd']:,.0f} @ ${data.get('price', 0):.2f} (롱 포지션 강제 청산)")
                
        except Exception as e:
            print(f"❌ 청산 이벤트 처리 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_volume_spike(self, volume_data: Dict):
        """거래량 급증 처리"""
        self.last_volume_summary = self.websocket_handler.on_volume_spike(
            volume_data, 
            self.volume_spike_buffer, 
            self.last_volume_summary,
            self.volume_summary_cooldown,
            self.display_handler.print_volume_spike_summary,
            self._analyze_realtime_liquidation
        )
    
    def _analyze_realtime_technical(self):
        """실시간 기술적 분석"""
        try:
            # 세션 기반 전략과 고급 청산 전략만 실행
            websocket = self.core.get_websocket()
            
            # 세션 기반 전략 분석
            session_signal = self._analyze_session_strategy(websocket)
            if session_signal:
                # 세션 전략 신호 직접 처리
                self._process_integrated_signal({
                    'session_signal': session_signal
                })
            
            # 고급 청산 전략 분석
            advanced_liquidation_signal = self._analyze_advanced_liquidation_strategy(websocket)
            if advanced_liquidation_signal:
                # 고급 청산 전략 신호 직접 처리
                self._process_integrated_signal({
                    'advanced_liquidation_signal': advanced_liquidation_signal
                })
                
        except Exception as e:
            print(f"❌ 실시간 기술적 분석 오류: {e}")
    
    def _analyze_realtime_liquidation(self, data=None):
        """실시간 통합 청산 신호 분석 (ENHANCED_LIQUIDATION + Prediction 통합)"""
        try:
            # 현재 가격 가져오기
            websocket = self.core.get_websocket()
            if not websocket.price_history:
                return
            
            current_price = websocket.price_history[-1]['price']
            
            # 청산 통계 분석
            liquidation_stats = websocket.get_liquidation_stats(self.config.liquidation_window_minutes)
            volume_analysis = websocket.get_volume_analysis(3)
            
            # 통합 청산 신호 분석 (ENHANCED_LIQUIDATION + Prediction)
            integrated_liquidation_signal = self._analyze_integrated_liquidation(
                liquidation_stats, volume_analysis, current_price, websocket
            )
            
            # 청산 신호만 처리 (세션 전략은 정각 1분마다 별도 실행)
            if integrated_liquidation_signal:
                self._process_integrated_signal({
                    'liquidation_signal': integrated_liquidation_signal
                })
            
        except Exception as e:
            print(f"❌ 실시간 청산 분석 오류: {e}")
    
    def _analyze_session_strategy(self, websocket) -> Optional[Dict]:
        """세션 기반 전략 분석"""
        try:
            if not self.config.enable_session_strategy:
                return None
            
            # 1분봉 데이터 로드
            df_1m = self.core.get_data_loader().load_klines(
                self.config.symbol, 
                self.config.session_timeframe, 
                1500  # 현재 시간까지 커버하기 위해 더 증가
            )
            
            if df_1m.empty:
                return None
            
            # 키 레벨 계산 (전일 H/L, 스윙 레벨 등)
            key_levels = self._calculate_session_key_levels(df_1m)
            
            # 현재 시간 (UTC 명시)
            current_time = datetime.datetime.now(datetime.timezone.utc)
            
            # 세션 전략 분석 (직접 SessionBasedStrategy 사용)
            from signals.session_based_strategy import SessionBasedStrategy, SessionConfig
            session_config = SessionConfig()  # 기본 설정으로 생성
            session_strategy = SessionBasedStrategy(session_config)
            
            session_signal = session_strategy.analyze_session_strategy(
                df_1m, key_levels, current_time
            )
            
            return session_signal
            
        except Exception as e:
            print(f"❌ 세션 전략 분석 오류: {e}")
            return None
    
    def _calculate_session_key_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """세션 전략용 키 레벨 계산"""
        try:
            if df.empty:
                return {}
            
            # 전일 고가/저가/종가
            daily_data = df.resample('D').agg({
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            if len(daily_data) < 2:
                return {}
            
            prev_day = daily_data.iloc[-2]
            
            # 최근 스윙 고점/저점 (20봉 기준)
            lookback = min(20, len(df))
            recent_data = df.tail(lookback)
            
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            
            return {
                'prev_day_high': prev_day['high'],
                'prev_day_low': prev_day['low'],
                'prev_day_close': prev_day['close'],
                'last_swing_high': swing_high,
                'last_swing_low': swing_low
            }
            
        except Exception as e:
            print(f"❌ 세션 키 레벨 계산 오류: {e}")
            return {}
    
    def _analyze_advanced_liquidation_strategy(self, websocket) -> Optional[Dict]:
        """고급 청산 전략 분석"""
        try:
            if not self.config.enable_advanced_liquidation:
                return None
            
            # 1분봉 데이터 로드
            df_1m = self.core.get_data_loader().load_klines(
                self.config.symbol, 
                "1m", 
                500  # 충분한 데이터
            )
            
            if df_1m.empty:
                return None
            
            # 키 레벨 계산
            key_levels = self._calculate_session_key_levels(df_1m)
            
            # 오프닝 레인지 계산
            opening_range = self._calculate_opening_range(df_1m)
            
            # VWAP 및 표준편차 계산
            vwap, vwap_std = self._calculate_vwap_and_std(df_1m)
            
            # ATR 계산
            from indicators.atr import calculate_atr
            atr = calculate_atr(df_1m, 14)
            if pd.isna(atr):
                atr = df_1m['close'].iloc[-1] * 0.02  # 기본값
            
            # 기존에 생성된 AdvancedLiquidationStrategy 인스턴스 사용
            if hasattr(self, '_adv_liquidation_strategy'):
                adv_strategy = self._adv_liquidation_strategy
            else:
                # 새로 생성
                from signals.advanced_liquidation_strategy import AdvancedLiquidationStrategy, AdvancedLiquidationConfig
                adv_config = AdvancedLiquidationConfig()
                self._adv_liquidation_strategy = AdvancedLiquidationStrategy(adv_config)
                adv_strategy = self._adv_liquidation_strategy
            
            # 워밍업 상태 및 청산 데이터 상태 확인
            warmup_status = adv_strategy.get_warmup_status()
            print(f"   🔥 워밍업 상태: SETUP={warmup_status['can_setup']}, ENTRY={warmup_status['can_entry']}")
            print(f"   📊 청산 샘플: 롱={warmup_status['long_samples']}, 숏={warmup_status['short_samples']}")
            
            # 현재 청산 메트릭 확인
            try:
                metrics = adv_strategy.get_current_liquidation_metrics()
                if metrics:
                    print(f"   📈 청산 지표: 롱 Z={metrics['z_long']:.2f}, 숏 Z={metrics['z_short']:.2f}, LPI={metrics['lpi']:.3f}")
                    
                    # 청산 데이터 방향성 확인
                    if warmup_status['long_samples'] > 0 or warmup_status['short_samples'] > 0:
                        print(f"   📊 청산 데이터 방향성:")
                        print(f"      - 롱 샘플: {warmup_status['long_samples']}개 (롱 포지션 강제 청산)")
                        print(f"      - 숏 샘플: {warmup_status['short_samples']}개 (숏 포지션 강제 청산)")
                        
                        # 최근 청산 이벤트 확인
                        if hasattr(adv_strategy, 'long_bins') and adv_strategy.long_bins:
                            recent_long = list(adv_strategy.long_bins)[-1] if adv_strategy.long_bins else None
                            if recent_long:
                                print(f"      - 최근 롱 청산: ${recent_long[1]:,.0f}")
                        
                        if hasattr(adv_strategy, 'short_bins') and adv_strategy.short_bins:
                            recent_short = list(adv_strategy.short_bins)[-1] if adv_strategy.short_bins else None
                            if recent_short:
                                print(f"      - 최근 숏 청산: ${recent_short[1]:,.0f}")
            except Exception as e:
                print(f"   ❌ 청산 메트릭 확인 실패: {e}")
            
            # 현재 가격
            current_price = df_1m['close'].iloc[-1]
            
            # 고급 청산 전략 분석 실행
            advanced_signal = adv_strategy.analyze_all_strategies(
                df_1m, key_levels, opening_range, vwap, vwap_std, atr
            )
            
            # 디버깅 정보 추가
            if advanced_signal:
                print(f"   📊 분석 완료: {advanced_signal.get('action', 'UNKNOWN')} | {advanced_signal.get('tier', 'UNKNOWN')} | 점수: {advanced_signal.get('total_score', 0.00):.3f}")
            else:
                print(f"   📊 분석 완료: 신호 없음")
                
            # 전략별 분석 결과 디버깅
            print(f"   🔍 전략별 분석 디버깅:")
            try:
                # 전략 A: 스윕&리클레임
                signal_a = adv_strategy.analyze_strategy_a_sweep_reclaim(
                    adv_strategy.get_current_liquidation_metrics(), df_1m, key_levels, atr
                )
                print(f"      - 전략 A: {'신호 있음' if signal_a else '신호 없음'}")
                if signal_a:
                    print(f"        액션: {signal_a.get('action')}, 점수: {signal_a.get('total_score', 0):.3f}")
                
                # 전략 B: 스퀴즈 추세지속
                signal_b = adv_strategy.analyze_strategy_b_squeeze_trend_continuation(
                    adv_strategy.get_current_liquidation_metrics(), df_1m, opening_range, atr
                )
                print(f"      - 전략 B: {'신호 있음' if signal_b else '신호 없음'}")
                if signal_b:
                    print(f"        액션: {signal_b.get('action')}, 점수: {signal_b.get('total_score', 0):.3f}")
                
                # 전략 C: 과열-소멸 페이드
                signal_c = adv_strategy.analyze_strategy_c_overheat_extinction_fade(
                    adv_strategy.get_current_liquidation_metrics(), df_1m, vwap, vwap_std, atr
                )
                print(f"      - 전략 C: {'신호 있음' if signal_c else '신호 없음'}")
                if signal_c:
                    print(f"        액션: {signal_c.get('action')}, 점수: {signal_c.get('total_score', 0):.3f}")
                    
            except Exception as e:
                print(f"      ❌ 전략별 분석 디버깅 실패: {e}")
            
            return advanced_signal
            
        except Exception as e:
            print(f"❌ 고급 청산 전략 분석 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_opening_range(self, df: pd.DataFrame) -> Dict[str, float]:
        """오프닝 레인지 계산"""
        try:
            if df.empty:
                return {}
            
            # 첫 15분 데이터
            or_minutes = 15
            if len(df) < or_minutes:
                return {}
            
            or_data = df.head(or_minutes)
            
            return {
                'high': or_data['high'].max(),
                'low': or_data['low'].min(),
                'center': (or_data['high'].max() + or_data['low'].min()) / 2,
                'range': or_data['high'].max() - or_data['low'].min()
            }
            
        except Exception as e:
            print(f"❌ 오프닝 레인지 계산 오류: {e}")
            return {}
    
    def _calculate_vwap_and_std(self, df: pd.DataFrame) -> Tuple[float, float]:
        """VWAP 및 표준편차 계산"""
        try:
            if df.empty:
                return 0.0, 0.0
            
            # 가격과 거래량으로 VWAP 계산
            vwap = np.average(df['close'], weights=df['volume'])
            
            # 표준편차 계산
            std = np.std(df['close'])
            
            return vwap, std
            
        except Exception as e:
            print(f"❌ VWAP 및 표준편차 계산 오류: {e}")
            return 0.0, 0.0
    

    
    def _analyze_integrated_liquidation(self, liquidation_stats: Dict, volume_analysis: Dict, current_price: float, websocket) -> Optional[Dict]:
        """통합 청산 신호 분석 (ENHANCED_LIQUIDATION + Prediction)"""
        try:
            # 기본 청산 신호 분석
            basic_signal = self.liquidation_analyzer.analyze_liquidation_signal(
                liquidation_stats, volume_analysis, current_price
            )
            
            # 청산 예측 분석
            recent_liquidations = websocket.get_recent_liquidations(self.config.liquidation_window_minutes)
            prediction_signal = self.core.get_integrated_strategy().analyze_liquidation_prediction(
                recent_liquidations, current_price
            )
            
            # 두 신호를 통합하여 최종 신호 생성
            if basic_signal and prediction_signal:
                # 둘 다 신호가 있는 경우 - 통합 효과
                return self._create_liquidation_integrated_signal(basic_signal, prediction_signal, current_price)
            elif basic_signal:
                # 기본 청산 신호만 있는 경우
                return basic_signal
            elif prediction_signal:
                # 예측 신호만 있는 경우 - 예측 신호를 기본 형태로 변환
                return self._convert_prediction_to_liquidation_signal(prediction_signal, current_price)
            else:
                return None
                
        except Exception as e:
            print(f"❌ 통합 청산 신호 분석 오류: {e}")
            return None
    
    def _create_liquidation_integrated_signal(self, basic_signal: Dict, prediction_signal: Dict, current_price: float) -> Dict:
        """청산 통합 신호 생성"""
        try:
            # 기본 신호 정보
            action = basic_signal.get('action', 'NEUTRAL')
            confidence = basic_signal.get('confidence', 0)
            
            # 예측 신호 정보
            pred_type = prediction_signal.get('type', 'UNKNOWN')
            pred_confidence = prediction_signal.get('confidence', 0)
            target_price = prediction_signal.get('target_price', current_price)
            
            # 통합 신뢰도 계산 (기본 + 예측)
            integrated_confidence = min(0.95, (confidence + pred_confidence) / 2 + 0.1)
            
            # 리스크 관리 (기본 신호 기준)
            if action == 'BUY':
                stop_loss = basic_signal.get('stop_loss', current_price * 0.98)
                take_profit1 = basic_signal.get('take_profit1', current_price * 1.04)
                take_profit2 = basic_signal.get('take_profit2', current_price * 1.06)
            elif action == 'SELL':
                stop_loss = basic_signal.get('stop_loss', current_price * 1.02)
                take_profit1 = basic_signal.get('take_profit1', current_price * 0.96)
                take_profit2 = basic_signal.get('take_profit2', current_price * 0.94)
            else:
                return basic_signal
            
            # 리스크/보상 비율 계산
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit1 - current_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            # 통합 이유 생성
            integrated_reason = f"청산 급증 + {pred_type} 예측 일치 | 신뢰도: {confidence:.1%} + {pred_confidence:.1%}"
            
            return {
                'signal_type': 'INTEGRATED_LIQUIDATION',
                'action': action,
                'confidence': integrated_confidence,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit1': take_profit1,
                'take_profit2': take_profit2,
                'risk_reward': risk_reward,
                'liquidation_stats': basic_signal.get('liquidation_stats', {}),
                'volume_analysis': basic_signal.get('volume_analysis', {}),
                'prediction_info': {
                    'type': pred_type,
                    'target_price': target_price,
                    'confidence': pred_confidence
                },
                'timestamp': basic_signal.get('timestamp'),
                'reason': integrated_reason,
                'is_integrated': True
            }
            
        except Exception as e:
            print(f"❌ 청산 통합 신호 생성 오류: {e}")
            return basic_signal
    
    def _convert_prediction_to_liquidation_signal(self, prediction_signal: Dict, current_price: float) -> Dict:
        """예측 신호를 청산 신호 형태로 변환"""
        try:
            pred_type = prediction_signal.get('type', 'UNKNOWN')
            confidence = prediction_signal.get('confidence', 0)
            target_price = prediction_signal.get('target_price', current_price)
            
            # 예측 타입에 따른 액션 결정
            if pred_type == 'EXPLOSION_UP':
                action = 'BUY'
                stop_loss = current_price * 0.98
                take_profit1 = target_price
                take_profit2 = target_price * 1.02
            elif pred_type == 'EXPLOSION_DOWN':
                action = 'SELL'
                stop_loss = current_price * 1.02
                take_profit1 = target_price
                take_profit2 = target_price * 0.98
            else:
                return None
            
            # 리스크/보상 비율 계산
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit1 - current_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            return {
                'signal_type': 'INTEGRATED_LIQUIDATION',
                'action': action,
                'confidence': confidence,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit1': take_profit1,
                'take_profit2': take_profit2,
                'risk_reward': risk_reward,
                'liquidation_stats': {},
                'volume_analysis': {},
                'prediction_info': {
                    'type': pred_type,
                    'target_price': target_price,
                    'confidence': confidence
                },
                'timestamp': prediction_signal.get('timestamp'),
                'reason': f"{pred_type} 예측 기반 {action} 신호 | 목표가: ${target_price:.2f}",
                'is_integrated': False
            }
            
        except Exception as e:
            print(f"❌ 예측 신호 변환 오류: {e}")
            return None
    
    def _run_periodic_analysis(self):
        """주기적 분석 (5분봉 기반)"""
        while self.running:
            try:
                # 5분봉 타이밍 계산
                next_candle = get_next_5min_candle_time()
                now = datetime.datetime.now()
                
                if now >= next_candle:
                    # 1초 후 분석 시작
                    time.sleep(1)
                    
                    print(f"\n⏰ {now.strftime('%H:%M:%S')} - 5분봉 주기적 분석 시작")
                    
                    # 세션 기반 전략과 고급 청산 전략 분석
                    websocket = self.core.get_websocket()
                    
                    session_signal = self._analyze_session_strategy(websocket)
                    advanced_liquidation_signal = self._analyze_advanced_liquidation_strategy(websocket)
                    
                    if session_signal or advanced_liquidation_signal:
                        print(f"🎯 전략 신호 생성됨!")
                        self._process_integrated_signal({
                            'session_signal': session_signal,
                            'advanced_liquidation_signal': advanced_liquidation_signal
                        })
                    else:
                        # 신호가 없어도 분석 상태 출력 (간단하게)
                        current_price = websocket.price_history[-1]['price'] if websocket.price_history else 0
                        print(f"📊 분석 완료 | ${current_price:.2f} | 다음: {(next_candle + datetime.timedelta(minutes=5)).strftime('%H:%M')}")
                    
                    self.last_5min_analysis = now
                    print(f"✅ {now.strftime('%H:%M')} - 5분봉 분석 완료")
                
                    # 다음 5분봉까지 대기 (더 짧은 간격으로 체크)
                    time.sleep(30)  # 30초마다 체크
                else:
                    # 다음 5분봉까지 대기 (더 짧은 간격으로 체크)
                    time.sleep(10)  # 10초마다 체크
                    
            except Exception as e:
                print(f"❌ 주기적 분석 오류: {e}")
                time.sleep(10)
    
    def _process_integrated_signal(self, signal: Dict):
        """개별 전략 신호 처리 - 깔끔하게 정리"""
        try:
            # 세션 신호와 고급 청산 신호 처리
            session_signal = signal.get('session_signal')
            advanced_liquidation_signal = signal.get('advanced_liquidation_signal')
            now = datetime.datetime.now()
            
            # 세션 신호 처리
            if session_signal:
                self._print_session_signal(session_signal, now)
            
            # 고급 청산 신호 처리
            if advanced_liquidation_signal:
                self._print_advanced_liquidation_signal(advanced_liquidation_signal, now)
            
            # 통합 신호가 있는 경우
            if signal.get('signal_type'):
                self._print_integrated_signal(signal, now)
            
        except Exception as e:
            print(f"❌ 신호 처리 오류: {e}")
    
    def _print_session_signal(self, signal: Dict, now: datetime.datetime):
        """세션 신호 출력 - 간단하게"""
        try:
            action = signal.get('action', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            signal_type = signal.get('signal_type', 'N/A')
            
            print(f"📊 세션 전략: {action} | {signal_type} | {confidence:.0%}")
            
        except Exception as e:
            print(f"❌ 세션 신호 출력 오류: {e}")
    
    def _print_advanced_liquidation_signal(self, signal: Dict, now: datetime.datetime):
        """고급 청산 신호 출력 - 간단하게"""
        try:
            action = signal.get('action', 'NEUTRAL')
            playbook = signal.get('playbook', 'N/A')
            tier = signal.get('tier', 'N/A')
            total_score = signal.get('total_score', 0)
            
            print(f"⚡ 고급 청산: {action} | {playbook} | {tier} | {total_score:.2f}")
            
        except Exception as e:
            print(f"❌ 고급 청산 신호 출력 오류: {e}")
    
    def _print_integrated_signal(self, signal: Dict, now: datetime.datetime):
        """통합 신호 출력 - 간단하게"""
        try:
            signal_type = signal.get('signal_type', 'UNKNOWN')
            action = signal.get('action', 'NEUTRAL')
            
            signal_icon = self._get_signal_icon(signal_type)
            signal_name = self._get_signal_name(signal_type)
            
            print(f"{signal_icon} {signal_name}: {action}")
            
        except Exception as e:
            print(f"❌ 통합 신호 출력 오류: {e}")
    
    def _get_signal_icon(self, signal_type: str) -> str:
        """신호 타입별 아이콘 반환"""
        icons = {
            'SESSION': '📊',
            'ADVANCED_LIQUIDATION': '⚡',
            'INTEGRATED_LIQUIDATION': '🎯',
            'INTEGRATED': '🎯',
            'UNKNOWN': '❓'
        }
        return icons.get(signal_type, '❓')
    
    def _get_signal_name(self, signal_type: str) -> str:
        """신호 타입별 이름 반환"""
        names = {
            'SESSION': '세션 전략',
            'ADVANCED_LIQUIDATION': '고급 청산 전략',
            'INTEGRATED_LIQUIDATION': '통합 청산 전략',
            'INTEGRATED': '통합 전략',
            'UNKNOWN': 'UNKNOWN'
        }
        return names.get(signal_type, 'UNKNOWN')
    
    def start(self):
        """트레이더 시작"""
        self._print_startup_info()
        
        self.running = True
        
        # 웹소켓 백그라운드 시작
        self.core.start_websocket()
        
        # 웹소켓 시작 후 콜백 설정
        self._setup_callbacks()
        
        # 주기적 분석 스레드 (옵션)
        if self.config.use_periodic_hybrid:
            self.core.periodic_thread = threading.Thread(target=self._run_periodic_analysis, daemon=True)
            self.core.periodic_thread.start()
        
        # 메인 루프
        self._run_main_loop()
    
    def _print_startup_info(self):
        """시작 정보 출력"""
        print(f"🚀 {self.config.symbol} 통합 스마트 트레이더 시작!")
        print(f"📊 세션: {'활성' if self.config.enable_session_strategy else '비활성'}")
        print(f"⏰ 모드: {'주기(5m)' if self.config.use_periodic_hybrid else '실시간'}")
        print("=" * 60)
        print("💡 실시간 분석 중... 신호가 나올 때만 알림을 표시합니다.")
        print("=" * 60)
    
    def _run_main_loop(self):
        """메인 실행 루프"""
        try:
            last_technical_analysis = None
            last_advanced_liquidation_analysis = None
            api_call_count = 0
            last_api_reset = datetime.datetime.now()
            max_api_calls_per_minute = 2400
            
            while self.running:
                now = datetime.datetime.now()
                
                # API 호출 제한 체크 (1분마다 리셋)
                if (now - last_api_reset).total_seconds() >= 60:
                    api_call_count = 0
                    last_api_reset = now
                
                # 정각 1분마다 세션 전략 분석 (00초)
                if (now.second == 0 and 
                    (not last_technical_analysis or 
                        (now - last_technical_analysis).total_seconds() >= 60)):
                    
                    # API 호출 제한 체크
                    if api_call_count < max_api_calls_per_minute:
                        # 정각 1분마다 세션 전략 분석 실행
                        self._analyze_realtime_technical()
                        last_technical_analysis = now
                        api_call_count += 1
                        # print(f"📊 정각 1분 분석: {now.strftime('%H:%M')}")  # 조용한 모드
                    else:
                        # API 제한 도달 시 5초 대기
                        if not last_technical_analysis or (now - last_technical_analysis).total_seconds() > 5:
                            print(f"⚠️ API 제한 도달, 5초 대기...")
                            self._analyze_realtime_technical()
                            last_technical_analysis = now
                            api_call_count += 1
                
                # 고급 청산 전략을 30초마다 실행 (더 자주 분석)
                if (not last_advanced_liquidation_analysis or 
                    (now - last_advanced_liquidation_analysis).total_seconds() >= 30):
                    
                    if api_call_count < max_api_calls_per_minute:
                        websocket = self.core.get_websocket()
                        if websocket and websocket.price_history:
                            # 고급 청산 전략 분석 실행
                            advanced_signal = self._analyze_advanced_liquidation_strategy(websocket)
                            
                            # 분석 결과 출력 (디버깅 정보 포함)
                            if advanced_signal:
                                print(f"🔍 고급 청산 분석 결과: {now.strftime('%H:%M:%S')}")
                                print(f"   - 신호: {advanced_signal.get('action', 'UNKNOWN')}")
                                print(f"   - 등급: {advanced_signal.get('tier', 'UNKNOWN')}")
                                print(f"   - 전략: {advanced_signal.get('playbook', 'UNKNOWN')}")
                                print(f"   - 점수: {advanced_signal.get('total_score', 0.00):.3f}")
                                print(f"   - 이유: {advanced_signal.get('reason', 'N/A')}")
                                
                                # 중요 신호인 경우 강조 표시
                                if advanced_signal.get('tier') in ['ENTRY', 'SETUP']:
                                    print(f"⚡ ⚡ ⚡ 중요 신호 감지! ⚡ ⚡ ⚡")
                            else:
                                print(f"🔍 고급 청산 분석: {now.strftime('%H:%M:%S')} - 신호 없음")
                            
                            last_advanced_liquidation_analysis = now
                            api_call_count += 1
                
                # 웹소켓 콜백으로 인한 자동 분석은 별도로 처리 (가격 변동, 청산 등)
                # 여기서는 정각 1분마다만 분석 실행
                
                time.sleep(1)  # 1초마다 체크
                    
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중지됨")
        finally:
            self.stop()
    
    def stop(self):
        """트레이더 중지"""
        self.running = False
        self.core.stop_websocket()
        print("🛑 통합 스마트 자동 트레이더 중지됨")


# ==================== 메인 실행 부분 ====================

def main():
    """메인 함수"""
    try:
        config = IntegratedConfig()
        trader = IntegratedSmartTrader(config)
        trader.start()
    except KeyboardInterrupt:
        print("\n⏹️ 프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
