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
        
        # 상태 및 통계 초기화
        self._init_state_and_stats()
        
        # 콜백 설정
        self._setup_callbacks()
    
    def _init_state_and_stats(self):
        """상태 및 통계 초기화"""
        # 신호 관련
        self.signal_count = 0
        self.last_signal_time = None
        self.last_5min_analysis = None
        
        # 신호 관련 통계
        self.last_signal_time = None
        
        # 거래량 급증 집계
        self.volume_spike_buffer = []
        self.last_volume_summary = None
        self.volume_summary_cooldown = 30
    
    def _setup_callbacks(self):
        """웹소켓 콜백 설정"""
        callbacks = {
            'liquidation': lambda data: self.websocket_handler.on_liquidation(
                data, 
                self.display_handler.print_current_liquidation_density,
                self._analyze_realtime_liquidation
            ),
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
            
            # 고급 청산 전략 분석
            advanced_liquidation_signal = self._analyze_advanced_liquidation_strategy(websocket)
            
            # 통합 신호 생성
            if session_signal or advanced_liquidation_signal:
                final_signal = self.core.get_integrated_strategy().get_integrated_signal(
                    session_signal=session_signal,
                    advanced_liquidation_signal=advanced_liquidation_signal
                )
                
                if final_signal:
                    self._process_integrated_signal(final_signal)
                
        except Exception as e:
            print(f"❌ 실시간 기술적 분석 오류: {e}")
    
    def _analyze_realtime_liquidation(self):
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
            
            # 디버깅: 세션 시작 시간 직접 확인
            session_start = session_strategy.get_session_start_time(current_time)
            print(f"🔍 디버깅: 세션 시작 시간: {session_start}")
            
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
            
            # 최근 청산 이벤트 가져오기 (웹소켓에서)
            liquidation_events = websocket.get_recent_liquidations(5)  # 최근 5분
            
            # 고급 청산 전략 분석
            advanced_signal = self.core.get_integrated_strategy().analyze_advanced_liquidation_strategy(
                df_1m, liquidation_events, key_levels, opening_range, vwap, vwap_std
            )
            
            return advanced_signal
            
        except Exception as e:
            print(f"❌ 고급 청산 전략 분석 오류: {e}")
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
                        # 신호가 없어도 분석 상태 출력
                        current_price = websocket.price_history[-1]['price'] if websocket.price_history else 0
                        print(f"📊 주기적 분석 완료 - 신호 없음")
                        print(f"   💰 현재가: ${current_price:.2f}")
                        print(f"   📈 세션 전략: {'활성' if self.config.enable_session_strategy else '비활성'}")
                        print(f"   ⚡ 고급 청산 전략: {'활성' if self.config.enable_advanced_liquidation else '비활성'}")
                        print(f"   ⏰ 다음 분석: {(next_candle + datetime.timedelta(minutes=5)).strftime('%H:%M:%S')}")
                    
                    self.last_5min_analysis = now
                    print(f"✅ {now.strftime('%H:%M:%S')} - 5분봉 분석 완료")
                
                    # 다음 5분봉까지 대기 (더 짧은 간격으로 체크)
                    time.sleep(30)  # 30초마다 체크
                else:
                    # 다음 5분봉까지 대기 (더 짧은 간격으로 체크)
                    time.sleep(10)  # 10초마다 체크
                    
            except Exception as e:
                print(f"❌ 주기적 분석 오류: {e}")
                time.sleep(10)
    
    def _process_integrated_signal(self, signal: Dict):
        """통합 신호 처리 - 깔끔하게 정리"""
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
        """세션 신호 출력"""
        try:
            action = signal.get('action', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            
            print(f"\n📊 SESSION 전략: {action} | {now.strftime('%H:%M:%S')}")
            print(f"💰 ${entry_price:.2f} | 🎯 {confidence:.0%}")
            print(f"🛑 ${stop_loss:.2f} | 🎯 ${take_profit:.2f}")
            
            reason = signal.get('reason', '')
            if reason:
                print(f"📝 {reason}")
                
        except Exception as e:
            print(f"❌ 세션 신호 출력 오류: {e}")
    
    def _print_advanced_liquidation_signal(self, signal: Dict, now: datetime.datetime):
        """고급 청산 신호 출력"""
        try:
            action = signal.get('action', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            
            print(f"\n⚡ 고급 청산 전략: {action} | {now.strftime('%H:%M:%S')}")
            print(f"💰 ${entry_price:.2f} | 🎯 {confidence:.0%}")
            print(f"🛑 ${stop_loss:.2f} | 🎯 ${take_profit:.2f}")
            
            reason = signal.get('reason', '')
            if reason:
                print(f"📝 {reason}")
                
        except Exception as e:
            print(f"❌ 고급 청산 신호 출력 오류: {e}")
    
    def _print_integrated_signal(self, signal: Dict, now: datetime.datetime):
        """통합 신호 출력"""
        try:
            signal_type = signal.get('signal_type', 'UNKNOWN')
            action = signal.get('action', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            entry_price = signal.get('entry_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)
            
            signal_icon = self._get_signal_icon(signal_type)
            signal_name = self._get_signal_name(signal_type)
            
            print(f"\n{signal_icon} {signal_name}: {action} | {now.strftime('%H:%M:%S')}")
            print(f"💰 ${entry_price:.2f} | 🎯 {confidence:.0%}")
            print(f"🛑 ${stop_loss:.2f} | 🎯 ${take_profit:.2f}")
            
            reason = signal.get('reason', '')
            if reason:
                print(f"📝 {reason}")
                
        except Exception as e:
            print(f"❌ 통합 신호 출력 오류: {e}")
    
    def _get_signal_icon(self, signal_type: str) -> str:
        """신호 타입별 아이콘 반환"""
        icons = {
            'SESSION': '📊',
            'ADVANCED_LIQUIDATION': '⚡',
            'INTEGRATED_LIQUIDATION': '🔥',
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
    
    def _print_status(self):
        """상태 출력 - 간단하게"""
        websocket = self.core.get_websocket()
        liquidation_stats = websocket.get_liquidation_stats(5)
        volume_analysis = websocket.get_volume_analysis(3)
        
        print(f"\n📊 통합 상태 | {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"🔥 청산: {liquidation_stats['total_count']}개 (${liquidation_stats['total_value']:,.0f})")
        print(f"📈 거래량: {volume_analysis['volume_trend']} ({volume_analysis['volume_ratio']:.1f}x)")
        print(f"🎯 신호: {self.signal_count}개")
        print(f"📊 세션 전략: {'활성' if self.config.enable_session_strategy else '비활성'}")
        print(f"⚡ 고급 청산 전략: {'활성' if self.config.enable_advanced_liquidation else '비활성'}")
        
        if self.last_signal_time:
            time_since = datetime.datetime.now() - self.last_signal_time
            print(f"⏰ 마지막 신호: {format_time_delta(time_since)} 전")
    
    def start(self):
        """트레이더 시작"""
        self._print_startup_info()
        
        self.running = True
        
        # 웹소켓 백그라운드 시작
        self.core.start_websocket()
        
        # 주기적 분석 스레드 (옵션)
        if self.config.use_periodic_hybrid:
            self.core.periodic_thread = threading.Thread(target=self._run_periodic_analysis, daemon=True)
            self.core.periodic_thread.start()
        
        # 메인 루프
        self._run_main_loop()
    
    def _print_startup_info(self):
        """시작 정보 출력"""
        print(f"🚀 {self.config.symbol} 통합 스마트 자동 트레이더 시작! (리팩토링 버전)")
        print(f"📊 세션 전략: {'활성' if self.config.enable_session_strategy else '비활성'}")
        print(f"⚡ 고급 청산 전략: {'활성' if self.config.enable_advanced_liquidation else '비활성'}")
        print(f"🔥 청산 전략: {'활성' if self.config.enable_liquidation_strategy else '비활성'}")
        print(f"⏰ 모드: {'주기(5m)' if self.config.use_periodic_hybrid else '실시간'}")
        print(f"📈 신호 민감도: 높음")
        print(f"📊 주기적 분석: 5분봉 기반 (세션 + 고급 청산)")
        print(f"📊 실시간 분석: 정각 1분마다 (세션 + 고급 청산)")
        print(f"📊 거래량 급증 집계: 30초마다 요약 출력")
        print(f"💰 가격 변동 감지: 0.1% 이상 (스캘핑용)")
        print(f"🛡️ API 제한 보호: 분당 최대 1200회 (제한 도달 시 5초 대기)")
        print(f"🔥 청산 임계값: {self.config.liquidation_min_count}개, ${self.config.liquidation_min_value:,.0f}")
        print("=" * 60)
        print("💡 실시간 분석 중... 신호가 나올 때만 알림을 표시합니다.")
        print("💡 거래량 급증은 3.0x 이상일 때만 감지됩니다 (노이즈 감소).")
        print("💡 거래량 급증은 30초마다 요약해서 표시됩니다.")
        print("💡 실시간 분석: 0.1% 가격 변동 감지, 정각 1분마다 전략 분석")
        print("💡 주기적 분석: 5분봉 기반 자동 실행")
        print("💡 청산 밀도 분석: 1분마다 자동 출력")
        print("💡 API 제한 보호: 분당 1200회 초과 시 자동으로 5초 대기")
        print("=" * 60)
    
    def _run_main_loop(self):
        """메인 실행 루프"""
        try:
            last_technical_analysis = None
            last_status_output = datetime.datetime.now()
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
                        print(f"📊 정각 1분 분석 실행: {now.strftime('%H:%M:%S')}")
                    else:
                        # API 제한 도달 시 5초 대기
                        if not last_technical_analysis or (now - last_technical_analysis).total_seconds() > 5:
                            print(f"⚠️ API 호출 제한 도달, 5초 대기 중... ({api_call_count}/분)")
                            self._analyze_realtime_technical()
                            last_technical_analysis = now
                            api_call_count += 1
                
                # 웹소켓 콜백으로 인한 자동 분석은 별도로 처리 (가격 변동, 청산 등)
                # 여기서는 정각 1분마다만 분석 실행
                
                # 통계 출력 (1분마다)
                if (now - last_status_output).total_seconds() >= 60:
                    self._print_status()
                    last_status_output = now
                
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
