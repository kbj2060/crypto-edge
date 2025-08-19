#!/usr/bin/env python3
"""
통합 스마트 자동 트레이더 (리팩토링 버전)
하이브리드 전략(5분봉) + 실시간 청산 전략의 시너지 효과를 활용합니다.
"""

import time
import datetime
import threading
from typing import Dict, Any, Optional
from core.trader_core import TraderCore
from analyzers.liquidation_analyzer import LiquidationAnalyzer
from analyzers.technical_analyzer import TechnicalAnalyzer
from handlers.websocket_handler import WebSocketHandler
from handlers.display_handler import DisplayHandler
from utils.trader_utils import get_next_5min_candle_time, format_time_delta
from config.integrated_config import IntegratedConfig
import pandas as pd


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
        self.synergy_count = 0
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
                self._analyze_realtime_technical
            ),
            'kline': lambda data: self.websocket_handler.on_kline(
                data, 
                self._analyze_realtime_technical
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
            # 기존 실시간 기술적 분석만 실행
            integrated_signal = self.technical_analyzer.analyze_realtime_technical(
                self.core.get_websocket(),
                self.core.get_integrated_strategy(),
                self.liquidation_analyzer
            )
            
            if integrated_signal:
                self._process_integrated_signal(integrated_signal)
                
        except Exception as e:
            print(f"❌ 실시간 기술적 분석 오류: {e}")
    
    def _analyze_realtime_liquidation(self):
        """실시간 청산 신호 분석"""
        try:
            # 현재 가격과 ATR 가져오기
            websocket = self.core.get_websocket()
            if not websocket.price_history:
                return
            
            current_price = websocket.price_history[-1]['price']
            
            # ATR 계산
            atr = self.liquidation_analyzer._calculate_current_atr()
            if not atr:
                atr = current_price * 0.02  # 기본값
            
            # 청산 통계 분석
            liquidation_stats = websocket.get_liquidation_stats(self.config.liquidation_window_minutes)
            volume_analysis = websocket.get_volume_analysis(3)
            
            # 청산 신호 분석
            liquidation_signal = self.core.get_integrated_strategy().analyze_liquidation_strategy(
                liquidation_stats, volume_analysis, current_price, atr
            )
            
            # 청산 예측 분석
            recent_liquidations = websocket.get_recent_liquidations(self.config.liquidation_window_minutes)
            prediction_signal = self.core.get_integrated_strategy().analyze_liquidation_prediction(
                recent_liquidations, current_price
            )
            
            # 폭등/폭락 경고 생성 (안전한 호출)
            try:
                explosion_alert = self.core.get_integrated_strategy().get_explosion_alert(
                    hybrid_signal=self.core.get_integrated_strategy().last_hybrid_signal,
                    liquidation_signal=liquidation_signal,
                    prediction_signal=prediction_signal
                )
                
                if explosion_alert:
                    self._process_explosion_alert(explosion_alert)
            except Exception as e:
                print(f"⚠️ 폭등/폭락 경고 생성 오류: {e}")
            
            if liquidation_signal or prediction_signal:
                # 통합 신호 생성
                integrated_signal = self.core.get_integrated_strategy().get_integrated_signal(
                    hybrid_signal=self.core.get_integrated_strategy().last_hybrid_signal,
                    liquidation_signal=liquidation_signal,
                    prediction_signal=prediction_signal
                )
                
                if integrated_signal:
                    self._process_integrated_signal(integrated_signal)
            
        except Exception as e:
            print(f"❌ 실시간 청산 분석 오류: {e}")
    
    def _run_hybrid_analysis_quick(self):
        """빠른 하이브리드 분석 (10초마다 실행)"""
        try:
            # 하이브리드 전략 분석
            hybrid_signal = self.technical_analyzer.analyze_hybrid_strategy(
                self.core.get_websocket(),
                self.core.get_integrated_strategy()
            )
            
            # ENHANCED_LIQUIDATION 신호 분석
            enhanced_liquidation_signal = self._analyze_enhanced_liquidation()
            
            # 10초마다 정리된 신호 출력
            self._print_10sec_signals_summary(hybrid_signal, enhanced_liquidation_signal)
            
        except Exception as e:
            print(f"❌ 빠른 하이브리드 분석 오류: {e}")
    
    def _run_hybrid_analysis(self):
        """하이브리드 전략 분석 (5분봉 기반)"""
        while self.running:
            try:
                # 5분봉 타이밍 계산
                next_candle = get_next_5min_candle_time()
                now = datetime.datetime.now()
                
                if now >= next_candle:
                    # 1초 후 분석 시작
                    time.sleep(1)
                    
                    print(f"\n⏰ {now.strftime('%H:%M:%S')} - 5분봉 하이브리드 분석 시작")
                    
                    # 하이브리드 전략 분석
                    integrated_signal = self.technical_analyzer.analyze_hybrid_strategy(
                        self.core.get_websocket(),
                        self.core.get_integrated_strategy()
                    )
                    
                    if integrated_signal:
                        print(f"🎯 하이브리드 신호 생성됨!")
                        self._process_integrated_signal(integrated_signal)
                    else:
                        # 신호가 없어도 분석 상태 출력
                        current_price = self.core.get_websocket().price_history[-1]['price'] if self.core.get_websocket().price_history else 0
                        print(f"📊 하이브리드 분석 완료 - 신호 없음")
                        print(f"   💰 현재가: ${current_price:.2f}")
                        print(f"   📈 신뢰도 임계값: {self.config.hybrid_min_confidence:.1%}")
                        print(f"   ⏰ 다음 분석: {(next_candle + datetime.timedelta(minutes=5)).strftime('%H:%M:%S')}")
                    
                    self.last_5min_analysis = now
                    print(f"✅ {now.strftime('%H:%M:%S')} - 5분봉 분석 완료")
                
                    # 다음 5분봉까지 대기 (더 짧은 간격으로 체크)
                    time.sleep(30)  # 30초마다 체크 (1분에서 변경)
                else:
                    # 다음 5분봉까지 대기 (더 짧은 간격으로 체크)
                    time.sleep(10)  # 10초마다 체크 (1초에서 변경)
                    
            except Exception as e:
                print(f"❌ 하이브리드 분석 오류: {e}")
                time.sleep(10)
    
    def _process_integrated_signal(self, signal: Dict):
        """통합 신호 처리"""
        try:
            signal_type = signal.get('signal_type', 'UNKNOWN')
            action = signal.get('final_signal') or signal.get('action')
            confidence = signal.get('confidence', 0)
            risk_reward = signal.get('risk_reward', 0)
            
            # 진입가 설정 (HYBRID 신호의 경우 current_price를 entry_price로 사용)
            entry_price = signal.get('entry_price', 0)
            if entry_price == 0 and signal_type == 'HYBRID':
                entry_price = signal.get('current_price', 0)
            
            stop_loss = signal.get('stop_loss', 0)
            take_profit1 = signal.get('take_profit1', 0)
            take_profit2 = signal.get('take_profit2', 0)
            
            # 현재 시간 기록
            now = datetime.datetime.now()
            
            # 시너지 신호 특별 처리
            if signal_type == 'SYNERGY':
                print(f"\n🔥🔥🔥 SYNERGY 신호! 🔥🔥🔥")
                print(f"🎯 {action} - {now.strftime('%H:%M:%S')}")
                print(f"💰 ${entry_price:.2f} | 📊 {confidence:.1%} | ⚖️ {risk_reward:.1f}")
                print(f"🛑 ${stop_loss:.2f} | 💎 ${take_profit1:.2f} | 💎 ${take_profit2:.2f}")
                print(f"🔍 {signal.get('synergy_reason', '')}")
                self.synergy_count += 1
            else:
                # 일반 신호 출력
                if action == "BUY":
                    print(f"\n📈 {signal_type} BUY 신호 - {now.strftime('%H:%M:%S')}")
                    print(f"💰 ${entry_price:.2f} | 📊 {confidence:.1%} | ⚖️ {risk_reward:.1f}")
                    print(f"🛑 ${stop_loss:.2f} | 💎 ${take_profit1:.2f} | 💎 ${take_profit2:.2f}")
                elif action == "SELL":
                    print(f"\n📉 {signal_type} SELL 신호 - {now.strftime('%H:%M:%S')}")
                    print(f"💰 ${entry_price:.2f} | 📊 {confidence:.1%} | ⚖️ {risk_reward:.1f}")
                    print(f"🛑 ${stop_loss:.2f} | 💎 ${take_profit1:.2f} | 💎 ${take_profit2:.2f}")
            
            # 신호 통계 업데이트
            self.signal_count += 1
            self.last_signal_time = now
            
        except Exception as e:
            print(f"❌ 통합 신호 처리 오류: {e}")
    
    def _process_explosion_alert(self, alert: Dict):
        """폭등/폭락 경고 처리"""
        try:
            total_alerts = alert.get('total_alerts', 0)
            critical_alerts = alert.get('critical_alerts', 0)
            high_alerts = alert.get('high_alerts', 0)
            
            print(f"\n🚨 폭등/폭락 경고 - {datetime.datetime.now().strftime('%H:%M:%S')}")
            print(f"📊 총 경고: {total_alerts}개 (🔥🔥🔥 {critical_alerts}개, 🔥🔥 {high_alerts}개)")
            
            # 개별 경고 출력
            for alert_item in alert.get('alerts', []):
                alert_type = alert_item.get('type', 'UNKNOWN')
                level = alert_item.get('level', 'UNKNOWN')
                message = alert_item.get('message', '')
                
                if level == 'CRITICAL':
                    print(f"🔥🔥🔥 {message}")
                elif level == 'HIGH':
                    print(f"🔥🔥 {message}")
                elif level == 'MEDIUM':
                    print(f"🔥 {message}")
                
                # 예측 정보가 있으면 추가 출력
                if 'expected_time' in alert_item:
                    expected_time = alert_item['expected_time']
                    time_until = expected_time - datetime.datetime.now()
                    formatted_time = format_time_delta(time_until)
                    print(f"⏰ 예상 시간: {expected_time.strftime('%H:%M:%S')} (약 {formatted_time} 후)")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ 폭등/폭락 경고 처리 오류: {e}")
    
    def _print_status(self):
        """상태 출력"""
        websocket = self.core.get_websocket()
        liquidation_stats = websocket.get_liquidation_stats(5)
        volume_analysis = websocket.get_volume_analysis(3)
        signal_summary = self.core.get_integrated_strategy().get_signal_summary()
        
        # 예측 요약 정보
        prediction_summary = self.core.get_integrated_strategy().prediction_strategy.get_prediction_summary()
        
        print(f"\n📊 통합 상태 - {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"🔥 최근 1분 청산: {liquidation_stats['total_count']}개 (${liquidation_stats['total_value']:,.0f})")
        print(f"📈 거래량 트렌드: {volume_analysis['volume_trend']} ({volume_analysis['volume_ratio']:.1f}x)")
        print(f"🎯 총 신호: {self.signal_count}개 | 🔥🔥🔥 시너지: {self.synergy_count}개")
        print(f"🔮 예측 신호: {len(prediction_summary.get('current_predictions', []))}개 | 정확도: {prediction_summary.get('accuracy', 0):.1%}")
        print(f"⚙️ 하이브리드: {'활성' if signal_summary['config']['enable_hybrid'] else '비활성'}")
        print(f"⚙️ 청산: {'활성' if signal_summary['config']['enable_liquidation'] else '비활성'}")
        print(f"⚙️ 시너지: {'활성' if signal_summary['config']['enable_synergy'] else '비활성'}")
        print(f"⚙️ 예측: {'활성' if self.config.enable_liquidation_prediction else '비활성'}")
        print(f"📊 청산 분석: {self.config.liquidation_window_minutes}분 윈도우 | 최소 {self.config.liquidation_min_count}개")
        
        # 청산 밀도 분석 출력
        self.display_handler.print_liquidation_density_analysis()
        
        if self.last_signal_time:
            time_since = datetime.datetime.now() - self.last_signal_time
            print(f"⏰ 마지막 신호: {format_time_delta(time_since)} 전")
        
        if self.last_5min_analysis:
            time_since = datetime.datetime.now() - self.last_5min_analysis
            print(f"⏰ 마지막 5분봉 분석: {format_time_delta(time_since)} 전")
        
        # 현재 예측 신호 출력
        current_predictions = prediction_summary.get('current_predictions', [])
        if current_predictions:
            # 현재 가격 가져오기
            current_price = websocket.price_history[-1]['price'] if websocket.price_history else 0
            
            print(f"\n🔮 현재 예측 신호 (현재가: ${current_price:.2f}):")
            for i, pred in enumerate(current_predictions[:3]):  # 상위 3개만
                pred_type = pred.get('type', 'UNKNOWN')
                confidence = pred.get('confidence', 0)
                target_price = pred.get('target_price', 0)
                
                if current_price > 0 and target_price > 0:
                    # 퍼센트 변화 계산
                    price_change = ((target_price - current_price) / current_price) * 100
                    change_sign = "+" if price_change > 0 else ""
                    
                    if pred_type == 'EXPLOSION_UP':
                        print(f"  {i+1}. 🚀 폭등 예측: ${target_price:.2f} ({change_sign}{price_change:.2f}%) | 신뢰도: {confidence:.1%}")
                    elif pred_type == 'EXPLOSION_DOWN':
                        print(f"  {i+1}. 💥 폭락 예측: ${target_price:.2f} ({change_sign}{price_change:.2f}%) | 신뢰도: {confidence:.1%}")
                else:
                    # 가격 정보가 없을 때 기본 출력
                    if pred_type == 'EXPLOSION_UP':
                        print(f"  {i+1}. 🚀 폭등 예측: ${target_price:.2f} | 신뢰도: {confidence:.1%}")
                    elif pred_type == 'EXPLOSION_DOWN':
                        print(f"  {i+1}. 💥 폭락 예측: ${target_price:.2f} | 신뢰도: {confidence:.1%}")
    
    def _analyze_enhanced_liquidation(self) -> Optional[Dict]:
        """ENHANCED_LIQUIDATION 신호 분석"""
        try:
            websocket = self.core.get_websocket()
            if not websocket.price_history:
                return None
            
            current_price = websocket.price_history[-1]['price']
            recent_liquidations = websocket.get_recent_liquidations(self.config.liquidation_window_minutes)
            liquidation_density = websocket.get_liquidation_density_analysis(current_price, 2.0)
            
            # 청산 데이터 수집 상태 확인
            total_liquidations = len(websocket.liquidations)
            print(f"🔍 청산 데이터 상태: 총 {total_liquidations}개, 최근 {len(recent_liquidations)}개 (윈도우: {self.config.liquidation_window_minutes}분)")
            
            # 청산 데이터가 없으면 중립 신호 생성 (디버깅 출력 없음)
            if not recent_liquidations:
                return {
                    'signal_type': 'ENHANCED_LIQUIDATION',
                    'action': 'NEUTRAL',
                    'confidence': 0.0,
                    'entry_price': current_price,
                    'stop_loss': current_price,
                    'take_profit1': current_price,
                    'take_profit2': current_price,
                    'liquidation_volume': 0.0,
                    'price_momentum': 0.0,
                    'volume_trend': 1.0,
                    'ema_slope': 0.0,
                    'rsi_k': 50.0,
                    'timestamp': datetime.datetime.now(),
                    'reason': f'청산 데이터 없음 - 총 {total_liquidations}개 중 최근 {self.config.liquidation_window_minutes}분 윈도우에 해당하는 데이터 없음'
                }
            
            # 5분봉 데이터 로딩
            df_5m = self._load_5m_data()
            if df_5m.empty:
                return {
                    'signal_type': 'ENHANCED_LIQUIDATION',
                    'action': 'NEUTRAL',
                    'confidence': 0.0,
                    'entry_price': current_price,
                    'stop_loss': current_price,
                    'take_profit1': current_price,
                    'take_profit2': current_price,
                    'liquidation_volume': len(recent_liquidations),
                    'price_momentum': 0.0,
                    'volume_trend': 1.0,
                    'ema_slope': 0.0,
                    'rsi_k': 50.0,
                    'timestamp': datetime.datetime.now(),
                    'reason': '5분봉 데이터 없음 - 대기 중'
                }
            
            # ENHANCED_LIQUIDATION 신호 생성
            enhanced_signal = self.liquidation_analyzer.analyze_liquidation_with_technical(
                recent_liquidations, liquidation_density, df_5m, current_price
            )
            
            return enhanced_signal
            
        except Exception as e:
            print(f"❌ ENHANCED_LIQUIDATION 분석 오류: {e}")
            return None
    
    def _load_5m_data(self) -> pd.DataFrame:
        """5분봉 데이터 로딩"""
        try:
            from data.loader import build_df
            df_5m = build_df(self.config.symbol, '5m', self.config.hybrid_limit_5m, 14,
                            market='futures', price_source='last', ma_type='ema')
            return df_5m
        except Exception:
            return pd.DataFrame()
    
    def _print_10sec_signals_summary(self, hybrid_signal: Optional[Dict], enhanced_signal: Optional[Dict]):
        """10초마다 정리된 신호 요약 출력"""
        now = datetime.datetime.now()
        
        # 하이브리드 신호가 있거나 ENHANCED_LIQUIDATION 신호가 있는 경우 출력
        if hybrid_signal or enhanced_signal:
            print(f"\n⏰ {now.strftime('%H:%M:%S')} - 10초 신호 요약")
            print("=" * 50)
            
            # 하이브리드 신호 출력
            if hybrid_signal:
                self._print_signal_summary("🎯 HYBRID", hybrid_signal)
            
            # ENHANCED_LIQUIDATION 신호 출력
            if enhanced_signal:
                self._print_signal_summary("🔥 ENHANCED_LIQUIDATION", enhanced_signal)
            
            print("=" * 50)
        else:
            # 신호가 없을 때도 중립 상태 출력
            print(f"\n⏰ {now.strftime('%H:%M:%S')} - 10초 신호 요약")
            print("=" * 50)
            print("📊 현재 상태: 신호 없음 (중립)")
            print("  🎯 HYBRID: 대기 중")
            print("  🔥 ENHANCED_LIQUIDATION: 대기 중")
            print("=" * 50)
    
    def _print_signal_summary(self, signal_type: str, signal: Dict):
        """개별 신호 요약 출력"""
        try:
            action = signal.get('final_signal') or signal.get('action', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            entry_price = signal.get('entry_price') or signal.get('current_price', 0)
            stop_loss = signal.get('stop_loss', 0)
            take_profit1 = signal.get('take_profit1', 0)
            take_profit2 = signal.get('take_profit2', 0)
            
            if action == "NEUTRAL":
                print(f"{signal_type} NEUTRAL 신호")
                print(f"  📊 현재가: ${entry_price:.2f}")
                print(f"  🎯 상태: 대기 중")
                if 'reason' in signal:
                    print(f"  💡 이유: {signal['reason']}")
                
                # 추가 정보가 있으면 출력
                if 'liquidation_volume' in signal:
                    print(f"  🔥 청산량: {signal['liquidation_volume']:.2f} ETH")
                if 'price_momentum' in signal:
                    print(f"  📈 가격모멘텀: {signal['price_momentum']:+.2f}%")
                if 'ema_slope' in signal:
                    print(f"  📉 EMA 기울기: {signal['ema_slope']:+.4f}%")
                if 'rsi_k' in signal:
                    print(f"  🔄 RSI_K: {signal['rsi_k']:.2f}")
                
                print()  # 빈 줄 추가
            else:
                print(f"{signal_type} {action} 신호")
                print(f"  💰 진입가: ${entry_price:.2f}")
                print(f"  📊 신뢰도: {confidence:.1%}")
                print(f"  🛑 손절가: ${stop_loss:.2f}")
                print(f"  💎 익절가1: ${take_profit1:.2f}")
                print(f"  💎 익절가2: ${take_profit2:.2f}")
                
                # 추가 정보가 있으면 출력
                if 'liquidation_volume' in signal:
                    print(f"  🔥 청산량: {signal['liquidation_volume']:.2f} ETH")
                if 'price_momentum' in signal:
                    print(f"  📈 가격모멘텀: {signal['price_momentum']:+.2f}%")
                
                print()  # 빈 줄 추가
                
        except Exception as e:
            print(f"❌ 신호 요약 출력 오류: {e}")
    
    
    def start(self):
        """트레이더 시작"""
        self._print_startup_info()
        
        self.running = True
        
        # 웹소켓 백그라운드 시작
        self.core.start_websocket()
        
        # 하이브리드 분석 스레드 (옵션)
        if self.config.use_periodic_hybrid:
            self.core.hybrid_thread = threading.Thread(target=self._run_hybrid_analysis, daemon=True)
            self.core.hybrid_thread.start()
        
        # 메인 루프
        self._run_main_loop()
    
    def _print_startup_info(self):
        """시작 정보 출력"""
        print(f"🚀 {self.config.symbol} 통합 스마트 자동 트레이더 시작! (리팩토링 버전)")
        print(f"📊 하이브리드 전략: {'활성' if self.config.enable_hybrid_strategy else '비활성'}")
        print(f"🔥 청산 전략: {'활성' if self.config.enable_liquidation_strategy else '비활성'}")
        print(f"🎯 시너지 신호: {'활성' if self.config.enable_synergy_signals else '비활성'}")
        print(f"🔮 청산 예측: {'활성' if self.config.enable_liquidation_prediction else '비활성'}")
        print(f"⏰ 모드: {'주기(5m)' if self.config.use_periodic_hybrid else '실시간'}")
        print(f"📈 신호 민감도: 높음 (신뢰도 임계값: {self.config.hybrid_min_confidence:.1%})")
        print(f"📊 주기적 분석: 10초마다 (스캘핑용 - API 제한 고려)")
        print(f"📊 하이브리드 분석: 10초마다 (실시간 모드)")
        print(f"📊 거래량 급증 집계: 30초마다 요약 출력 (개별 출력 제한)")
        print(f"💰 가격 변동 감지: 0.1% 이상 (스캘핑용)")
        print(f"🛡️ API 제한 보호: 분당 최대 1200회 (제한 도달 시 5초 대기)")
        print(f"🔥 청산 임계값: {self.config.liquidation_min_count}개, ${self.config.liquidation_min_value:,.0f}")
        print(f"🔮 예측 설정: 밀도 {self.config.prediction_min_density}개, 연쇄 {self.config.prediction_cascade_threshold}개")
        print("=" * 60)
        print("💡 실시간 분석 중... 신호가 나올 때만 알림을 표시합니다.")
        print("💡 거래량 급증은 3.0x 이상일 때만 감지됩니다 (노이즈 감소).")
        print("💡 거래량 급증은 30초마다 요약해서 표시됩니다.")
        print("💡 실시간 분석: 0.1% 가격 변동 감지, 10초마다 기술적 분석")
        print("💡 하이브리드 분석: 10초마다 자동 실행")
        print("💡 청산 밀도 분석: 1분마다 자동 출력")
        print("💡 API 제한 보호: 분당 1200회 초과 시 자동으로 5초 대기")
        print("=" * 60)
    
    def _run_main_loop(self):
        """메인 실행 루프"""
        try:
            last_technical_analysis = None
            last_status_output = datetime.datetime.now()  # 상태 출력 타이머 추가
            api_call_count = 0
            last_api_reset = datetime.datetime.now()
            max_api_calls_per_minute = 2400  # 바이낸스 분당 최대 호출 제한 (안전하게 설정)
            
            while self.running:
                now = datetime.datetime.now()
                
                # API 호출 제한 체크 (1분마다 리셋)
                if (now - last_api_reset).total_seconds() >= 60:
                    api_call_count = 0
                    last_api_reset = now
                
                # 주기적 기술적 분석 (10초마다 - 스캘핑용, API 제한 고려)
                if (not last_technical_analysis or 
                    (now - last_technical_analysis).total_seconds() > 10):
                    
                    # API 호출 제한 체크
                    if api_call_count < max_api_calls_per_minute:
                        # 하이브리드 분석 실행 (10초마다) - 한 번만 실행
                        self._run_hybrid_analysis_quick()
                        # 실시간 기술적 분석은 하이브리드 분석과 별도로 실행하지 않음
                        # (하이브리드 분석에서 이미 모든 분석을 수행)
                        last_technical_analysis = now
                        api_call_count += 1
                    else:
                        # API 제한 도달 시 5초 대기
                        if not last_technical_analysis or (now - last_technical_analysis).total_seconds() > 5:
                            print(f"⚠️ API 호출 제한 도달, 5초 대기 중... ({api_call_count}/분)")
                            # API 제한 상황에서는 하이브리드 분석만 실행
                            self._run_hybrid_analysis_quick()
                            last_technical_analysis = now
                            api_call_count += 1
                
                # 통계 출력 (1분마다) - 별도 타이머 사용
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
