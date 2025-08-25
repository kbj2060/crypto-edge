#!/usr/bin/env python3
"""
통합 스마트 자동 트레이더 (리팩토링 버전)
실시간 청산 전략 + 세션 기반 전략 + 고급 청산 전략을 활용합니다.
"""

import time
import datetime
import threading
import requests
import pandas as pd
from typing import Dict, Any, Optional, List
from core.trader_core import TraderCore

from config.integrated_config import IntegratedConfig
from data.data_manager import get_data_manager
from indicators.global_indicators import get_global_indicator_manager

class IntegratedSmartTrader:
    """통합 스마트 자동 트레이더 (리팩토링 버전)"""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self.running = False
        
        # 핵심 컴포넌트 초기화
        self.core = TraderCore(config)
        self.global_manager = get_global_indicator_manager()
        
        # 상태 관리
        self.running = False
        self.last_analysis_time = None
        self.last_3min_analysis = None
        self.last_60sec_bucket = None
        
        # 청산 버킷 관리 (60초 단위)
        self.liquidation_bucket = []
        self.bucket_start_time = self.core.time_manager.get_current_time()
        self.last_60sec_bucket = None
        
        # 🚀 1단계: DataManager 우선 초기화 (데이터 준비)
        self._init_data_manager()
        
        # 🚀 2단계: 글로벌 지표 시스템 초기화
        self._init_global_indicators()
        
        # 🚀 3단계: 고급 청산 전략 초기화
        self._init_advanced_liquidation_strategy()
        
        # 🚀 4단계: 세션 기반 전략 초기화
        self._init_session_strategy()
    
    def _init_data_manager(self):
        """DataManager 우선 초기화 (데이터 준비)"""
        try:
            print("🚀 1단계: DataManager 우선 초기화 시작...")
            
            from data.data_manager import get_data_manager
            
            # DataManager 싱글톤 인스턴스 가져오기
            data_manager = get_data_manager()
            
            # 초기 데이터 로딩 (전날 00시부터 현재까지)
            print("📊 DataManager 초기 데이터 로딩 시작...")
            data_loaded = data_manager.load_initial_data('ETHUSDT')

            if data_loaded:
                print(f"🎯 중앙 데이터 저장소 준비 완료!")
            else:
                print("❌ DataManager 초기화 실패")
                raise Exception("DataManager 초기 데이터 로딩 실패")
                
        except Exception as e:
            print(f"❌ DataManager 초기화 오류: {e}")
            import traceback
            traceback.print_exc()
            raise  # 상위로 에러 전파하여 프로그램 중단
    
    def _init_global_indicators(self):
        """글로벌 지표 시스템 초기화"""
        try:
            print("🚀 2단계: 글로벌 지표 시스템 초기화 시작...")
            
            from indicators.global_indicators import get_global_indicator_manager
            
            # 글로벌 지표 매니저 가져오기
            global_manager = get_global_indicator_manager()
            
            # 지표들 초기화 (DataManager가 이미 준비된 상태)
            global_manager.initialize_indicators()
            
            print("🎯 글로벌 지표 시스템 초기화 완료!")
            
        except Exception as e:
            print(f"❌ 글로벌 지표 시스템 초기화 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _init_advanced_liquidation_strategy(self):
        """고급 청산 전략 초기화"""
        try:
            print("🚀 고급 청산 전략 초기화 시작...")
            
            from signals.advanced_liquidation_strategy import AdvancedLiquidationStrategy, AdvancedLiquidationConfig
            
            adv_config = AdvancedLiquidationConfig()
            
            self._adv_liquidation_strategy = AdvancedLiquidationStrategy(adv_config)
            
            # 외부 서버에서 청산 데이터 가져와서 워밍업
            print("🌐 외부 서버에서 청산 데이터 가져오기 시작...")
            external_liquidation_data = self._fetch_external_liquidation_data()
            
            if external_liquidation_data:
                print(f"📊 외부 데이터 {len(external_liquidation_data)}개 수신, 워밍업 시작")
                self._warmup_strategy_with_data(external_liquidation_data)
            else:
                print("⚠️ 외부 청산 데이터가 없어 워밍업을 건너뜀")
                
            print("🎯 고급 청산 전략 초기화 완료!")
                
        except Exception as e:
            print(f"❌ 고급 청산 전략 초기화 오류: {e}")
            import traceback
            traceback.print_exc()
            self._adv_liquidation_strategy = None
    
    def _init_session_strategy(self):
        """세션 기반 전략 초기화"""
        try:
            if not self.config.enable_session_strategy:
                print("⚠️ 세션 전략이 비활성화됨")
                self._session_strategy = None
                return
                
            print("🚀 세션 기반 전략 초기화 시작...")
            
            from signals.session_based_strategy import SessionBasedStrategy, SessionConfig
            
            session_config = SessionConfig()
            
            self._session_strategy = SessionBasedStrategy(session_config)
            
            print("🎯 세션 기반 전략 초기화 완료!")
                
        except Exception as e:
            print(f"❌ 세션 기반 전략 초기화 오류: {e}")
            import traceback
            traceback.print_exc()
            self._session_strategy = None
    

    def _fetch_external_liquidation_data(self) -> List[Dict]:
        """외부 서버에서 청산 데이터 가져오기"""
        try:
            # 외부 API 엔드포인트에서 최근 24시간 청산 데이터 가져오기
            import requests
            
            # 외부 서버 URL
            external_server_url = getattr(self.config, 'external_server_url', None)
            if not external_server_url:
                print("⚠️ 외부 청산 데이터 API URL이 설정되지 않음")
                return []
            
            # 엔드포인트 구성
            external_api_url = f"{external_server_url.rstrip('/')}/liquidations"
            
            print(f"🔍 외부 API 요청 URL: {external_api_url}")
            
            # API 요청 헤더 (인증이 필요한 경우)
            headers = {}
            if hasattr(self.config, 'external_api_key'):
                headers['Authorization'] = f'Bearer {self.config.external_api_key}'
            
            # 외부 서버에서 데이터 가져오기
            response = requests.get(external_api_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # 응답 내용 확인 및 디버깅
            response_text = response.text.strip()
            if not response_text:
                print("⚠️ 외부 API에서 빈 응답을 받았습니다.")
                return []

            # 외부 데이터를 내부 형식으로 변환
            external_data = response.json()
            liquidation_data = []
            
            # 응답 구조 확인 및 데이터 추출
            if isinstance(external_data, list):
                data_items = external_data
            else:
                print("⚠️ 외부 API 응답이 리스트 형태가 아닙니다.")
                return []
            
            if not data_items:
                print("⚠️ 외부 API 응답에 데이터가 없습니다.")
                return []
            
            for item in data_items:                    
                # 타임스탬프 처리 (UST -> KST 변환)
                timestamp = item.get('timestamp')
                
                # datetime 문자열을 UTC 타임스탬프로 변환 후 KST로 조정
                # '2025-08-21 16:06:51.478000' 형식 파싱
                dt = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
                utc_timestamp = dt.timestamp()
                kst_timestamp = utc_timestamp + (9 * 3600)  # UTC+9
                timestamp = int(kst_timestamp)
                
                # 변환된 데이터 생성
                converted_data = {
                    'timestamp': timestamp,
                    'symbol': item.get('symbol', self.config.symbol),
                    'side': item.get('side', 'unknown'),
                    'size': item.get('size', 0.0),
                    'price': item.get('price', 0.0)
                }
                
                liquidation_data.append(converted_data)
            
            # 데이터 품질 검증 및 개선
            long_count = sum(1 for item in liquidation_data if item.get('side') == 'SELL')
            short_count = sum(1 for item in liquidation_data if item.get('side') == 'BUY')
            
            print(f"🌐 외부 서버에서 {len(liquidation_data)}개의 청산 데이터를 가져왔습니다.")
            print(f"📊 데이터 품질: 롱 {long_count}개, 숏 {short_count}개")

            
            if long_count < 5:
                print("⚠️ 롱 청산 데이터가 부족합니다 (5개 필요)")
            if short_count < 5:
                print("⚠️ 숏 청산 데이터가 부족합니다 (5개 필요)")
            
            return liquidation_data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 외부 API 요청 오류: {e}")
            return []
        except ValueError as e:
            print(f"❌ 외부 API 응답 파싱 오류: {e}")
            return []
        except Exception as e:
            print(f"❌ 외부 청산 데이터 가져오기 오류: {e}")
            return []
    
    def _warmup_strategy_with_data(self, liquidation_data: List[Dict]):
        """외부 데이터로 전략 워밍업"""
        if not hasattr(self, '_adv_liquidation_strategy') or not self._adv_liquidation_strategy:
            print("⚠️ 고급 청산 전략이 초기화되지 않음")
            return
        
        # 워밍업 전 데이터 품질 재확인
        long_count = sum(1 for item in liquidation_data if item.get('side') == 'SELL')
        short_count = sum(1 for item in liquidation_data if item.get('side') == 'BUY')
        
        print(f"🔥 전략 워밍업 시작: {len(liquidation_data)}개 이벤트")
        print(f"📊 워밍업 데이터 품질: 롱 {long_count}개, 숏 {short_count}개")
        
        # 워밍업 가능 여부 확인
        if long_count < 5 or short_count < 5:
            print("⚠️ 워밍업 데이터가 부족하여 전략 성능이 제한될 수 있습니다")
            print("💡 제한된 데이터로도 최대한의 워밍업을 시도합니다")
        else:
            print("✅ 워밍업 데이터 품질이 양호합니다")
        
        try:
            processed_count = 0
            
            # 데이터가 부족할 경우 반복 처리로 워밍업 효과 증대
            repeat_count = 1
            if len(liquidation_data) < 50:
                repeat_count = 2  # 데이터가 적으면 2번 반복
                print(f"🔄 데이터 부족으로 {repeat_count}번 반복 워밍업 수행")
            
            for repeat in range(repeat_count):
                if repeat > 0:
                    print(f"🔄 {repeat+1}번째 워밍업 라운드 시작...")
                
                for i, data in enumerate(liquidation_data):
                    liquidation_event = {
                        'timestamp': data.get('timestamp', int(time.time())),
                        'side': data.get('side', 'unknown'),
                        'qty_usd': data.get('size', 0.0) * data.get('price', 0.0)
                    }
                    
                    # 고급 청산 전략에 이벤트 전달
                    self._adv_liquidation_strategy.process_liquidation_event(liquidation_event)
                    processed_count += 1
            
            print(f"🎯 전략 워밍업 완료: {processed_count}개 이벤트 처리됨 (반복: {repeat_count}회)")
            
        except Exception as e:
            print(f"❌ 전략 워밍업 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _setup_websocket_strategies(self):
        """웹소켓 전략 설정"""
        websocket = self.core.get_websocket()
        
        # 전략 실행기를 웹소켓에 설정
        websocket.set_strategies(
            session_strategy=self._session_strategy,
            advanced_liquidation_strategy=self._adv_liquidation_strategy
        )
        
        print("✅ 웹소켓에서 직접 전략 실행하도록 설정 완료")
        print("   - 1분봉마다: 청산 전략 실행")
        print("   - 3분마다: 세션 전략 실행 (1분봉 시뮬레이션)")
    
    def _handle_liquidation_event(self, data: Dict):
        """청산 이벤트 처리"""
        try:
            self._process_advanced_liquidation_event(data)
                
        except Exception as e:
            print(f"❌ 청산 이벤트 처리 오류: {e}")
            import traceback
            traceback.print_exc()
    
    # def _handle_3m_kline_close(self, data: Dict):
    #     """3분봉 마감 이벤트 처리"""
    #     try:
    #         if self._is_or_completed(self.core.time_manager.get_current_time()):
    #             print(f"\n⏰ {data['timestamp'].strftime('%H:%M:%S')} - 3분봉 마감! 세션 전략 분석 시작")
                
    #             session_signal = self._analyze_session_strategy()
    #             if session_signal:
    #                 self._print_session_signal(session_signal, data['timestamp'])
                
    #             print(f"✅ {data['timestamp'].strftime('%H:%M')} - 세션 전략 분석 완료")
    #         else:
    #             print(f"⏰ {data['timestamp'].strftime('%H:%M:%S')} - 3분봉 마감 (OR 미완성, 세션 전략 스킵)")
                
    #     except Exception as e:
    #         print(f"❌ 3분봉 마감 이벤트 처리 오류: {e}")
    #         import traceback
    #         traceback.print_exc()
    
    # def _is_or_completed(self, now: datetime.datetime) -> bool:
    #     """세션 윈도우(2시간) 제한 + 세션 오픈 후 30분 신호 차단"""
    #     try:
    #         # 뉴욕 시장 오픈 시간 (UTC 13:30, KST 22:30)
    #         ny_open_utc = now.replace(hour=13, minute=30, second=0, microsecond=0)
            
    #         # 유럽 시장 오픈+확장 시간 (UTC 07:00, KST 16:00)
    #         eu_open_utc = now.replace(hour=7, minute=0, second=0, microsecond=0)
            
    #         # 현재 시간이 뉴욕 오픈 후 30분이 지났는지 체크 (세션 윈도우 2시간 제한)
    #         if now >= ny_open_utc:
    #             time_since_open = now - ny_open_utc
    #             if 1800 <= time_since_open.total_seconds() <= 9000:  # 30분 ~ 2시간 30분 (2시간 윈도우)
    #                 return True
            
    #         # 현재 시간이 유럽 오픈 후 30분이 지났는지 체크 (세션 윈도우 2시간 제한)
    #         if now >= eu_open_utc:
    #             time_since_open = now - eu_open_utc
    #             if 1800 <= time_since_open.total_seconds() <= 9000:  # 30분 ~ 2시간 30분 (2시간 윈도우)
    #                 return True
            
    #         return False
            
    #     except Exception as e:
    #         print(f"❌ OR 완성 체크 오류: {e}")
    #         return False
    
    """_summary_
    바이낸스 청산 이벤트 형식
    data = {'timestamp': datetime.datetime(2025, 8, 22, 1, 42, 47, 173880), 
        'symbol': 'ETHUSDT', 'side': 'BUY', 
        'quantity': 0.048, 'price': 4255.65, 'qty_usd': 204.2712, 'time': 1755794568097}
    """ 
    def _process_advanced_liquidation_event(self, data: Dict):
        """고급 청산 전략 이벤트 처리"""
        try:
            # 바이낸스 청산 데이터 형식 처리
            side = 'short' if data.get('side') == 'BUY' else 'long'

            # 타임스탬프 안전하게 변환
            timestamp = data.get('timestamp', time.time())
            timestamp = int(timestamp.timestamp())
            
            liquidation_event = {
                'timestamp': timestamp,
                'side': side,
                'qty_usd': data.get('size', 0.0)*data.get('price', 0.0)
            }
            
            # 60초 버킷에 청산 이벤트 추가
            if not hasattr(self, 'liquidation_bucket'):
                self.liquidation_bucket = []
            # bucket_start_time은 __init__에서만 설정하고 여기서는 재설정하지 않음
                
            self.liquidation_bucket.append(liquidation_event)
            
            # 고급 청산 전략에 이벤트 전달 (실시간 처리용)
            self._adv_liquidation_strategy.process_liquidation_event(liquidation_event)
            
        except Exception as e:
            print(f"❌ 고급 청산 이벤트 처리 오류: {e}")
    
    # def _analyze_session_strategy(self) -> Optional[Dict]:
    #     """세션 기반 전략 분석"""
    #     try:
    #         if not self.config.enable_session_strategy:
    #             return None
            
    #         # 3분봉 데이터 로드
    #         data_manager = get_data_manager()
    #         df_3m = data_manager.get_dataframe()
            
            
    #         if df_3m.empty:
    #             return None
            
    #         # 키 레벨 계산
    #         key_levels = self.global_manager.get_indicator('daily_levels').get_status()
            
    #         # 현재 시간 (UTC)
    #         current_time = datetime.datetime.now(datetime.timezone.utc)
            
    #         # 세션 전략 분석
    #         from signals.session_based_strategy import SessionBasedStrategy, SessionConfig
    #         session_config = SessionConfig()
    #         session_strategy = SessionBasedStrategy(session_config)
            
    #         return session_strategy.analyze_session_strategy(
    #             df_3m, key_levels, current_time
    #         )
            
    #     except Exception as e:
    #         print(f"❌ 세션 전략 분석 오류: {e}")
    #         return None
    
    # def _calculate_session_key_levels(self, df) -> Dict[str, float]:
    #     """세션 전략용 키 레벨 계산"""
    #     try:
    #         if df.empty:
    #             return {}
            
    #         # 전일 고가/저가/종가
    #         daily_data = df.resample('D').agg({
    #             'high': 'max',
    #             'low': 'min',
    #             'close': 'last'
    #         }).dropna()
            
    #         if len(daily_data) < 2:
    #             return {}
            
    #         prev_day = daily_data.iloc[-2]
            
    #         # 최근 스윙 고점/저점 (20봉 기준)
    #         lookback = min(20, len(df))
    #         recent_data = df.tail(lookback)
            
    #         return {
    #             'prev_day_high': prev_day['high'],
    #             'prev_day_low': prev_day['low'],
    #             'prev_day_close': prev_day['close'],
    #             'last_swing_high': recent_data['high'].max(),
    #             'last_swing_low': recent_data['low'].min()
    #         }
            
    #     except Exception as e:
    #         print(f"❌ 세션 키 레벨 계산 오류: {e}")
    #         return {}
    
    # def _analyze_advanced_liquidation_strategy(self) -> Optional[Dict]:
    #     """고급 청산 전략 분석"""
    #     try:
    #         if not self._adv_liquidation_strategy:
    #             print("❌ 고급 청산 전략이 초기화되지 않음")
    #             return None
            
    #         # 현재 가격 데이터 가져오기
    #         websocket = self.core.get_websocket()
    #         if not websocket.price_history:
    #             print("❌ 가격 히스토리가 비어있음 - 1분봉 데이터 대기 중...")
    #             return None
            
    #         current_price = websocket.price_history[-1]['price']
    #         print(f"💰 현재 가격: {current_price}")
            
    #         # 60초 버킷 데이터로 분석
    #         if hasattr(self, 'liquidation_bucket') and self.liquidation_bucket:
    #             print(f"📦 버킷 데이터 {len(self.liquidation_bucket)}개로 분석 시작...")
    #             # 버킷 데이터를 전략에 전달하여 분석
    #             signal = self._adv_liquidation_strategy.analyze_bucket_liquidations(
    #                 self.liquidation_bucket, current_price
    #             )
    #             print(f"🎯 전략 분석 결과: {signal}")
    #             return signal
    #         else:
    #             print("❌ 청산 버킷이 비어있음")
    #             return None
            
    #     except Exception as e:
    #         print(f"❌ 고급 청산 전략 분석 오류: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None
    
    # def _calculate_opening_range(self, df) -> Dict[str, float]:
    #     """오프닝 레인지 계산"""
    #     try:
    #         if df.empty:
    #             return {}
            
    #         or_minutes = 15
    #         if len(df) < or_minutes:
    #             return {}
            
    #         or_data = df.head(or_minutes)
            
    #         return {
    #             'high': or_data['high'].max(),
    #             'low': or_data['low'].min(),
    #             'center': (or_data['high'].max() + or_data['low'].min()) / 2,
    #             'range': or_data['high'].max() - or_data['low'].min()
    #         }
            
    #     except Exception as e:
    #         print(f"❌ 오프닝 레인지 계산 오류: {e}")
    #         return {}
    
    # def _calculate_vwap_and_std(self, df) -> tuple[float, float]:
    #     """VWAP 및 표준편차 계산"""
    #     try:
    #         if df.empty:
    #             return 0.0, 0.0
            
    #         # 가격과 거래량으로 VWAP 계산
    #         vwap = sum(df['close'] * df['volume']) / sum(df['volume']) if sum(df['volume']) > 0 else 0
            
    #         # 표준편차 계산
    #         mean_price = df['close'].mean()
    #         std = (sum((df['close'] - mean_price) ** 2) / len(df)) ** 0.5
            
    #         return vwap, std
            
    #     except Exception as e:
    #         print(f"❌ VWAP 및 표준편차 계산 오류: {e}")
    #         return 0.0, 0.0
    
    
    # def _print_advanced_liquidation_signal(self, signal: Dict, now: datetime.datetime):
    #     """고급 청산 신호 출력"""
    #     try:
    #         if signal is None:
    #             signal = {}
            
    #         action = signal.get('action', 'NEUTRAL')
    #         playbook = signal.get('playbook', 'NO_SIGNAL')
    #         tier = signal.get('tier', 'NEUTRAL')
    #         total_score = signal.get('total_score', 0.000)
    #         reason = signal.get('reason', '모든 전략에서 신호 없음')
            
    #         print(f"\n{'='*50}")
    #         print(f"⚡ 고급 청산 전략 신호 감지!")
    #         print(f"{'='*50}")
    #         print(f"⏰ 시간: {now.strftime('%H:%M:%S')}")
    #         print(f"🎯 액션: {action}")
    #         print(f"📚 플레이북: {playbook}")
    #         print(f"🏆 등급: {tier}")
    #         print(f"📊 총점: {total_score:.3f}")
    #         print(f"📝 이유: {reason}")
    #         print(f"{'='*50}\n")
            
    #     except Exception as e:
    #         print(f"❌ 고급 청산 신호 출력 오류: {e}")
    
    def start(self):
        """트레이더 시작"""
        self._print_startup_info()
        
        # 웹소켓 전략 설정 (전략 초기화 완료 후)
        self._setup_websocket_strategies()
        
        self.running = True
        
        # 웹소켓 백그라운드 시작
        self.core.start_websocket()
        
        # 메인 루프
        self._run_main_loop()
    
    def _print_startup_info(self):
        """시작 정보 출력"""
        print(f"🚀 {self.config.symbol} 통합 스마트 트레이더 시작!")
        
        # 현재 세션 정보 출력
        try:
            from utils.time_manager import get_time_manager
            time_manager = get_time_manager()
            
            # 현재 세션 상태 확인
            session_config = time_manager.get_indicator_mode_config()
            
            if session_config['use_session_mode']:
                session_name = session_config.get('session_name', 'UNKNOWN')
                session_start = session_config.get('session_start_time')
                elapsed_minutes = session_config.get('elapsed_minutes', 0)
                session_status = session_config.get('session_status', 'UNKNOWN')
                
                print(f"📊 현재 세션: {session_name}")
                print(f"🕐 세션 시작: {session_start}")
                print(f"⏱️ 경과 시간: {elapsed_minutes:.1f}분")
                print(f"📈 세션 상태: {session_status}")
            else:
                print(f"📊 현재 세션: 세션 외 시간 (룩백 모드)")
                
        except Exception as e:
            print(f"⚠️ 세션 정보 출력 오류: {e}")
        
        print(f"📊 세션 전략: {'활성' if self.config.enable_session_strategy else '비활성'}")
        print(f"⏰ 세션 전략: 1분봉 기반 3분마다 실행 (OR 30분 완성 후)")
        print(f"⚡ 청산 전략: 1분봉마다 실행")
        print("=" * 60)
        print("💡 웹소켓에서 직접 전략 실행 - 메인 루프 단순화됨")
        print("⚠️  첫 1분봉 데이터 수집까지 대기 중... (약 1분)")
        print("=" * 60)
    
    def _run_main_loop(self):
        """메인 실행 루프 - 단순화됨"""
        try:
            print("🔄 메인 루프 시작 - 웹소켓에서 전략 실행")
            while self.running:
                time.sleep(1)  # 1초마다 상태 체크만
                
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
