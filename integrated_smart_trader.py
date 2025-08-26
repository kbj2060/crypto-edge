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
            print("\n🚀 1단계: DataManager 우선 초기화 시작...")
            
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
                
            from signals.session_based_strategy import SessionBasedStrategy, SessionConfig
            
            session_config = SessionConfig()
            
            self._session_strategy = SessionBasedStrategy(session_config)
                            
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
                # 타임스탬프 처리 (UTC datetime으로 변환)
                timestamp = item.get('timestamp')
                
                # datetime 문자열을 UTC datetime으로 변환
                # '2025-08-21 16:06:51.478000' 형식 파싱
                dt = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f')
                # UTC timezone 설정
                utc_dt = dt.replace(tzinfo=datetime.timezone.utc)
                
                # 변환된 데이터 생성
                converted_data = {
                    'timestamp': utc_dt,
                    'symbol': item.get('symbol', self.config.symbol),
                    'side': item.get('side', 'unknown'),
                    'size': item.get('size', 0.0),
                    'price': item.get('price', 0.0)
                }
                
                liquidation_data.append(converted_data)
            
            # 데이터 품질 검증 및 개선
            long_count = sum(1 for item in liquidation_data if item.get('side') == 'SELL')
            short_count = sum(1 for item in liquidation_data if item.get('side') == 'BUY')
            
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
                    # timestamp를 안전하게 처리 (TimeManager 사용)
                    utc_timestamp = self.core.time_manager.get_timestamp_datetime(data.get('timestamp'))
                    
                    liquidation_event = {
                        'timestamp': utc_timestamp,
                        'side': data.get('side'),
                        'qty_usd': data.get('size') * data.get('price')
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
        
    
    def start(self):
        """트레이더 시작"""
        # 웹소켓 전략 설정 (전략 초기화 완료 후)
        self._setup_websocket_strategies()
        
        self.running = True
        
        # 웹소켓 백그라운드 시작
        self.core.start_websocket()
        
        # 메인 루프
        self._run_main_loop()
    
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
