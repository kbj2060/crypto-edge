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
from data.bucket_aggregator import BucketAggregator
from data.data_manager import get_data_manager
from indicators.global_indicators import get_global_indicator_manager
from signals.liquidation_strategies_lite import SqueezeMomentumStrategy, MomentumConfig, FadeReentryStrategy, FadeConfig
from signals.session_or_lite import SessionORLite, SessionORLiteCfg
from signals.vpvr_golden_strategy import LVNGoldenPocket

class IntegratedSmartTrader:
    """통합 스마트 자동 트레이더 (리팩토링 버전)"""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self.running = False
        
        # 핵심 컴포넌트 초기화
        self.core = TraderCore(config)
        self.global_manager = get_global_indicator_manager()
        self.bucket_aggregator = None
        
        # 청산 버킷 관리 (60초 단위)
        self.liquidation_bucket = []
        
        # 🚀 1단계: DataManager 우선 초기화 (데이터 준비)
        self._init_data_manager()
        self._init_global_indicators()

        self._init_bucket_aggregator()

        # 🚀 3단계: 고급 청산 전략 초기화
        # self._init_advanced_liquidation_strategy()
        
        # 🚀 4단계: 세션 기반 전략 초기화
        self._init_vpvr_golden_strategy()
        self._init_session_strategy()
        self._init_squeeze_momentum_strategy()
        self._init_fade_reentry_strategy()
    
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

    

    def _init_bucket_aggregator(self):
        """버킷 집계기 초기화"""
        self.bucket_aggregator = BucketAggregator()
        self.liquidation_bucket = self.bucket_aggregator.load_external_data()

    def _init_vpvr_golden_strategy(self):
        """VPVR 골든 포켓 전략 초기화"""
        try:
            self._vpvr_golden_strategy = LVNGoldenPocket()

        except Exception as e:
            print(f"❌ VPVR 골든 포켓 전략 초기화 오류: {e}")
            import traceback
            traceback.print_exc()
            self._vpvr_golden_strategy = None

    def _init_squeeze_momentum_strategy(self):
        """스퀴즈 모멘텀 전략 초기화"""
        try:
            squeeze_config = MomentumConfig()
            self._squeeze_momentum_strategy = SqueezeMomentumStrategy(squeeze_config)
            self._squeeze_momentum_strategy.warmup(self.liquidation_bucket)

        except Exception as e:
            print(f"❌ 스퀴즈 모멘텀 전략 초기화 오류: {e}")
            import traceback
            traceback.print_exc()
            self._squeeze_momentum_strategy = None
    
    def _init_fade_reentry_strategy(self):
        """페이드 리입 전략 초기화"""
        try:
            fade_config = FadeConfig()
            self._fade_reentry_strategy = FadeReentryStrategy(fade_config)
            self._fade_reentry_strategy.warmup(self.liquidation_bucket)

        except Exception as e:
            print(f"❌ 페이드 리입 전략 초기화 오류: {e}")
            import traceback
            traceback.print_exc()
            self._fade_reentry_strategy = None
            
    def _init_session_strategy(self):
        """세션 기반 전략 초기화"""
        try:
            session_config = SessionORLiteCfg()
            self._session_strategy = SessionORLite(session_config)

        except Exception as e:
            print(f"❌ 세션 기반 전략 초기화 오류: {e}")
            import traceback
            traceback.print_exc()
            self._session_strategy = None

    def _setup_websocket_strategies(self):
        """웹소켓 전략 설정 - 메인 컨트롤러에서 전략 인스턴스 전달"""
        try:
            websocket = self.core.get_websocket()
            
            # 전략 실행기를 웹소켓에 설정 (None인 전략은 제외)
            strategies = {
                'session_strategy': self._session_strategy,
                'squeeze_momentum_strategy': self._squeeze_momentum_strategy,
                'fade_reentry_strategy': self._fade_reentry_strategy,
                'vpvr_golden_strategy': self._vpvr_golden_strategy
            }
            
            # None이 아닌 전략만 필터링하여 전달
            active_strategies = {k: v for k, v in strategies.items() if v is not None}
            
            if active_strategies:
                websocket.set_strategies(**active_strategies)
                print(f"🎯 웹소켓에 {len(active_strategies)}개 전략 설정 완료: {list(active_strategies.keys())}")
            else:
                print("⚠️ 활성화된 전략이 없습니다")
                
        except Exception as e:
            print(f"❌ 웹소켓 전략 설정 오류: {e}")
            import traceback
            traceback.print_exc()
        
    
    def start(self):
        """트레이더 시작"""
        # 웹소켓 전략 설정 (전략 초기화 완료 후)
        self._setup_websocket_strategies()
        
        self.running = True
        
        # 웹소켓 백그라운드 시작
        self.core.start_websocket()
        # self.core.get_websocket().add_callback('kline_1m', self.process_kline_1m)
        
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
