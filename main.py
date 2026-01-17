#!/usr/bin/env python3
"""
통합 스마트 자동 트레이더 (리팩토링 버전)
실시간 청산 전략 + 세션 기반 전략 + 고급 청산 전략을 활용합니다.
"""

from datetime import datetime, timezone
import time
import os
from pathlib import Path

# .env 파일 로드 (가장 먼저 실행)
try:
    from dotenv import load_dotenv
    # 프로젝트 루트 디렉토리에서 .env 파일 찾기
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ .env 파일 로드 완료: {env_path}")
    else:
        print(f"⚠️ .env 파일을 찾을 수 없습니다: {env_path}")
        print("   환경 변수를 직접 설정하거나 .env 파일을 생성하세요.")
except ImportError:
    print("⚠️ python-dotenv가 설치되지 않았습니다. pip install python-dotenv로 설치하세요.")
    print("   환경 변수를 직접 설정하거나 .env 파일을 사용할 수 없습니다.")

from config.integrated_config import IntegratedConfig
from managers.bucket_aggregator import BucketAggregator
from indicators.global_indicators import get_global_indicator_manager

class IntegratedSmartTrader:
    """통합 스마트 자동 트레이더 (리팩토링 버전)"""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self.running = False
        
        # 핵심 컴포넌트 초기화
        self.global_manager = get_global_indicator_manager(target_time=datetime.now(timezone.utc))
        self.bucket_aggregator = None
        
        # 청산 버킷 관리 (60초 단위)
        self.liquidation_bucket = []
        
        self._init_data_manager()
        self._init_global_indicators()
        self._init_bucket_aggregator()
        
        # 전략 실행기 초기화 (내부에서 모든 전략 자동 초기화)
        self._init_strategy_executor()

    #     self.warmup_strategies()


    # def warmup_strategies(self):
    #     """전략 웜업"""

    def _init_data_manager(self):
        """DataManager 우선 초기화 (데이터 준비)"""
        try:
            print("\n🚀 1단계: DataManager 우선 초기화 시작...")
            
            from managers.data_manager import get_data_manager
            
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

    def _init_strategy_executor(self):
        """전략 실행기 초기화"""
        try:
            from managers.strategy_executor import StrategyExecutor
            from core.trader_core import TraderCore
            
            # 전략 실행기 인스턴스 생성 (내부에서 모든 전략 자동 초기화)
            self.strategy_executor = StrategyExecutor()
            
            # TraderCore 초기화 (strategy_executor와 함께)
            self.core = TraderCore(self.config, self.strategy_executor)
            
        except Exception as e:
            print(f"❌ 전략 실행기 초기화 오류: {e}")
            import traceback
            traceback.print_exc()
        

    def start(self):
        """트레이더 시작"""
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
                time.sleep(0.5)  # 1초마다 상태 체크만
                
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중지됨")
        finally:
            self.stop()
    
    def get_strategy_executor(self):
        """전략 실행기 반환"""
        return self.strategy_executor
    
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
