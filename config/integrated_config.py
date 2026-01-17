from dataclasses import dataclass

@dataclass
class IntegratedConfig:
    """통합 설정"""
    
    def __init__(self):
        # 기본 설정
        self.symbol: str = "ETHUSDT"
        
        # 거래 실행 설정
        self.enable_trading: bool = True   # 실제 거래 활성화 여부
        self.simulation_mode: bool = False  # 시뮬레이션 모드 (False: API 실제 호출)
        self.use_demo: bool = True         # Demo Trading 사용 여부
        
        # 세션 기반 전략 설정
        self.enable_session_strategy = True
        self.session_timeframe = "3m"
        
        # 고급 청산 전략 설정
        self.enable_advanced_liquidation = True
        self.adv_liq_symbol = "ETHUSDT"
        
        # 외부 서버 설정
        self.external_server_url = "http://158.180.82.65"  # 외부 서버 URL
        self.external_api_key = None  # 외부 서버 API 키 (필요시 설정)
        self.initial_data_hours = 24  # 초기 데이터 로딩 시 몇 시간 전까지 가져올지

        self.agent_start_idx = 500