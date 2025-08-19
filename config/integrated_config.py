from dataclasses import dataclass, field
from typing import List

@dataclass
class IntegratedConfig:
    """통합 전략 설정"""
    
    # 기본 설정
    symbol: str = "ETHUSDT"
    enable_hybrid_strategy: bool = True      # 5분봉 하이브리드 전략
    enable_liquidation_strategy: bool = True # 실시간 청산 전략
    
    # 실시간/주기 모드 선택
    use_periodic_hybrid: bool = False        # True면 5분 주기 분석 스레드 사용, False면 실시간만 사용
    
    # 하이브리드 전략 설정
    hybrid_interval_15m: str = "15m"
    hybrid_interval_5m: str = "5m"
    hybrid_limit_15m: int = 200
    hybrid_limit_5m: int = 300
    hybrid_min_confidence: float = 0.50  # 0.12에서 0.50으로 높임 (더 신중한 진입)
    hybrid_trend_weight: float = 0.4
    hybrid_entry_weight: float = 0.6
    
    # VPVR 설정
    liquidation_vpvr_bins: int = 50
    liquidation_vpvr_lookback: int = 200
    
    # 청산 전략 설정
    liquidation_min_count: int = 1  # 2에서 1로 낮춤 (더 예민하게)
    liquidation_min_value: float = 25000.0  # 50000에서 25000으로 낮춤 (더 예민하게)
    liquidation_buy_ratio: float = 0.5  # 0.6에서 0.5로 낮춤 (더 예민하게)
    liquidation_sell_ratio: float = 0.5  # 0.6에서 0.5로 낮춤 (더 예민하게)
    liquidation_volume_threshold: float = 1.2  # 1.5에서 1.2로 낮춤 (더 예민하게)
    liquidation_window_minutes: int = 2  # 3에서 2로 낮춤 (더 예민하게)
    
    # 청산 예측 전략 설정
    enable_liquidation_prediction: bool = True
    prediction_price_bin_size: float = 0.0002  # 0.0005에서 0.0002로 낮춤 (더 빠른 예측)
    prediction_min_density: int = 1  # 2에서 1로 낮춤 (더 빠른 예측)
    prediction_cascade_threshold: int = 2  # 3에서 2로 낮춤 (더 빠른 예측)
    prediction_min_confidence: float = 0.3  # 0.4에서 0.3으로 낮춤 (더 빠른 예측)
    prediction_max_horizon_hours: int = 1  # 2에서 1로 낮춤 (더 빠른 예측)
    
    # 타이밍 전략 설정
    timing_entry_confidence_min: float = 0.12  # 0.15에서 0.12로 낮춤 (스캘핑용)
    timing_entry_rr_min: float = 0.10  # 0.12에서 0.10으로 낮춤 (스캘핑용)
    timing_entry_score_threshold: float = 0.25  # 0.30에서 0.25로 낮춤 (스캘핑용)
    timing_max_hold_time_hours: int = 4  # 24에서 4로 낮춤 (스캘핑용)
    timing_trailing_stop_atr: float = 1.5  # 2.0에서 1.5로 낮춤 (스캘핑용)
    
    # 통합 신호 설정
    enable_synergy_signals: bool = True      # 시너지 신호 활성화
