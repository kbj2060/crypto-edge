from dataclasses import dataclass, field
from typing import List

@dataclass
class IntegratedConfig:
    """통합 설정"""
    
    def __init__(self):
        # 기본 설정
        self.symbol: str = "ETHUSDT"
        self.enable_hybrid_strategy: bool = True      # 5분봉 하이브리드 전략
        self.enable_liquidation_strategy: bool = True # 실시간 청산 전략
        
        # 실시간/주기 모드 선택
        self.use_periodic_hybrid: bool = False        # True면 5분 주기 분석 스레드 사용, False면 실시간만 사용
        
        # 하이브리드 전략 설정
        self.hybrid_interval_15m: str = "15m"
        self.hybrid_interval_5m: str = "5m"
        self.hybrid_limit_15m: int = 200
        self.hybrid_limit_5m: int = 300
        self.hybrid_min_confidence: float = 0.4     # 0.50에서 0.25로 낮춤 (더 민감하게)
        self.hybrid_trend_weight: float = 0.4
        self.hybrid_entry_weight: float = 0.6
        
        # VPVR 설정
        self.liquidation_vpvr_bins: int = 50
        self.liquidation_vpvr_lookback: int = 200
        
        # 청산 전략 설정
        self.liquidation_min_count: int = 1  # 2에서 1로 낮춤 (더 예민하게)
        self.liquidation_min_value: float = 25000.0  # 50000에서 25000으로 낮춤 (더 예민하게)
        self.liquidation_buy_ratio: float = 0.5  # 0.6에서 0.5로 낮춤 (더 예민하게)
        self.liquidation_sell_ratio: float = 0.5  # 0.6에서 0.5로 낮춤 (더 예민하게)
        self.liquidation_volume_threshold: float = 1.2  # 1.5에서 1.2로 낮춤 (더 예민하게)
        self.liquidation_window_minutes: int = 10  # 2에서 10으로 늘림 (더 많은 데이터 유지)
        
        # 청산 예측 전략 설정
        self.enable_liquidation_prediction: bool = True
        self.prediction_price_bin_size: float = 0.0002  # 0.0005에서 0.0002로 낮춤 (더 빠른 예측)
        self.prediction_min_density: int = 1  # 2에서 1로 낮춤 (더 빠른 예측)
        self.prediction_cascade_threshold: int = 2  # 3에서 2로 낮춤 (더 빠른 예측)
        self.prediction_min_confidence: float = 0.3  # 0.4에서 0.3으로 낮춤 (더 빠른 예측)
        self.prediction_max_horizon_hours: int = 1  # 2에서 1로 낮춤 (더 빠른 예측)
        
        # 타이밍 전략 설정
        self.timing_entry_confidence_min: float = 0.12  # 0.15에서 0.12로 낮춤 (스캘핑용)
        self.timing_entry_rr_min: float = 0.10  # 0.12에서 0.10으로 낮춤 (스캘핑용)
        self.timing_entry_score_threshold: float = 0.25  # 0.30에서 0.25로 낮춤 (스캘핑용)
        self.timing_max_hold_time_hours: int = 4  # 24에서 4로 낮춤 (스캘핑용)
        self.timing_trailing_stop_atr: float = 1.5  # 2.0에서 1.5로 낮춤 (스캘핑용)
        
        # 통합 신호 설정
        self.enable_synergy_signals: bool = True      # 시너지 신호 활성화

        # 세션 기반 전략 설정
        self.enable_session_strategy = True
        self.session_timeframe = "1m"
        # 뉴욕 시장 오픈 시간 기준으로 고정 (한국 시간 22:30)
        self.session_vwap_start_utc = "13:30 UTC"  # NY Open (KST 22:30, UTC 17:30)
        self.session_or_minutes = 15
        
        # 플레이북 A: 오프닝 드라이브 풀백
        self.session_min_drive_return_R = 1.2
        self.session_pullback_depth_atr = (0.6, 1.4)
        self.session_trigger_type = "close_reject"
        self.session_stop_atr_mult = 1.1
        self.session_tp1_R = 1.5
        self.session_tp2_to_level = "OR_ext|PrevHigh|VWAP"
        self.session_partial_out = 0.5
        self.session_max_hold_min = 60
        self.session_max_slippage_pct = 0.02
        
        # 플레이북 B: 유동성 스윕 & 리클레임
        self.session_sweep_depth_atr_min = 0.3
        self.session_reclaim_close_rule = "close_above_level"

        self.session_stop_buffer_atr = 0.6
        self.session_tp1_to = "VWAP"
        self.session_tp2_to = "opposite_range_edge"
        
        # 플레이북 C: VWAP 리버전 페이드
        self.session_sd_k_enter = 2.0
        self.session_sd_k_reenter = 1.5
        self.session_stop_outside_sd_k = 2.5
        self.session_tp1_to = "VWAP"
        self.session_tp2_to_band = 0.5
        self.session_trend_filter_slope = 0.0
        
        # 고급 청산 전략 설정
        self.enable_advanced_liquidation = True
        self.adv_liq_symbol = "ETHUSDT"  # 심볼 추가
        self.adv_liq_bin_sec = 1
        self.adv_liq_agg_window_sec = 30
        self.adv_liq_background_window_min = 60
        
        # 스파이크 판정 설정
        self.adv_liq_z_spike = 2.0  # 3.0에서 2.0으로 수정 (더 민감하게)
        self.adv_liq_z_strong = 3.0  # 4.0에서 3.0으로 수정 (더 민감하게)
        self.adv_liq_lpi_bias = 0.3  # 0.4에서 0.3으로 수정 (더 민감하게)
        
        # 캐스케이드 설정
        self.adv_liq_cascade_seconds = 10
        self.adv_liq_cascade_count = 3
        self.adv_liq_cascade_z = 3.0
        
        # 쿨다운 설정
        self.adv_liq_cooldown_after_strong_sec = 10  # 30에서 10으로 수정 (더 빠른 재시작)
        
        # 리스크 설정
        self.adv_liq_risk_pct = 0.4
        self.adv_liq_slippage_max_pct = 0.02
        
        # 레벨 설정
        self.adv_liq_or_minutes = 15
        self.adv_liq_atr_len = 14
        self.adv_liq_vwap_sd_enter = 2.0
        self.adv_liq_vwap_sd_stop = 2.5
        
        # 전략 A: 스윕&리클레임
        self.adv_liq_sweep_buffer_atr = 0.3
        self.adv_liq_tp1_R = 1.2
        self.adv_liq_tp2 = "VWAP_or_range_edge"
        
        # 전략 B: 스퀴즈 추세지속
        self.adv_liq_retest_atr_tol = 0.4
        self.adv_liq_tp1_R = 1.5
        self.adv_liq_or_extension = True
        
        # 전략 C: 과열-소멸 페이드
        self.adv_liq_post_spike_decay_ratio = 0.8
        self.adv_liq_stop_atr = 0.35
        self.adv_liq_tp2_sigma = 0.5
