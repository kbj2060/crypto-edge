from dataclasses import dataclass

@dataclass
class RunConfig:
    symbol: str = "ETHUSDT"
    interval: str = "15m"
    limit: int = 1500
    vpvr_bins: int = 50
    vpvr_lookback: int = 300

    # 신호 컨펌 수
    min_confirms_long: int = 3
    min_confirms_short: int = 3

    # 필터(기본: 켜짐)
    require_bb_expand: bool = True
    require_macd_slope: bool = True

    # VPVR 헤드룸(0.3% -> 0.2%로 낮춰서 덜 막히게)
    min_vpvr_headroom: float = 0.002

    # ATR
    atr_len: int = 14
    atr_stop_mult_long: float = 1.0
    atr_stop_mult_short: float = 1.0
    atr_tp1_mult: float = 1.0
    atr_tp2_mult: float = 2.0

    # 이하 새 옵션
    relaxed: bool = False      # 완화 모드 (조건 느슨하게)
    debug: bool = False        # 디버그 로그 출력
