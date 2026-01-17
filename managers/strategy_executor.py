from typing import Dict, Any
from managers.data_manager import get_data_manager
from signals.ema_confluence_15m import EMAConfluence
from signals.funding_rate_strategy import FundingRateStrategy
from signals.multitimeframe_strategy import MultiTimeframeStrategy
from signals.oi_delta_strategy import OIDeltaStrategy
from signals.liquidity_grab_strategy import LiquidityGrabStrategy
from signals.macd_histogram_strategy import MACDHistogramStrategy
from signals.support_resistance_15m import SupportResistance
from signals.vpvr_micro import VPVRConfig
from signals.vwap_pinball_strategy import VWAPPinballCfg
from managers.time_manager import get_time_manager

# 전략 imports
from signals.vpvr_golden_strategy import LVNGoldenPocket
from signals.vpvr_micro import VPVRMicro, VPVRConfig
from signals.bollinger_squeeze_strategy import BollingerSqueezeStrategy
from signals.rsi_divergence import RSIDivergence
from signals.ichimoku import Ichimoku
from signals.htf_trend import HTFTrend
from signals.orderflow_cvd import OrderflowCVD
from signals.vwap_pinball_strategy import VWAPPinballStrategy
from signals.vol_spike_3m import VolSpike
from signals.zscore_mean_reversion import ZScoreMeanReversion


class StrategyExecutor:
    """전략 실행을 담당하는 클래스"""
    
    def __init__(self):
        self.data_manager = get_data_manager()
        self.time_manager = get_time_manager()
        self.signals = {}
        
        # 전략 인스턴스들 직접 초기화
        # VPVR 골든 포켓 전략
        self.vpvr_golden_strategy = LVNGoldenPocket()
        
        # VPVR 마이크로 전략
        self.vpvr_micro_strategy = VPVRMicro()
        
        # 볼린저 스퀴즈 전략
        self.bollinger_squeeze_strategy = BollingerSqueezeStrategy()
                
        # RSI 다이버전스 전략
        self.rsi_divergence_strategy = RSIDivergence()
        
        # 일목균형 전략
        self.ichimoku_strategy = Ichimoku()
        
        # HTF 트렌드 전략
        self.htf_trend_strategy = HTFTrend()
        
        # 오더플로우 CVD 전략
        self.orderflow_cvd_strategy = OrderflowCVD()
        
        # VWAP 피니언 전략
        self.vwap_pinball_strategy = VWAPPinballStrategy()
        
        # 볼륨 스파이크 전략
        self.vol_spike_strategy = VolSpike()
        
        # Z-Score 평균 회귀 전략
        self.zscore_mean_reversion_strategy = ZScoreMeanReversion()
        
        self.funding_rate_strategy = FundingRateStrategy()

        self.oiDelta_strategy = OIDeltaStrategy()

        self.macd_histogram_strategy = MACDHistogramStrategy()

        self.liquidity_grab_strategy = LiquidityGrabStrategy()

        self.multitimeframe_strategy = MultiTimeframeStrategy()

        self.support_resistance_strategy = SupportResistance()
        self.ema_confluence_strategy = EMAConfluence()
        print("🎯 모든 전략 초기화 완료")

    def execute_all_strategies(self):
        """모든 전략 실행 (최적화: 시간 계산 캐싱)"""
        self.signals = {}  # 시그널 초기화
        
        # 시간을 한 번만 계산하여 클래스 변수에 저장 (성능 최적화 - 각 전략마다 호출하는 것 방지)
        self._cached_timestamp = self.time_manager.get_current_time()
        
        # 전략 실행
        self._execute_vpvr_golden_strategy()
        self._execute_bollinger_squeeze_strategy()
        self._execute_orderflow_cvd_strategy()
        self._execute_ichimoku_strategy()
        self._execute_vwap_pinball_strategy()
        self._execute_vol_spike_strategy()
        self._execute_liquidity_grab_strategy()
        self._execute_vpvr_micro_strategy()
        self._execute_zscore_mean_reversion_strategy()
        # 15분봉 전략
        self._execute_htf_trend_strategy()
        self._execute_oiDelta_strategy()
        self._execute_funding_rate_strategy()
        self._execute_multitimeframe_strategy()
        self._execute_support_resistance_strategy()
        self._execute_ema_confluence_strategy()
        
        # 캐시 정리
        delattr(self, '_cached_timestamp')

    def _execute_support_resistance_strategy(self):
        """Support Resistance 전략 실행"""
        if not self.support_resistance_strategy:
            return
        timestamp = getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
        result = self.support_resistance_strategy.on_kline_close_15m()
        if result:
            self.signals['SUPPORT_RESISTANCE'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': timestamp
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['SUPPORT_RESISTANCE'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': timestamp
            }

    def _execute_ema_confluence_strategy(self):
        """EMA Confluence 전략 실행"""
        if not self.ema_confluence_strategy:
            return
        timestamp = getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
        result = self.ema_confluence_strategy.on_kline_close_15m()
        if result:
            self.signals['EMA_CONFLUENCE'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': timestamp
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['EMA_CONFLUENCE'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': timestamp
            }

    def _execute_oliverkeel_strategy(self):
        """Oliver Keel 전략 실행"""
        if not self.oliverkeel_strategy:
            return
        result = self.oliverkeel_strategy.on_kline_close_15m()
        if result:
            self.signals['OLIVER_KEEL'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['OLIVER_KEEL'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_multitimeframe_strategy(self):
        """Multi-Timeframe 전략 실행"""
        if not self.multitimeframe_strategy:
            return
        result = self.multitimeframe_strategy.on_kline_close_3m()
        if result:
            self.signals['MULTI_TIMEFRAME'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['MULTI_TIMEFRAME'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_liquidity_grab_strategy(self):
        """Liquidity Grab 전략 실행"""
        if not self.liquidity_grab_strategy:
            return
        result = self.liquidity_grab_strategy.on_kline_close_3m()
        if result:
            self.signals['LIQUIDITY_GRAB'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['LIQUIDITY_GRAB'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_funding_rate_strategy(self):
        """Funding Rate 전략 실행"""
        if not self.funding_rate_strategy:
            return
        result = self.funding_rate_strategy.on_kline_close_3m()
        if result:
            self.signals['FUNDING_RATE'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['FUNDING_RATE'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_macd_histogram_strategy(self):
        """MACD Histogram 전략 실행"""
        if not self.macd_histogram_strategy:
            return
        result = self.macd_histogram_strategy.on_kline_close_3m()
        if result:
            self.signals['MACD_HISTOGRAM'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['MACD_HISTOGRAM'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_oiDelta_strategy(self):
        """IO Delta 전략 실행"""
        if not self.oiDelta_strategy:
            return
        result = self.oiDelta_strategy.on_kline_close_3m()
        if result:
            self.signals['OI_DELTA'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['OI_DELTA'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_htf_trend_strategy(self):
        """HTF Trend 15m 전략 실행"""
        if not self.htf_trend_strategy:
            return
        df_15m = self.data_manager.get_latest_data_15m(count=300)
        df_1h = self.data_manager.get_latest_data_1h(count=300)
        result = self.htf_trend_strategy.on_kline_close_15m(df_15m=df_15m, df_1h=df_1h)
        
        if result:
            self.signals['HTF_TREND'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['HTF_TREND'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_vpvr_micro_strategy(self):
        """VPVR Micro 전략 실행"""
        if not self.vpvr_micro_strategy:
            return
        
        config = VPVRConfig()
        df_3m = self.data_manager.get_latest_data(count=config.lookback_bars)
        result = self.vpvr_micro_strategy.on_kline_close_3m(df_3m)
        
        if result and result.get('action') != 'HOLD':
            self.signals['VPVR_MICRO'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'entry': result.get('entry'),
                'stop': result.get('stop'),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['VPVR_MICRO'] = {
                'action': 'HOLD',
                'score': 0,
                'entry': 0,
                'stop': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_zscore_mean_reversion_strategy(self):
        """ZScore Mean Reversion 전략 실행"""
        if not self.zscore_mean_reversion_strategy:
            return

        result = self.zscore_mean_reversion_strategy.on_kline_close_3m()
        
        if result:
            self.signals['ZSCORE_MEAN_REVERSION'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['ZSCORE_MEAN_REVERSION'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_vpvr_golden_strategy(self):
        """VPVR 골든 포켓 전략 실행"""
        if not self.vpvr_golden_strategy:
            return
        
        config = self.vpvr_golden_strategy.VPVRConfig()
        df_15m = self.data_manager.get_latest_data_15m(count=config.lookback_bars + 5)
        sig = self.vpvr_golden_strategy.evaluate(df_15m)
        
        if sig:
            self.signals['VPVR'] = {
                'action': sig.get('action', 'UNKNOWN'),
                'score': sig.get('score', 0), 
                'confidence': sig.get('confidence', 'LOW'),
                'entry': sig.get('entry', 0),
                'stop': sig.get('stop', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['VPVR'] = {
                'action': 'HOLD',
                'score': 0,
                'confidence': 'LOW',
                'entry': 0,
                'stop': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_bollinger_squeeze_strategy(self):
        """볼린저 스퀴즈 전략 실행"""
        if not self.bollinger_squeeze_strategy:
            return
        
        result = self.bollinger_squeeze_strategy.evaluate()

        if result:
            self.signals['BOLLINGER_SQUEEZE'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'entry': result.get('entry', 0),
                'stop': result.get('stop', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['BOLLINGER_SQUEEZE'] = {
                'action': 'HOLD',
                'score': 0,
                'entry': 0,
                'stop': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_orderflow_cvd_strategy(self):
        """체결 불균형 근사 전략 실행"""
        if not self.orderflow_cvd_strategy:
            return
        
        result = self.orderflow_cvd_strategy.on_kline_close_3m()
        if result:
            self.signals['ORDERFLOW_CVD'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['ORDERFLOW_CVD'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_rsi_divergence_strategy(self):
        """HTF RSI Divergence 전략 실행"""
        if not self.rsi_divergence_strategy:
            return
        
        result = self.rsi_divergence_strategy.on_kline_close_3m()
        if result:
            self.signals['RSI_DIV'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['RSI_DIV'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_ichimoku_strategy(self):
        """Ichimoku 전략 실행"""
        if not self.ichimoku_strategy:
            return
        
        result = self.ichimoku_strategy.on_kline_close_htf()
        if result:
            self.signals['ICHIMOKU'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['ICHIMOKU'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_vwap_pinball_strategy(self):
        """VWAP 피니언 전략 실행"""
        if not self.vwap_pinball_strategy:
            return
        config = VWAPPinballCfg()
        df_3m = self.data_manager.get_latest_data(count=config.lookback_bars)
        result = self.vwap_pinball_strategy.on_kline_close_3m(df_3m)

        if result:
            self.signals['VWAP_PINBALL'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'entry': result.get('entry', 0),
                'stop': result.get('stop', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['VWAP_PINBALL'] = {
                'action': 'HOLD',
                'score': 0,
                'entry': 0,
                'stop': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def _execute_vol_spike_strategy(self):
        """Vol Spike 전략 실행"""
        if not self.vol_spike_strategy:
            return
        
        from signals.vol_spike_3m import VolSpikeConfig
        config = VolSpikeConfig()
        _count = max(5, config.window + 1)

        df_3m = self.data_manager.get_latest_data(count=_count)
        result = self.vol_spike_strategy.on_kline_close_3m(df_3m)
        
        if result:
            self.signals['VOL_SPIKE'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }
        else:
            # result가 없을 때 기본 결과값 저장
            self.signals['VOL_SPIKE'] = {
                'action': 'HOLD',
                'score': 0,
                'timestamp': getattr(self, '_cached_timestamp', self.time_manager.get_current_time())
            }

    def get_signals(self) -> Dict[str, Dict[str, Any]]:
        """현재 시그널 반환 (최적화: copy 최소화)"""
        # copy()는 필요하지만, 호출 빈도를 줄이기 위해 최적화
        return self.signals.copy()
