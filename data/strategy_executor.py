from typing import Dict, Any, Optional
from datetime import datetime
from data.data_manager import get_data_manager
from signals.vpvr_micro import VPVRConfig
from signals.vwap_pinball_strategy import VWAPPinballCfg
from utils.time_manager import get_time_manager


class StrategyExecutor:
    """전략 실행을 담당하는 클래스"""
    
    def __init__(self):
        self.data_manager = get_data_manager()
        self.time_manager = get_time_manager()
        self.signals = {}
        
        # 전략 인스턴스들
        self.session_strategy = None
        self.vpvr_golden_strategy = None
        self.bollinger_squeeze_strategy = None
        self.ema_trend_15m_strategy = None
        self.orderflow_cvd_strategy = None
        self.rsi_divergence_strategy = None
        self.ichimoku_strategy = None
        self.vwap_pinball_strategy = None
        self.vol_spike_strategy = None
        self.vpvr_micro_strategy = None
        self.zscore_mean_reversion_strategy = None
        self.htf_trend_15m_strategy = None
        
        # 전략들 자동 초기화
        self._initialize_all_strategies()

    def _initialize_all_strategies(self):
        """모든 전략 초기화"""
        try:
            # 세션 전략
            self._init_session_strategy()
            
            # VPVR 전략들
            self._init_vpvr_golden_strategy()
            self._init_vpvr_micro_strategy()
            
            # 기술적 분석 전략들
            self._init_bollinger_squeeze_strategy()
            self._init_ema_trend_15m_strategy()
            self._init_rsi_divergence_strategy()
            self._init_ichimoku_strategy()
            self._init_htf_trend_15m_strategy()
            
            # 고급 전략들
            self._init_orderflow_cvd_strategy()
            self._init_vwap_pinball_strategy()
            self._init_vol_spike_strategy()
            self._init_zscore_mean_reversion_strategy()
            
            print("🎯 모든 전략 초기화 완료")
            
        except Exception as e:
            print(f"❌ 전략 초기화 오류: {e}")
            import traceback
            traceback.print_exc()

    def _init_session_strategy(self):
        """세션 전략 초기화"""
        try:
            from signals.session_or_lite import SessionORLite, SessionORLiteCfg
            config = SessionORLiteCfg()
            self.session_strategy = SessionORLite(config)
        except Exception as e:
            print(f"❌ 세션 전략 초기화 오류: {e}")
            self.session_strategy = None

    def _init_vpvr_golden_strategy(self):
        """VPVR 골든 포켓 전략 초기화"""
        try:
            from signals.vpvr_golden_strategy import LVNGoldenPocket
            self.vpvr_golden_strategy = LVNGoldenPocket()
        except Exception as e:
            print(f"❌ VPVR 골든 포켓 전략 초기화 오류: {e}")
            self.vpvr_golden_strategy = None

    def _init_vpvr_micro_strategy(self):
        """VPVR 마이크로 전략 초기화"""
        try:
            from signals.vpvr_micro import VPVRMicro, VPVRConfig
            config = VPVRConfig()
            self.vpvr_micro_strategy = VPVRMicro(config)
        except Exception as e:
            print(f"❌ VPVR 마이크로 전략 초기화 오류: {e}")
            self.vpvr_micro_strategy = None

    def _init_bollinger_squeeze_strategy(self):
        """볼린저 스퀴즈 전략 초기화"""
        try:
            from signals.bollinger_squeeze_strategy import BollingerSqueezeStrategy, BBSqueezeCfg
            config = BBSqueezeCfg()
            self.bollinger_squeeze_strategy = BollingerSqueezeStrategy(config)
        except Exception as e:
            print(f"❌ 볼린저 스퀴즈 전략 초기화 오류: {e}")
            self.bollinger_squeeze_strategy = None

    def _init_ema_trend_15m_strategy(self):
        """EMA 트렌드 전략 초기화"""
        try:
            from signals.ema_trend_15m import EMATrend15m
            self.ema_trend_15m_strategy = EMATrend15m()
        except Exception as e:
            print(f"❌ EMA 트렌드 전략 초기화 오류: {e}")
            self.ema_trend_15m_strategy = None

    def _init_rsi_divergence_strategy(self):
        """RSI 다이버전스 전략 초기화"""
        try:
            from signals.rsi_divergence import RSIDivergence
            self.rsi_divergence_strategy = RSIDivergence()
        except Exception as e:
            print(f"❌ RSI 다이버전스 전략 초기화 오류: {e}")
            self.rsi_divergence_strategy = None

    def _init_ichimoku_strategy(self):
        """일목균형 전략 초기화"""
        try:
            from signals.ichimoku import Ichimoku
            self.ichimoku_strategy = Ichimoku()
        except Exception as e:
            print(f"❌ 일목균형 전략 초기화 오류: {e}")
            self.ichimoku_strategy = None

    def _init_htf_trend_15m_strategy(self):
        """HTF 트렌드 전략 초기화"""
        try:
            from signals.htf_trend import HTFTrend, HTFConfig
            config = HTFConfig()
            self.htf_trend_15m_strategy = HTFTrend(config)
        except Exception as e:
            print(f"❌ HTF 트렌드 전략 초기화 오류: {e}")
            self.htf_trend_15m_strategy = None

    def _init_orderflow_cvd_strategy(self):
        """오더플로우 CVD 전략 초기화"""
        try:
            from signals.orderflow_cvd import OrderflowCVD
            self.orderflow_cvd_strategy = OrderflowCVD()
        except Exception as e:
            print(f"❌ 오더플로우 CVD 전략 초기화 오류: {e}")
            self.orderflow_cvd_strategy = None

    def _init_vwap_pinball_strategy(self):
        """VWAP 피니언 전략 초기화"""
        try:
            from signals.vwap_pinball_strategy import VWAPPinballStrategy
            self.vwap_pinball_strategy = VWAPPinballStrategy()
        except Exception as e:
            print(f"❌ VWAP 피니언 전략 초기화 오류: {e}")
            self.vwap_pinball_strategy = None

    def _init_vol_spike_strategy(self):
        """볼륨 스파이크 전략 초기화"""
        try:
            from signals.vol_spike_3m import VolSpike
            self.vol_spike_strategy = VolSpike()
        except Exception as e:
            print(f"❌ 볼륨 스파이크 전략 초기화 오류: {e}")
            self.vol_spike_strategy = None

    def _init_zscore_mean_reversion_strategy(self):
        """Z-Score 평균 회귀 전략 초기화"""
        try:
            from signals.zscore_mean_reversion import ZScoreMeanReversion, ZScoreConfig
            config = ZScoreConfig()
            self.zscore_mean_reversion_strategy = ZScoreMeanReversion(config)
        except Exception as e:
            print(f"❌ Z-Score 평균 회귀 전략 초기화 오류: {e}")
            self.zscore_mean_reversion_strategy = None

    def set_strategies(
        self,
        session_strategy=None,
        bollinger_squeeze_strategy=None,
        vpvr_golden_strategy=None,
        ema_trend_15m_strategy=None,
        orderflow_cvd_strategy=None,
        rsi_divergence_strategy=None,
        ichimoku_strategy=None,
        vwap_pinball_strategy=None,
        vol_spike_strategy=None,
        vpvr_micro_strategy=None,
        zscore_mean_reversion_strategy=None,
        htf_trend_15m_strategy=None,
    ):
        """전략 실행기 설정"""
        try:
            if session_strategy is not None:
                self.session_strategy = session_strategy
                print(f"✅ 세션 전략 설정 완료: {type(session_strategy).__name__}")
            
            if bollinger_squeeze_strategy is not None:
                self.bollinger_squeeze_strategy = bollinger_squeeze_strategy
                print(f"✅ 볼린저 스퀴즈 전략 설정 완료: {type(bollinger_squeeze_strategy).__name__}")
            
            if vpvr_golden_strategy is not None:
                self.vpvr_golden_strategy = vpvr_golden_strategy
                print(f"✅ VPVR 골든 포켓 전략 설정 완료: {type(vpvr_golden_strategy).__name__}")
                
            if ema_trend_15m_strategy is not None:
                self.ema_trend_15m_strategy = ema_trend_15m_strategy
                print(f"✅ EMA 트렌드 전략 설정 완료: {type(ema_trend_15m_strategy).__name__}")
            
            if orderflow_cvd_strategy is not None:
                self.orderflow_cvd_strategy = orderflow_cvd_strategy
                print(f"✅ ORDERFLOW CVD 전략 설정 완료: {type(orderflow_cvd_strategy).__name__}")
            
            if rsi_divergence_strategy is not None:
                self.rsi_divergence_strategy = rsi_divergence_strategy
                print(f"✅ HTF RSI Divergence 전략 설정 완료: {type(rsi_divergence_strategy).__name__}")
            
            if ichimoku_strategy is not None:
                self.ichimoku_strategy = ichimoku_strategy
                print(f"✅ Ichimoku 전략 설정 완료: {type(ichimoku_strategy).__name__}")
            
            if vwap_pinball_strategy is not None:
                self.vwap_pinball_strategy = vwap_pinball_strategy
                print(f"✅ VWAP Pinball 전략 설정 완료: {type(vwap_pinball_strategy).__name__}")
            
            if vol_spike_strategy is not None:
                self.vol_spike_strategy = vol_spike_strategy
                print(f"✅ Vol Spike 전략 설정 완료: {type(vol_spike_strategy).__name__}")
            
            if vpvr_micro_strategy is not None:
                self.vpvr_micro_strategy = vpvr_micro_strategy
                print(f"✅ VPVR Micro 전략 설정 완료: {type(vpvr_micro_strategy).__name__}")
            
            if zscore_mean_reversion_strategy is not None:
                self.zscore_mean_reversion_strategy = zscore_mean_reversion_strategy
                print(f"✅ ZScore Mean Reversion 전략 설정 완료: {type(zscore_mean_reversion_strategy).__name__}")
            
            if htf_trend_15m_strategy is not None:
                self.htf_trend_15m_strategy = htf_trend_15m_strategy
                print(f"✅ HTF Trend 15m 전략 설정 완료: {type(htf_trend_15m_strategy).__name__}")

        except Exception as e:
            print(f"❌ 전략 설정 오류: {e}")
            import traceback
            traceback.print_exc()

    def execute_all_strategies(self):
        """모든 전략 실행"""
        self.signals = {}  # 시그널 초기화
        
        self._execute_session_strategy()
        self._execute_vpvr_golden_strategy()
        self._execute_bollinger_squeeze_strategy()
        self._execute_ema_trend_15m_strategy()
        self._execute_orderflow_cvd_strategy()
        self._execute_rsi_divergence_strategy()
        self._execute_ichimoku_strategy()
        self._execute_vwap_pinball_strategy()
        self._execute_vol_spike_strategy()
        self._execute_vpvr_micro_strategy()
        self._execute_zscore_mean_reversion_strategy()
        self._execute_htf_trend_15m_strategy()

    def _execute_htf_trend_15m_strategy(self):
        """HTF Trend 15m 전략 실행"""
        if not self.htf_trend_15m_strategy:
            return
        df_15m = self.data_manager.get_15m_data(count=300)
        df_1h = self.data_manager.get_1h_data(count=300)
        result = self.htf_trend_15m_strategy.on_kline_close_15m(df_15m=df_15m, df_1h=df_1h)
        
        if result:
            self.signals['HTF_TREND_15M'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'confidence': result.get('confidence', 'LOW'),
                'timestamp': self.time_manager.get_current_time()
            }

    def _execute_vpvr_micro_strategy(self):
        """VPVR Micro 전략 실행"""
        if not self.vpvr_micro_strategy:
            return
        
        config = VPVRConfig()
        df_3m = self.data_manager.get_latest_data(count=config.lookback_bars)
        result = self.vpvr_micro_strategy.on_kline_close_3m(df_3m)
        
        if result:
            self.signals['VPVR_MICRO'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'confidence': result.get('confidence', 'LOW'),
                'timestamp': self.time_manager.get_current_time()
            }

    def _execute_zscore_mean_reversion_strategy(self):
        """ZScore Mean Reversion 전략 실행"""
        if not self.zscore_mean_reversion_strategy:
            return

        df_3m = self.data_manager.get_latest_data(count=300)
        result = self.zscore_mean_reversion_strategy.on_kline_close_3m(df_3m)
        
        if result:
            self.signals['ZSCORE_MEAN_REVERSION'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'confidence': result.get('confidence', 'LOW'),
                'timestamp': self.time_manager.get_current_time()
            }

    def _execute_session_strategy(self):
        """세션 전략 실행"""
        if not self.session_strategy:
            return
        
        df_3m = self.data_manager.get_latest_data(count=2)
        result = self.session_strategy.on_kline_close_3m(df_3m)
        
        if result:
            self.signals['SESSION'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0), 
                'confidence': result.get('confidence', 'LOW'),
                'entry': result.get('entry', 0),
                'stop': result.get('stop', 0),
                'timestamp': self.time_manager.get_current_time()
            }

    def _execute_vpvr_golden_strategy(self):
        """VPVR 골든 포켓 전략 실행"""
        if not self.vpvr_golden_strategy:
            return
        
        config = self.vpvr_golden_strategy.VPVRConfig()
        df_3m = self.data_manager.get_latest_data(count=config.lookback_bars + 5)
        sig = self.vpvr_golden_strategy.evaluate(df_3m)
        
        if sig:
            self.signals['VPVR'] = {
                'action': sig.get('action', 'UNKNOWN'),
                'score': sig.get('score', 0), 
                'confidence': sig.get('confidence', 'LOW'),
                'entry': sig.get('entry', 0),
                'stop': sig.get('stop', 0),
                'timestamp': self.time_manager.get_current_time()
            }

    def _execute_bollinger_squeeze_strategy(self):
        """볼린저 스퀴즈 전략 실행"""
        if not self.bollinger_squeeze_strategy:
            return
        
        result = self.bollinger_squeeze_strategy.evaluate()

        if result:
            self.signals['BB_SQUEEZE'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'confidence': result.get('confidence', 'LOW'),
                'entry': result.get('entry', 0),
                'stop': result.get('stop', 0),
                'timestamp': self.time_manager.get_current_time()
            }

    def _execute_ema_trend_15m_strategy(self):
        """EMA 트렌드 전략 실행 (15분봉)"""
        if not self.ema_trend_15m_strategy:
            return
        
        result = self.ema_trend_15m_strategy.on_kline_close_15m()
        if result:
            self.signals['EMA_TREND_15m'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'confidence': result.get('confidence', 'LOW'),
                'timestamp': self.time_manager.get_current_time()
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
                'confidence': result.get('confidence', 'LOW'),
                'timestamp': self.time_manager.get_current_time()
            }

    def _execute_rsi_divergence_strategy(self):
        """HTF RSI Divergence 전략 실행"""
        if not self.rsi_divergence_strategy:
            return
        
        result = self.rsi_divergence_strategy.on_kline_close_htf()
        if result:
            self.signals['RSI_DIV'] = {
                'action': result.get('action', 'UNKNOWN'),
                'score': result.get('score', 0),
                'confidence': result.get('confidence', 'LOW'),
                'timestamp': self.time_manager.get_current_time()
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
                'confidence': result.get('confidence', 'LOW'),
                'timestamp': self.time_manager.get_current_time()
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
                'confidence': result.get('confidence', 'LOW'),
                'entry': result.get('entry', 0),
                'stop': result.get('stop', 0),
                'timestamp': self.time_manager.get_current_time()
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
                'confidence': result.get('confidence', 'LOW'),
                'timestamp': self.time_manager.get_current_time()
            }

    def get_signals(self) -> Dict[str, Dict[str, Any]]:
        """현재 시그널 반환"""
        return self.signals.copy()
