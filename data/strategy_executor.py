from typing import Dict, Any, Optional
from datetime import datetime
from data.data_manager import get_data_manager
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
            
        except Exception as e:
            print(f"❌ 전략 설정 오류: {e}")
            import traceback
            traceback.print_exc()

    def execute_all_strategies(self, session_activated: bool = False):
        """모든 전략 실행"""
        self.signals = {}  # 시그널 초기화
        
        self._execute_session_strategy(session_activated)
        self._execute_vpvr_golden_strategy()
        self._execute_bollinger_squeeze_strategy()
        self._execute_ema_trend_15m_strategy()
        self._execute_orderflow_cvd_strategy()
        self._execute_rsi_divergence_strategy()
        self._execute_ichimoku_strategy()
        self._execute_vwap_pinball_strategy()
        self._execute_vol_spike_strategy()

    def _execute_session_strategy(self, session_activated: bool):
        """세션 전략 실행"""
        if not self.session_strategy:
            return
        
        df_3m = self.data_manager.get_latest_data(count=2)
        result = self.session_strategy.on_kline_close_3m(df_3m, session_activated)
        
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
        
        df_3m = self.data_manager.get_latest_data(count=4)
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
