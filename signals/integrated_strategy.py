from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from config.integrated_config import IntegratedConfig
from signals.hybrid_strategy import make_hybrid_trade_plan, HybridConfig
from signals.liquidation_strategy import LiquidationStrategy, LiquidationConfig
from signals.liquidation_prediction import LiquidationPredictionStrategy, LiquidationPredictionConfig
from signals.timing_strategy import TimingStrategy, TimingConfig
from signals.session_based_strategy import SessionBasedStrategy, SessionConfig
from signals.advanced_liquidation_strategy import AdvancedLiquidationStrategy, AdvancedLiquidationConfig

class IntegratedStrategy:
    """통합 전략: 하이브리드 + 실시간 청산"""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self.last_hybrid_signal = None
        self.last_liquidation_signal = None
        self.synergy_signals = []
        
        # 하이브리드 전략 설정
        self.hybrid_cfg = HybridConfig(
            min_hybrid_confidence=config.hybrid_min_confidence,
            trend_weight=config.hybrid_trend_weight,
            entry_weight=config.hybrid_entry_weight,
            atr_len=14,
            atr_stop_mult=1.0,
            atr_tp1_mult=1.5,
            atr_tp2_mult=2.5,
            vpvr_bins=50,
            vpvr_lookback=200
        )
        
        # 청산 전략 설정
        self.liquidation_cfg = LiquidationConfig(
            min_liquidation_count=config.liquidation_min_count,
            min_liquidation_value=config.liquidation_min_value,
            buy_liquidation_ratio=config.liquidation_buy_ratio,
            sell_liquidation_ratio=config.liquidation_sell_ratio,
            volume_spike_threshold=config.liquidation_volume_threshold,
            liquidation_window_minutes=config.liquidation_window_minutes
        )
        
        # 청산 예측 전략 설정
        self.prediction_cfg = LiquidationPredictionConfig(
            price_bin_size=config.prediction_price_bin_size,
            min_liquidation_density=config.prediction_min_density,
            cascade_threshold=config.prediction_cascade_threshold,
            min_prediction_confidence=config.prediction_min_confidence,
            max_prediction_horizon_hours=config.prediction_max_horizon_hours
        )
        
        # 타이밍 전략 설정
        self.timing_cfg = TimingConfig(
            entry_confidence_min=config.timing_entry_confidence_min,
            entry_rr_min=config.timing_entry_rr_min,
            entry_score_threshold=config.timing_entry_score_threshold,
            max_hold_time_hours=config.timing_max_hold_time_hours,
            trailing_stop_atr=config.timing_trailing_stop_atr
        )
        
        # 세션 기반 전략 설정
        self.session_cfg = SessionConfig(
            symbol=config.symbol,
            timeframe=config.session_timeframe,
            ses_vwap_start_utc=config.session_vwap_start_utc,
            or_minutes=config.session_or_minutes,
            min_drive_return_R=config.session_min_drive_return_R,
            pullback_depth_atr=config.session_pullback_depth_atr,
            trigger_type=config.session_trigger_type,
            stop_atr_mult=config.session_stop_atr_mult,
            tp1_R=config.session_tp1_R,
            tp2_to_level=config.session_tp2_to_level,
            partial_out=config.session_partial_out,
            max_hold_min=config.session_max_hold_min,
            max_slippage_pct=config.session_max_slippage_pct,
            sweep_depth_atr_min=config.session_sweep_depth_atr_min,
            reclaim_close_rule=config.session_reclaim_close_rule,
            confirm_next_bar=config.session_confirm_next_bar,
            stop_buffer_atr=config.session_stop_buffer_atr,
            tp1_to=config.session_tp1_to,
            tp2_to=config.session_tp2_to,
            sd_k_enter=config.session_sd_k_enter,
            sd_k_reenter=config.session_sd_k_reenter,
            stop_outside_sd_k=config.session_stop_outside_sd_k,
            tp2_to_band=config.session_tp2_to_band,
            trend_filter_slope=config.session_trend_filter_slope
        )
        
        # 고급 청산 전략 설정
        self.adv_liquidation_cfg = AdvancedLiquidationConfig(
            symbol=config.symbol,
            bin_sec=config.adv_liq_bin_sec,
            agg_window_sec=config.adv_liq_agg_window_sec,
            background_window_min=config.adv_liq_background_window_min,
            z_spike=config.adv_liq_z_spike,
            z_strong=config.adv_liq_z_strong,
            lpi_bias=config.adv_liq_lpi_bias,
            cascade_seconds=config.adv_liq_cascade_seconds,
            cascade_count=config.adv_liq_cascade_count,
            cascade_z=config.adv_liq_cascade_z,
            cooldown_after_strong_sec=config.adv_liq_cooldown_after_strong_sec,
            risk_pct=config.adv_liq_risk_pct,
            slippage_max_pct=config.adv_liq_slippage_max_pct,
            or_minutes=config.adv_liq_or_minutes,
            atr_len=config.adv_liq_atr_len,
            vwap_sd_enter=config.adv_liq_vwap_sd_enter,
            vwap_sd_stop=config.adv_liq_vwap_sd_stop,
            sweep_buffer_atr=config.adv_liq_sweep_buffer_atr,
            tp1_R=config.adv_liq_tp1_R,
            tp2=config.adv_liq_tp2,
            retest_atr_tol=config.adv_liq_retest_atr_tol,
            or_extension=config.adv_liq_or_extension,
            post_spike_decay_ratio=config.adv_liq_post_spike_decay_ratio,
            stop_atr=config.adv_liq_stop_atr,
            tp2_sigma=config.adv_liq_tp2_sigma
        )
        
        # 전략 인스턴스
        self.liquidation_strategy = LiquidationStrategy(self.liquidation_cfg)
        self.prediction_strategy = LiquidationPredictionStrategy(self.prediction_cfg)
        self.timing_strategy = TimingStrategy(self.timing_cfg)
        self.session_strategy = SessionBasedStrategy(self.session_cfg)
        self.adv_liquidation_strategy = AdvancedLiquidationStrategy(self.adv_liquidation_cfg)
    
    def analyze_hybrid_strategy(self, df_15m: pd.DataFrame, df_5m: pd.DataFrame, vpvr_levels: List[Dict]) -> Optional[Dict]:
        """하이브리드 전략 분석"""
        if not self.config.enable_hybrid_strategy:
            return None
        
        try:
            plan = make_hybrid_trade_plan(df_15m, df_5m, vpvr_levels, self.hybrid_cfg)
            if plan and plan.get('final_signal') != 'NEUTRAL':
                self.last_hybrid_signal = plan
                return plan
        except Exception as e:
            print(f"❌ 하이브리드 전략 분석 오류: {e}")
        
        return None
    
    def analyze_liquidation_strategy(self, 
                                    liquidation_stats: Dict, 
                                    volume_analysis: Dict,
                                    current_price: float) -> Optional[Dict]:
        """청산 전략 분석 (ATR 제거)"""
        if not self.config.enable_liquidation_strategy:
            return None
        
        try:
            signal = self.liquidation_strategy.analyze_liquidation_signal(
                liquidation_stats, volume_analysis, current_price
            )
            if signal:
                self.last_liquidation_signal = signal
                return signal
        except Exception as e:
            print(f"❌ 청산 전략 분석 오류: {e}")
        
        return None
    
    def analyze_liquidation_prediction(self, 
                                        liquidations: List[Dict],
                                        current_price: float) -> Optional[Dict]:
        """청산 예측 분석"""
        if not self.config.enable_liquidation_prediction:
            return None
        
        try:
            # 폭등/폭락 지점 예측
            predictions = self.prediction_strategy.predict_explosion_points(liquidations, current_price)
            
            if predictions:
                # 가장 신뢰도 높은 예측 반환
                best_prediction = predictions[0]
                return {
                    'type': 'PREDICTION',
                    'predictions': predictions,
                    'best_prediction': best_prediction,
                    'current_price': current_price,
                    'timestamp': datetime.now(),
                    'prediction_summary': self.prediction_strategy.get_prediction_summary()
                }
        except Exception as e:
            print(f"❌ 청산 예측 분석 오류: {e}")
        
        return None
    
    def analyze_session_strategy(self, df_1m: pd.DataFrame, 
                                key_levels: Dict[str, float],
                                current_time: datetime) -> Optional[Dict]:
        """세션 기반 전략 분석"""
        if not self.config.enable_session_strategy:
            return None
        
        try:
            signal = self.session_strategy.analyze_session_strategy(
                df_1m, key_levels, current_time
            )
            if signal:
                return signal
        except Exception as e:
            print(f"❌ 세션 전략 분석 오류: {e}")
        
        return None
    
    def analyze_advanced_liquidation_strategy(self, 
                                            df_1m: pd.DataFrame,
                                            liquidation_events: List[Dict],
                                            key_levels: Dict[str, float],
                                            opening_range: Dict[str, float],
                                            vwap: float,
                                            vwap_std: float) -> Optional[Dict]:
        """고급 청산 전략 분석"""
        if not self.config.enable_advanced_liquidation:
            return None
        
        try:
            # ATR 계산
            from indicators.atr import calculate_atr
            atr = calculate_atr(df_1m, self.adv_liquidation_cfg.atr_len)
            if pd.isna(atr):
                atr = df_1m['close'].iloc[-1] * 0.02  # 기본값
            
            signal = self.adv_liquidation_strategy.analyze_all_strategies(
                df_1m, key_levels, opening_range, vwap, vwap_std, atr
            )
            if signal:
                return signal
        except Exception as e:
            print(f"❌ 고급 청산 전략 분석 오류: {e}")
        
        return None
    
    def check_synergy(self, hybrid_signal: Dict, liquidation_signal: Dict) -> Tuple[bool, float, str]:
        """시너지 효과 확인"""
        if not self.config.enable_synergy_signals:
            return False, 0.0, ""
        
        # 신호 방향 일치성 확인
        hybrid_action = hybrid_signal.get('final_signal', 'NEUTRAL')
        liquidation_action = liquidation_signal.get('action', 'WAIT')
        
        # 방향 일치 여부
        direction_match = False
        if hybrid_action == 'BUY' and liquidation_action == 'BUY':
            direction_match = True
        elif hybrid_action == 'SELL' and liquidation_action == 'SELL':
            direction_match = True
        
        if not direction_match:
            return False, 0.0, "방향 불일치"
        
        # 신뢰도 계산
        hybrid_confidence = hybrid_signal.get('confidence', 0)
        liquidation_confidence = liquidation_signal.get('confidence', 0)
        
        # 시너지 신뢰도 (두 전략의 평균 + 보너스)
        synergy_confidence = (hybrid_confidence + liquidation_confidence) / 2 * 1.2  # 기본 시너지 부스트
        synergy_confidence = min(0.95, synergy_confidence)  # 최대 95%로 제한
        
        # 최소 신뢰도 확인
        if synergy_confidence < 0.7:  # 기본 시너지 신뢰도 임계값
            return False, synergy_confidence, "신뢰도 부족"
        
        return True, synergy_confidence, "시너지 감지"
    
    def generate_synergy_signal(self, hybrid_signal: Dict, liquidation_signal: Dict) -> Dict:
        """시너지 신호 생성"""
        
        # 기본 정보는 하이브리드 신호에서 가져오기
        base_signal = hybrid_signal.copy()
        
        # 청산 정보 추가
        base_signal['liquidation_stats'] = liquidation_signal.get('liquidation_stats', {})
        base_signal['volume_analysis'] = liquidation_signal.get('volume_analysis', {})
        
        # 시너지 정보 추가
        base_signal['signal_type'] = 'SYNERGY'
        
        # 신뢰도 계산
        hybrid_confidence = hybrid_signal.get('confidence', 0)
        liquidation_confidence = liquidation_signal.get('confidence', 0)
        synergy_confidence = (hybrid_confidence + liquidation_confidence) / 2 * 1.2  # 기본 시너지 부스트
        synergy_confidence = min(0.95, synergy_confidence)  # 최대 95%로 제한
        
        base_signal['synergy_confidence'] = synergy_confidence
        base_signal['hybrid_confidence'] = hybrid_confidence
        base_signal['liquidation_confidence'] = liquidation_confidence
        
        # 신뢰도와 리스크/보상 보정
        base_signal['confidence'] = synergy_confidence
        base_signal['risk_reward'] = base_signal.get('risk_reward', 0) * 1.2  # 기본 시너지 RR 부스트
        
        # 시너지 이유
        base_signal['synergy_reason'] = (
            f"🔥🔥🔥 SYNERGY 신호! 🔥🔥🔥\n"
            f"하이브리드: {hybrid_confidence:.1%} | "
            f"청산: {liquidation_confidence:.1%} | "
            f"종합: {synergy_confidence:.1%}"
        )
        
        # 시너지 신호 저장
        synergy_signal = {
            'timestamp': datetime.now(),
            'hybrid_signal': hybrid_signal,
            'liquidation_signal': liquidation_signal,
            'synergy_confidence': synergy_confidence,
            'signal_type': 'SYNERGY'
        }
        self.synergy_signals.append(synergy_signal)
        
        return base_signal
    
    def get_integrated_signal(self, 
                             hybrid_signal: Optional[Dict] = None,
                             liquidation_signal: Optional[Dict] = None,
                             prediction_signal: Optional[Dict] = None,
                             session_signal: Optional[Dict] = None,
                             advanced_liquidation_signal: Optional[Dict] = None) -> Optional[Dict]:
        """통합 신호 생성 (모든 전략 포함)"""
        try:
            # 활성화된 전략들만 수집
            active_signals = []
            
            if hybrid_signal and self.config.enable_hybrid_strategy:
                active_signals.append(('HYBRID', hybrid_signal))
            
            if liquidation_signal and self.config.enable_liquidation_strategy:
                active_signals.append(('LIQUIDATION', liquidation_signal))
            
            if prediction_signal and self.config.enable_liquidation_prediction:
                active_signals.append(('PREDICTION', prediction_signal))
            
            if session_signal and self.config.enable_session_strategy:
                active_signals.append(('SESSION', session_signal))
            
            if advanced_liquidation_signal and self.config.enable_advanced_liquidation:
                active_signals.append(('ADVANCED_LIQUIDATION', advanced_liquidation_signal))
            
            if not active_signals:
                return None
            
            # 단일 신호인 경우
            if len(active_signals) == 1:
                signal_type, signal = active_signals[0]
                return {
                    'signal_type': signal_type,
                    'final_signal': signal.get('action', 'NEUTRAL'),
                    'confidence': signal.get('confidence', 0),
                    'entry_price': signal.get('entry_price', 0),
                    'stop_loss': signal.get('stop_loss', 0),
                    'take_profit1': signal.get('take_profit1', 0),
                    'take_profit2': signal.get('take_profit2', 0),
                    'risk_reward': signal.get('risk_reward', 0),
                    'timestamp': signal.get('timestamp'),
                    'reason': signal.get('reason', ''),
                    'playbook': signal.get('playbook', ''),
                    'partial_out': signal.get('partial_out', 0.5),
                    'max_hold_min': signal.get('max_hold_min', 60)
                }
            
            # 다중 신호인 경우 시너지 확인
            if len(active_signals) >= 2:
                # 시너지 효과 확인
                synergy_result = self._check_multi_synergy(active_signals)
                if synergy_result:
                    return synergy_result
            
            # 시너지가 없으면 가장 높은 신뢰도 신호 반환
            best_signal = max(active_signals, key=lambda x: x[1].get('confidence', 0))
            signal_type, signal = best_signal
            
            return {
                'signal_type': signal_type,
                'final_signal': signal.get('action', 'NEUTRAL'),
                'confidence': signal.get('confidence', 0),
                'entry_price': signal.get('entry_price', 0),
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit1': signal.get('take_profit1', 0),
                'take_profit2': signal.get('take_profit2', 0),
                'risk_reward': signal.get('risk_reward', 0),
                'timestamp': signal.get('timestamp'),
                'reason': signal.get('reason', ''),
                'playbook': signal.get('playbook', ''),
                'partial_out': signal.get('partial_out', 0.5),
                'max_hold_min': signal.get('max_hold_min', 60)
            }
            
        except Exception as e:
            print(f"❌ 통합 신호 생성 오류: {e}")
            return None
    
    def _check_multi_synergy(self, active_signals: List[Tuple[str, Dict]]) -> Optional[Dict]:
        """다중 전략 시너지 확인"""
        try:
            # 모든 신호가 같은 방향인지 확인
            actions = [signal.get('action', 'NEUTRAL') for _, signal in active_signals]
            unique_actions = set(actions)
            
            if len(unique_actions) != 1 or 'NEUTRAL' in unique_actions:
                return None
            
            action = list(unique_actions)[0]
            
            # 신뢰도 평균 계산
            confidences = [signal.get('confidence', 0) for _, signal in active_signals]
            avg_confidence = sum(confidences) / len(confidences)
            
            # 시너지 보너스 적용
            synergy_confidence = min(0.95, avg_confidence * 1.2)
            
            # 첫 번째 신호를 기준으로 통합
            base_signal = active_signals[0][1]
            
            # 시너지 이유 생성
            signal_names = [name for name, _ in active_signals]
            synergy_reason = f"시너지 효과: {' + '.join(signal_names)} | 신뢰도: {avg_confidence:.1%} → {synergy_confidence:.1%}"
            
            return {
                'signal_type': 'SYNERGY',
                'final_signal': action,
                'confidence': synergy_confidence,
                'entry_price': base_signal.get('entry_price', 0),
                'stop_loss': base_signal.get('stop_loss', 0),
                'take_profit1': base_signal.get('take_profit1', 0),
                'take_profit2': base_signal.get('take_profit2', 0),
                'risk_reward': base_signal.get('risk_reward', 0),
                'timestamp': base_signal.get('timestamp'),
                'synergy_reason': synergy_reason,
                'synergy_signals': signal_names,
                'playbook': base_signal.get('playbook', ''),
                'partial_out': base_signal.get('partial_out', 0.5),
                'max_hold_min': base_signal.get('max_hold_min', 60)
            }
            
        except Exception as e:
            print(f"❌ 다중 시너지 확인 오류: {e}")
            return None
    
    def _convert_prediction_to_signal(self, prediction: Dict) -> Dict:
        """예측을 거래 신호로 변환"""
        
        prediction_type = prediction.get('type', 'UNKNOWN')
        current_price = prediction.get('current_price', 0)
        center_price = prediction.get('center_price', current_price)
        confidence = prediction.get('confidence', 0)
        risk_score = prediction.get('risk_score', 0)
        
        # ATR 계산 (간단한 변동성)
        atr = current_price * 0.02
        
        if prediction_type == 'EXPLOSION_UP':
            # 폭등 예측 → BUY 신호
            action = 'BUY'
            bias = 'LONG'
            stop_loss = center_price - (atr * 1.5)
            take_profit1 = center_price + (atr * 2.0)
            take_profit2 = center_price + (atr * 3.0)
        elif prediction_type == 'EXPLOSION_DOWN':
            # 폭락 예측 → SELL 신호
            action = 'SELL'
            bias = 'SHORT'
            stop_loss = center_price + (atr * 1.5)
            take_profit1 = center_price - (atr * 2.0)
            take_profit2 = center_price - (atr * 3.0)
        else:
            return None
        
        # 리스크/보상 계산
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit1 - current_price)
        risk_reward = reward / risk if risk > 0 else 0
        
        return {
            'action': action,
            'bias': bias,
            'timestamp': datetime.now(),
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit1': take_profit1,
            'take_profit2': take_profit2,
            'confidence': confidence,
            'risk_reward': risk_reward,
            'atr': atr,
            'signal_type': 'PREDICTION',
            'prediction_type': prediction_type,
            'prediction_confidence': confidence,
            'risk_score': risk_score,
            'reason': f"🔮 {prediction_type}: {prediction.get('reason', '')}"
        }
    
    def get_signal_summary(self) -> Dict:
        """신호 요약 정보"""
        return {
            'last_hybrid_signal': self.last_hybrid_signal,
            'last_liquidation_signal': self.last_liquidation_signal,
            'synergy_signals_count': len(self.synergy_signals),
            'recent_synergy_signals': self.synergy_signals[-5:] if self.synergy_signals else [],
            'config': {
                'enable_hybrid': self.config.enable_hybrid_strategy,
                'enable_liquidation': self.config.enable_liquidation_strategy,
                'enable_synergy': self.config.enable_synergy_signals
            }
        }
    
    def cleanup_old_data(self, max_age_hours: int = 24):
        """오래된 데이터 정리"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # 시너지 신호 정리
        self.synergy_signals = [
            signal for signal in self.synergy_signals 
            if signal['timestamp'] > cutoff_time
        ]
        
        # 오래된 신호 정리
        if (self.last_hybrid_signal and 
            self.last_hybrid_signal.get('timestamp', datetime.now()) < cutoff_time):
            self.last_hybrid_signal = None
        
        if (self.last_liquidation_signal and 
            self.last_liquidation_signal.get('timestamp', datetime.now()) < cutoff_time):
            self.last_liquidation_signal = None

    def get_explosion_alert(self, 
                           hybrid_signal: Optional[Dict] = None,
                           liquidation_signal: Optional[Dict] = None,
                           prediction_signal: Optional[Dict] = None) -> Optional[Dict]:
        """폭등/폭락 경고 생성"""
        
        alerts = []
        
        # 1. 하이브리드 신호 기반 경고
        if hybrid_signal:
            confidence = hybrid_signal.get('confidence', 0)
            if confidence > 0.7:
                alerts.append({
                    'type': 'HYBRID_HIGH_CONFIDENCE',
                    'level': 'HIGH',
                    'message': f"하이브리드 신호 신뢰도 높음: {confidence:.1%}",
                    'confidence': confidence
                })
        
        # 2. 청산 신호 기반 경고
        if liquidation_signal:
            confidence = liquidation_signal.get('confidence', 0)
            if confidence > 0.6:
                alerts.append({
                    'type': 'LIQUIDATION_HIGH_CONFIDENCE',
                    'level': 'MEDIUM',
                    'message': f"청산 신호 신뢰도 높음: {confidence:.1%}",
                    'confidence': confidence
                })
        
        # 3. 예측 신호 기반 경고
        if prediction_signal:
            best_prediction = prediction_signal.get('best_prediction')
            if best_prediction:
                prediction_type = best_prediction.get('type', 'UNKNOWN')
                confidence = best_prediction.get('confidence', 0)
                target_price = best_prediction.get('target_price', 0)
                expected_time = best_prediction.get('expected_time')
                
                if prediction_type == 'EXPLOSION_UP':
                    alerts.append({
                        'type': 'EXPLOSION_UP_PREDICTION',
                        'level': 'CRITICAL',
                        'message': f"🚀 폭등 예측! 목표가: ${target_price:.2f}",
                        'confidence': confidence,
                        'expected_time': expected_time,
                        'prediction_type': prediction_type
                    })
                elif prediction_type == 'EXPLOSION_DOWN':
                    alerts.append({
                        'type': 'EXPLOSION_DOWN_PREDICTION',
                        'level': 'CRITICAL',
                        'message': f"💥 폭락 예측! 목표가: ${target_price:.2f}",
                        'confidence': confidence,
                        'expected_time': expected_time,
                        'prediction_type': prediction_type
                    })
        
        if alerts:
            return {
                'timestamp': datetime.now(),
                'alerts': alerts,
                'total_alerts': len(alerts),
                'critical_alerts': len([a for a in alerts if a['level'] == 'CRITICAL']),
                'high_alerts': len([a for a in alerts if a['level'] == 'HIGH']),
                'medium_alerts': len([a for a in alerts if a['level'] == 'MEDIUM'])
            }
        
        return None
