from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from config.integrated_config import IntegratedConfig
from signals.hybrid_strategy import make_hybrid_trade_plan, HybridConfig
from signals.liquidation_strategy import LiquidationStrategy, LiquidationConfig
from signals.liquidation_prediction import LiquidationPredictionStrategy, LiquidationPredictionConfig
from signals.timing_strategy import TimingStrategy, TimingConfig

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
        
        # 전략 인스턴스
        self.liquidation_strategy = LiquidationStrategy(self.liquidation_cfg)
        self.prediction_strategy = LiquidationPredictionStrategy(self.prediction_cfg)
        self.timing_strategy = TimingStrategy(self.timing_cfg)
    
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
                                   current_price: float,
                                   atr: float) -> Optional[Dict]:
        """청산 전략 분석"""
        if not self.config.enable_liquidation_strategy:
            return None
        
        try:
            signal = self.liquidation_strategy.analyze_liquidation_signal(
                liquidation_stats, volume_analysis, current_price, atr
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
        synergy_confidence = (hybrid_confidence + liquidation_confidence) / 2 * self.config.synergy_confidence_boost
        
        # 최소 신뢰도 확인
        if synergy_confidence < self.config.min_synergy_confidence:
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
        synergy_confidence = (hybrid_confidence + liquidation_confidence) / 2 * self.config.synergy_confidence_boost
        
        base_signal['synergy_confidence'] = synergy_confidence
        base_signal['hybrid_confidence'] = hybrid_confidence
        base_signal['liquidation_confidence'] = liquidation_confidence
        
        # 신뢰도와 리스크/보상 보정
        base_signal['confidence'] = synergy_confidence
        base_signal['risk_reward'] = base_signal.get('risk_reward', 0) * self.config.synergy_rr_boost
        
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
                             prediction_signal: Optional[Dict] = None) -> Optional[Dict]:
        """통합 신호 생성"""
        
        # 시너지 확인
        if (hybrid_signal and liquidation_signal and 
            self.config.enable_synergy_signals):
            
            has_synergy, synergy_confidence, reason = self.check_synergy(hybrid_signal, liquidation_signal)
            
            if has_synergy:
                print(f"🎯 시너지 감지: {synergy_confidence:.1%}")
                return self.generate_synergy_signal(hybrid_signal, liquidation_signal)
            else:
                print(f"⏳ 시너지 없음: {reason}")
        
        # 예측 신호가 있으면 우선 처리
        if prediction_signal and self.config.enable_liquidation_prediction:
            best_prediction = prediction_signal.get('best_prediction')
            if best_prediction:
                print(f"🔮 예측 신호 감지: {best_prediction.get('type', 'UNKNOWN')} - {best_prediction.get('confidence', 0):.1%}")
                return self._convert_prediction_to_signal(best_prediction)
        
        # 개별 신호 반환 (우선순위: 시너지 > 예측 > 하이브리드 > 청산)
        if hybrid_signal:
            # 하이브리드 신호에 signal_type이 없으면 추가
            if 'signal_type' not in hybrid_signal:
                hybrid_signal['signal_type'] = 'HYBRID'
            return hybrid_signal
        elif liquidation_signal:
            # 청산 신호에 signal_type이 없으면 추가
            if 'signal_type' not in liquidation_signal:
                liquidation_signal['signal_type'] = 'LIQUIDATION'
            return liquidation_signal
        
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
