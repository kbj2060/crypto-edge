# short_term_synergy_improvement.py
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

@dataclass
class SynergyConfig:
    # 신뢰도 기반 가중치
    confidence_weights: Dict[str, float] = None
    
    # 충돌 페널티
    conflict_penalties: Dict[str, float] = None
    
    # 최소 임계값 - 완화
    min_score_threshold: float = 0.3  # 0.6 -> 0.3 (완화)
    min_net_threshold: float = 0.1   # 0.15 -> 0.1 (완화)
    
    # 시너지 보너스
    consensus_bonus: float = 0.2
    high_confidence_boost: float = 1.3

    def __post_init__(self):
        if self.confidence_weights is None:
            self.confidence_weights = {
                'VWAP_PINBALL': 1.4,        # 실시간성 높음
                'LIQUIDITY_GRAB': 1.5,     # 패턴 완성도 높음  
                'ZSCORE_MEAN_REVERSION': 1.0,  # 통계적 신뢰도
                'VOL_SPIKE': 1.1,          # 거래량 확실성
                'ORDERFLOW_CVD': 0.8       # 노이즈 많음
            }
        
        if self.conflict_penalties is None:
            self.conflict_penalties = {
                'vwap_zscore_conflict': 0.8,      # 평균회귀 충돌
                'vol_orderflow_conflict': 0.7,    # 거래량 해석 차이
                'momentum_reversion_conflict': 0.75  # 방향성 충돌
            }
    
    def _create_hold_result(self, category: str, context: str = 'NEUTRAL') -> Dict[str, Any]:
        """HOLD 결과 생성"""
        return {
            'action': 'HOLD',
            'score': 0.0,
            'net_score': 0.0,
            'buy_score': 0.0,
            'sell_score': 0.0,
            'confidence': 'LOW',
            'market_context': context,
            'conflicts_detected': [],
            'bonuses_applied': [],
            'signals_used': 0,
            'category': category,
            'breakdown': {
                'buy_signals': [],
                'sell_signals': []
            },
            'meta': {
                'total_bonus_applied': 0.0,
                'conflict_count': 0,
                'strong_signals': [],
                'trend_strength': 'WEAK',
                'consolidation_level': 'NEUTRAL'
            }
        }
    
    

class ShortTermSynergyEngine:
    """SHORT_TERM 전략 시너지 최적화 엔진"""
    
    def __init__(self, config: SynergyConfig = SynergyConfig()):
        self.config = config
    
    def _create_hold_result(self, category: str, context: str = 'NEUTRAL') -> Dict[str, Any]:
        """HOLD 결과 생성"""
        return {
            'action': 'HOLD',
            'score': 0.0,
            'net_score': 0.0,
            'buy_score': 0.0,
            'sell_score': 0.0,
            'confidence': 'LOW',
            'market_context': context,
            'conflicts_detected': [],
            'bonuses_applied': [],
            'signals_used': 0,
            'category': category,
            'breakdown': {
                'buy_signals': [],
                'sell_signals': []
            },
            'meta': {
                'total_bonus_applied': 0.0,
                'conflict_count': 0,
                'strong_signals': [],
                'trend_strength': 'WEAK',
                'consolidation_level': 'NEUTRAL'
            }
        }
    
    def _create_final_result(self, net_score: float, buy_score: float, sell_score: float,
                           context: str, conflicts: Dict, bonuses: Dict, signals: List, 
                           weighted_signals: List, total_bonus: float, category: str) -> Dict[str, Any]:
        """최종 결과 생성"""
        # 임계값: 단기는 0.1
        min_threshold = self.config.min_net_threshold
        
        if abs(net_score) < min_threshold:
            action = 'HOLD'
            confidence = 'LOW'
        else:
            action = 'BUY' if net_score > 0 else 'SELL'
            if abs(net_score) > 0.4:
                confidence = 'HIGH'
            elif abs(net_score) > 0.25:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
        
        return {
            'action': action,
            'score': abs(net_score),
            'net_score': net_score,
            'buy_score': buy_score,
            'sell_score': sell_score,
            'confidence': confidence,
            'market_context': context,
            'conflicts_detected': list(conflicts.keys()),
            'bonuses_applied': list(bonuses.keys()),
            'signals_used': len(signals),
            'category': category,
            'breakdown': {
                'buy_signals': [{'name': s['name'], 'score': s.get('final_score', s['score']), 
                               'base_weight': s.get('base_weight', 1.0), 'penalty': s.get('penalty_applied', 1.0)} 
                               for s in weighted_signals if s['action'] == 'BUY'],
                'sell_signals': [{'name': s['name'], 'score': s.get('final_score', s['score']),
                                'base_weight': s.get('base_weight', 1.0), 'penalty': s.get('penalty_applied', 1.0)} 
                                for s in weighted_signals if s['action'] == 'SELL']
            },
            'meta': {
                'total_bonus_applied': total_bonus,
                'conflict_count': len(conflicts),
                'strong_signals': [s['name'] for s in signals if s['score'] > 0.8],
                'trend_strength': self._calculate_trend_strength(weighted_signals, context),
                'consolidation_level': self._calculate_consolidation_level(weighted_signals, context)
            }
        }
    
    def _calculate_trend_strength(self, weighted_signals: List[Dict[str, Any]], context: str) -> str:
        """트렌드 강도 계산"""
        trend_signals = ['VOL_SPIKE', 'ORDERFLOW_CVD']
        trend_scores = [s['final_score'] for s in weighted_signals if s['name'] in trend_signals]
        
        if not trend_scores:
            return 'WEAK'
        
        avg_trend_score = np.mean(trend_scores)
        
        if context == 'TRENDING' and avg_trend_score > 0.8:
            return 'STRONG'
        elif avg_trend_score > 0.6:
            return 'MEDIUM'
        else:
            return 'WEAK'
    
    def _calculate_consolidation_level(self, weighted_signals: List[Dict[str, Any]], context: str) -> str:
        """통합 수준 계산"""
        consolidation_signals = ['VWAP_PINBALL', 'ZSCORE_MEAN_REVERSION']
        consolidation_scores = [s['final_score'] for s in weighted_signals if s['name'] in consolidation_signals]
        
        if not consolidation_scores:
            return 'NEUTRAL'
        
        avg_consolidation_score = np.mean(consolidation_scores)
        
        if context == 'RANGING' and avg_consolidation_score > 0.8:
            return 'HIGH'
        elif avg_consolidation_score > 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _convert_signal_format(self, signals) -> List[Dict[str, Any]]:
        """다양한 신호 형식을 표준 리스트 형태로 변환"""
        if isinstance(signals, dict):
            # Dictionary 형태 → List 형태로 변환
            converted = []
            for strategy_name, signal_data in signals.items():
                if isinstance(signal_data, dict) and 'action' in signal_data and 'score' in signal_data:
                    converted.append({
                        'name': strategy_name,
                        'action': signal_data['action'],
                        'score': signal_data['score'],
                        **{k: v for k, v in signal_data.items() if k not in ['action', 'score']}
                    })
            return converted
        elif isinstance(signals, list):
            # 이미 리스트 형태면 그대로 반환
            return signals
        else:
            return []

    def detect_market_context(self, signals: List[Dict[str, Any]]) -> str:
        """시장 상황 감지"""
        # 1. 거래량 상태 확인
        vol_signals = [s for s in signals if 'VOL' in s['name']]
        high_volume = any(s['score'] > 0.7 for s in vol_signals)
        
        # 2. 평균회귀 vs 모멘텀 강도
        reversion_signals = [s for s in signals if any(x in s['name'] for x in ['VWAP', 'ZSCORE'])]
        momentum_signals = [s for s in signals if any(x in s['name'] for x in ['VOL', 'ORDERFLOW'])]
        
        avg_reversion_score = np.mean([s['score'] for s in reversion_signals]) if reversion_signals else 0
        avg_momentum_score = np.mean([s['score'] for s in momentum_signals]) if momentum_signals else 0
        
        # 3. 시장 상황 판단
        if high_volume and avg_momentum_score > avg_reversion_score + 0.2:
            return 'TRENDING'
        elif avg_reversion_score > avg_momentum_score + 0.2:
            return 'RANGING'  
        elif any(s['name'] == 'LIQUIDITY_GRAB' and s['score'] > 0.8 for s in signals):
            return 'BREAKOUT'
        else:
            return 'NEUTRAL'
    
    def apply_context_weights(self, signals: List[Dict[str, Any]], context: str) -> List[Dict[str, Any]]:
        """시장 상황별 가중치 조정"""
        context_multipliers = {
            'TRENDING': {
                'VOL_SPIKE': 1.3,
                'ORDERFLOW_CVD': 1.2,
                'VWAP_PINBALL': 0.9,
                'ZSCORE_MEAN_REVERSION': 0.8,
                'LIQUIDITY_GRAB': 0.9
            },
            'RANGING': {
                'VWAP_PINBALL': 1.3,
                'ZSCORE_MEAN_REVERSION': 1.2,
                'LIQUIDITY_GRAB': 1.0,
                'VOL_SPIKE': 0.8,
                'ORDERFLOW_CVD': 0.7
            },
            'BREAKOUT': {
                'LIQUIDITY_GRAB': 1.5,
                'VOL_SPIKE': 1.2,
                'VWAP_PINBALL': 0.9,
                'ORDERFLOW_CVD': 0.8,
                'ZSCORE_MEAN_REVERSION': 0.6
            },
            'NEUTRAL': {s['name']: 1.0 for s in signals}
        }
        
        multipliers = context_multipliers.get(context, {})
        
        adjusted_signals = []
        for signal in signals:
            mult = multipliers.get(signal['name'], 1.0)
            adjusted_signal = signal.copy()
            adjusted_signal['context_adjusted_score'] = signal['score'] * mult
            adjusted_signals.append(adjusted_signal)
            
        return adjusted_signals
    
    def detect_conflicts(self, signals: List[Dict[str, Any]]) -> Dict[str, float]:
        """전략 간 충돌 감지"""
        conflicts = {}
        signal_dict = {s['name']: s for s in signals}
        
        # 1. VWAP vs ZSCORE 충돌 (평균회귀 전략 간)
        vwap = signal_dict.get('VWAP_PINBALL')
        zscore = signal_dict.get('ZSCORE_MEAN_REVERSION')
        if vwap and zscore and vwap['action'] != zscore['action']:
            conflicts['vwap_zscore_conflict'] = min(vwap['score'], zscore['score'])
        
        # 2. VOL_SPIKE vs ORDERFLOW 충돌 (거래량 해석)
        vol = signal_dict.get('VOL_SPIKE')
        orderflow = signal_dict.get('ORDERFLOW_CVD')
        if vol and orderflow and vol['action'] != orderflow['action']:
            conflicts['vol_orderflow_conflict'] = min(vol['score'], orderflow['score'])
        
        # 3. 전반적 모멘텀 vs 평균회귀 충돌
        momentum_strategies = ['VOL_SPIKE', 'ORDERFLOW_CVD']
        reversion_strategies = ['VWAP_PINBALL', 'ZSCORE_MEAN_REVERSION']
        
        momentum_actions = [signal_dict[name]['action'] for name in momentum_strategies if name in signal_dict]
        reversion_actions = [signal_dict[name]['action'] for name in reversion_strategies if name in signal_dict]
        
        if momentum_actions and reversion_actions:
            momentum_consensus = max(set(momentum_actions), key=momentum_actions.count) if momentum_actions else None
            reversion_consensus = max(set(reversion_actions), key=reversion_actions.count) if reversion_actions else None
            
            if momentum_consensus and reversion_consensus and momentum_consensus != reversion_consensus:
                avg_momentum_score = np.mean([signal_dict[name]['score'] for name in momentum_strategies if name in signal_dict])
                avg_reversion_score = np.mean([signal_dict[name]['score'] for name in reversion_strategies if name in signal_dict])
                conflicts['momentum_reversion_conflict'] = min(avg_momentum_score, avg_reversion_score)
        
        return conflicts
    
    def calculate_synergy_score(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """시너지를 고려한 최종 점수 계산"""
        signals = self._convert_signal_format(signals)
        
        # 1. 시장 상황 감지
        market_context = self.detect_market_context(signals)
        
        # 2. 상황별 가중치 적용
        adjusted_signals = self.apply_context_weights(signals, market_context)
        
        # 3. 최소 임계값 필터링
        filtered_signals = [s for s in adjusted_signals if s['score'] >= self.config.min_score_threshold]
        
        if not filtered_signals:
            return self._create_hold_result('SHORT_TERM', market_context)
        
        # 4. 충돌 감지 및 페널티 적용
        conflicts = self.detect_conflicts(filtered_signals)
        
        # 5. 전략별 신뢰도 가중치 적용
        weighted_signals = []
        for signal in filtered_signals:
            base_weight = self.config.confidence_weights.get(signal['name'], 1.0)
            context_score = signal.get('context_adjusted_score', signal['score'])
            
            # 충돌 페널티 적용
            penalty = 1.0
            for conflict_type, conflict_strength in conflicts.items():
                if signal['name'] in ['VWAP_PINBALL', 'ZSCORE_MEAN_REVERSION'] and 'vwap_zscore' in conflict_type:
                    penalty *= self.config.conflict_penalties[conflict_type]
                elif signal['name'] in ['VOL_SPIKE', 'ORDERFLOW_CVD'] and 'vol_orderflow' in conflict_type:
                    penalty *= self.config.conflict_penalties[conflict_type]
                elif 'momentum_reversion' in conflict_type:
                    penalty *= self.config.conflict_penalties[conflict_type]
            
            final_score = context_score * base_weight * penalty
            weighted_signals.append({
                **signal,
                'final_score': final_score,
                'penalty_applied': penalty
            })
        
        # 6. 방향별 점수 집계
        buy_signals = [s for s in weighted_signals if s['action'] == 'BUY']
        sell_signals = [s for s in weighted_signals if s['action'] == 'SELL']
        
        buy_score = np.mean([s['final_score'] for s in buy_signals]) if buy_signals else 0.0
        sell_score = np.mean([s['final_score'] for s in sell_signals]) if sell_signals else 0.0
        
        # 7. 컨센서스 보너스
        if len(buy_signals) >= 3:
            buy_score *= (1 + self.config.consensus_bonus)
        if len(sell_signals) >= 3:
            sell_score *= (1 + self.config.consensus_bonus)
        
        # 8. 최종 결과 계산
        net_score = buy_score - sell_score
        
        # 시너지 보너스 총합 계산
        total_bonus_applied = 0
        if len(buy_signals) >= 3:
            total_bonus_applied += self.config.consensus_bonus
        if len(sell_signals) >= 3:
            total_bonus_applied += self.config.consensus_bonus
        
        return self._create_final_result(net_score, buy_score, sell_score, market_context, 
                                        conflicts, {}, filtered_signals, weighted_signals,
                                        total_bonus_applied, 'SHORT_TERM')
