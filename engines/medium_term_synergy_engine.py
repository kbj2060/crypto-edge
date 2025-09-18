from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

@dataclass
class MediumTermConfig:
    """중기 전략 설정"""
    # 신뢰도 기반 가중치
    confidence_weights: Dict[str, float] = None
    
    # 시장 상황별 가중치 배수
    market_context_multipliers: Dict[str, Dict[str, float]] = None
    
    # 충돌 페널티
    conflict_penalties: Dict[str, float] = None
    
    # 트렌드 강도 임계값
    trend_strength_threshold: float = 0.7
    support_resistance_threshold: float = 0.75
    
    # 시너지 보너스
    trend_confluence_bonus: float = 0.25
    support_resistance_bonus: float = 0.3
    timeframe_alignment_bonus: float = 0.2

    def __post_init__(self):
        if self.confidence_weights is None:
            self.confidence_weights = {
                'HTF_TREND': 1.5,           # 상위 시간대 트렌드는 매우 신뢰도 높음
                'MULTI_TIMEFRAME': 1.4,     # 다중 시간대 분석 중요
                'SUPPORT_RESISTANCE': 1.3,  # 기술적 분석 핵심
                'EMA_CONFLUENCE': 1.2,      # 이동평균 집합 신뢰도
                'BOLLINGER_SQUEEZE': 1.0    # 변동성 기반, 보조 지표
            }
        
        if self.market_context_multipliers is None:
            self.market_context_multipliers = {
                'STRONG_TREND': {
                    'HTF_TREND': 1.4,
                    'MULTI_TIMEFRAME': 1.3,
                    'EMA_CONFLUENCE': 1.2,
                    'SUPPORT_RESISTANCE': 0.9,
                    'BOLLINGER_SQUEEZE': 0.8
                },
                'CONSOLIDATION': {
                    'SUPPORT_RESISTANCE': 1.4,
                    'BOLLINGER_SQUEEZE': 1.3,
                    'EMA_CONFLUENCE': 1.1,
                    'HTF_TREND': 0.8,
                    'MULTI_TIMEFRAME': 0.9
                },
                'BREAKOUT_PENDING': {
                    'BOLLINGER_SQUEEZE': 1.5,
                    'SUPPORT_RESISTANCE': 1.3,
                    'HTF_TREND': 1.1,
                    'MULTI_TIMEFRAME': 1.2,
                    'EMA_CONFLUENCE': 1.0
                },
                'REVERSAL_ZONE': {
                    'SUPPORT_RESISTANCE': 1.4,
                    'MULTI_TIMEFRAME': 1.3,
                    'EMA_CONFLUENCE': 1.2,
                    'HTF_TREND': 0.7,
                    'BOLLINGER_SQUEEZE': 0.9
                }
            }
        
        if self.conflict_penalties is None:
            self.conflict_penalties = {
                'trend_resistance_conflict': 0.6,     # 트렌드 vs 저항 충돌
                'timeframe_divergence': 0.7,          # 시간대 간 분기
                'squeeze_breakout_conflict': 0.8,     # 스퀴즈 방향성 충돌
                'ema_trend_conflict': 0.75            # EMA vs 트렌드 충돌
            }


class MediumTermSynergyEngine:
    """중기 전략 시너지 엔진 (1-4시간 보유)"""
    
    def __init__(self, config: MediumTermConfig = MediumTermConfig()):
        self.config = config
    
    def _convert_signal_format(self, signals) -> List[Dict[str, Any]]:
        """신호 형식 변환"""
        if isinstance(signals, dict):
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
            return signals
        else:
            return []
    
    def detect_market_context(self, signals: List[Dict[str, Any]]) -> str:
        """중기 시장 상황 감지"""
        signal_dict = {s['name']: s for s in signals}
        
        # 1. 트렌드 강도 분석
        htf_trend = signal_dict.get('HTF_TREND')
        multi_tf = signal_dict.get('MULTI_TIMEFRAME')
        
        trend_strength = 0
        if htf_trend and htf_trend['score'] > self.config.trend_strength_threshold:
            trend_strength += htf_trend['score']
        if multi_tf and multi_tf['score'] > self.config.trend_strength_threshold:
            trend_strength += multi_tf['score']
        
        # 2. 지지/저항 강도 분석  
        sr = signal_dict.get('SUPPORT_RESISTANCE')
        sr_strength = sr['score'] if sr and sr['score'] > self.config.support_resistance_threshold else 0
        
        # 3. 볼린저 스퀴즈 상태
        squeeze = signal_dict.get('BOLLINGER_SQUEEZE')
        is_squeezing = squeeze and squeeze['score'] > 0.7
        
        # 4. EMA 수렴/발산
        ema = signal_dict.get('EMA_CONFLUENCE')
        ema_aligned = ema and ema['score'] > 0.7
        
        # 상황 판단
        if trend_strength >= 1.4 and ema_aligned:
            return 'STRONG_TREND'
        elif sr_strength > 0.8 and trend_strength < 0.8:
            return 'CONSOLIDATION'
        elif is_squeezing and sr_strength > 0.7:
            return 'BREAKOUT_PENDING'
        elif sr_strength > 0.8 and trend_strength > 0.8:
            return 'REVERSAL_ZONE'
        else:
            return 'NEUTRAL'
    
    def apply_context_weights(self, signals: List[Dict[str, Any]], context: str) -> List[Dict[str, Any]]:
        """시장 상황별 가중치 조정"""
        multipliers = self.config.market_context_multipliers.get(context, {})
        
        adjusted_signals = []
        for signal in signals:
            mult = multipliers.get(signal['name'], 1.0)
            adjusted_signal = signal.copy()
            adjusted_signal['context_adjusted_score'] = signal['score'] * mult
            adjusted_signals.append(adjusted_signal)
            
        return adjusted_signals
    
    def detect_conflicts(self, signals: List[Dict[str, Any]]) -> Dict[str, float]:
        """중기 전략 간 충돌 감지"""
        conflicts = {}
        signal_dict = {s['name']: s for s in signals}
        
        # 1. 트렌드 vs 지지저항 충돌
        htf_trend = signal_dict.get('HTF_TREND')
        sr = signal_dict.get('SUPPORT_RESISTANCE')
        if (htf_trend and sr and htf_trend['action'] != sr['action'] and 
            htf_trend['score'] > 0.6 and sr['score'] > 0.6):
            conflicts['trend_resistance_conflict'] = min(htf_trend['score'], sr['score'])
        
        # 2. 다중 시간대 분기
        multi_tf = signal_dict.get('MULTI_TIMEFRAME')
        if htf_trend and multi_tf and htf_trend['action'] != multi_tf['action']:
            conflicts['timeframe_divergence'] = min(htf_trend['score'], multi_tf['score'])
        
        # 3. 스퀴즈 vs 기타 방향성 충돌
        squeeze = signal_dict.get('BOLLINGER_SQUEEZE')
        if squeeze and squeeze['score'] > 0.7:
            opposing_signals = [s for s in signals if s['name'] != 'BOLLINGER_SQUEEZE' 
                              and s['action'] != squeeze['action'] and s['score'] > 0.6]
            if len(opposing_signals) >= 2:
                avg_opposing_score = np.mean([s['score'] for s in opposing_signals])
                conflicts['squeeze_breakout_conflict'] = min(squeeze['score'], avg_opposing_score)
        
        # 4. EMA vs 트렌드 충돌
        ema = signal_dict.get('EMA_CONFLUENCE')
        if htf_trend and ema and htf_trend['action'] != ema['action']:
            conflicts['ema_trend_conflict'] = min(htf_trend['score'], ema['score'])
        
        return conflicts
    
    def calculate_synergy_bonuses(self, signals: List[Dict[str, Any]]) -> Dict[str, float]:
        """중기 시너지 보너스 계산"""
        bonuses = {}
        signal_dict = {s['name']: s for s in signals}
        
        # 1. 트렌드 합류 보너스
        trend_signals = ['HTF_TREND', 'MULTI_TIMEFRAME', 'EMA_CONFLUENCE']
        trend_consensus = [signal_dict[name] for name in trend_signals if name in signal_dict]
        if len(trend_consensus) >= 2:
            same_direction = len(set(s['action'] for s in trend_consensus)) == 1
            if same_direction and all(s['score'] > 0.6 for s in trend_consensus):
                bonuses['trend_confluence'] = self.config.trend_confluence_bonus
        
        # 2. 지지/저항 확인 보너스
        sr = signal_dict.get('SUPPORT_RESISTANCE')
        if sr and sr['score'] > 0.8:
            confirming_signals = [s for s in signals if s['name'] != 'SUPPORT_RESISTANCE' 
                                and s['action'] == sr['action'] and s['score'] > 0.6]
            if len(confirming_signals) >= 1:
                bonuses['support_resistance'] = self.config.support_resistance_bonus
        
        # 3. 시간대 정렬 보너스
        htf = signal_dict.get('HTF_TREND')
        multi = signal_dict.get('MULTI_TIMEFRAME')
        if (htf and multi and htf['action'] == multi['action'] and 
            htf['score'] > 0.7 and multi['score'] > 0.7):
            bonuses['timeframe_alignment'] = self.config.timeframe_alignment_bonus
        
        return bonuses
    
    def calculate_synergy_score(self, signals) -> Dict[str, Any]:
        """중기 시너지 점수 계산"""
        signals = self._convert_signal_format(signals)
        
        if not signals:
            return self._create_hold_result('MEDIUM_TERM')
        
        # 1. 시장 상황 감지
        market_context = self.detect_market_context(signals)
        
        # 2. 상황별 가중치 적용
        adjusted_signals = self.apply_context_weights(signals, market_context)
        
        # 3. 최소 임계값 필터링 (중기는 0.3 이상으로 완화)
        filtered_signals = [s for s in adjusted_signals if s['score'] >= 0.3]
        
        if not filtered_signals:
            return self._create_hold_result('MEDIUM_TERM', market_context)
        
        # 4. 충돌 감지 및 페널티 적용
        conflicts = self.detect_conflicts(filtered_signals)
        
        # 5. 시너지 보너스 계산
        bonuses = self.calculate_synergy_bonuses(filtered_signals)
        
        # 6. 최종 점수 계산
        weighted_signals = []
        for signal in filtered_signals:
            base_weight = self.config.confidence_weights.get(signal['name'], 1.0)
            context_score = signal.get('context_adjusted_score', signal['score'])
            
            # 충돌 페널티 적용
            penalty = 1.0
            for conflict_type, conflict_strength in conflicts.items():
                if signal['name'] in self._get_affected_strategies(conflict_type):
                    penalty *= self.config.conflict_penalties[conflict_type]
            
            final_score = context_score * base_weight * penalty
            weighted_signals.append({
                **signal,
                'final_score': final_score,
                'penalty_applied': penalty
            })
        
        # 7. 방향별 점수 집계
        buy_signals = [s for s in weighted_signals if s['action'] == 'BUY']
        sell_signals = [s for s in weighted_signals if s['action'] == 'SELL']
        
        buy_score = np.mean([s['final_score'] for s in buy_signals]) if buy_signals else 0.0
        sell_score = np.mean([s['final_score'] for s in sell_signals]) if sell_signals else 0.0
        
        # 8. 시너지 보너스 적용
        total_bonus_applied = 0
        for bonus_type, bonus_value in bonuses.items():
            if buy_score > sell_score:
                buy_score *= (1 + bonus_value)
                total_bonus_applied += bonus_value
            else:
                sell_score *= (1 + bonus_value)
                total_bonus_applied += bonus_value
        
        # 9. 최종 결과
        net_score = buy_score - sell_score
        return self._create_final_result(net_score, buy_score, sell_score, market_context, 
                                       conflicts, bonuses, filtered_signals, weighted_signals,
                                       total_bonus_applied, 'MEDIUM_TERM')
    
    def _get_affected_strategies(self, conflict_type: str) -> List[str]:
        """충돌 유형별 영향받는 전략들"""
        mapping = {
            'oi_funding_divergence': ['OI_DELTA', 'FUNDING_RATE'],
            'vpvr_ichimoku_conflict': ['VPVR', 'ICHIMOKU'],
            'institutional_retail_split': ['OI_DELTA', 'VPVR', 'FUNDING_RATE'],
            'macro_micro_conflict': ['ICHIMOKU', 'VPVR', 'OI_DELTA', 'FUNDING_RATE']
        }
        return mapping.get(conflict_type, [])
    
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
        # 임계값: 중기는 0.25
        min_threshold = 0.25
        
        if abs(net_score) < min_threshold:
            action = 'HOLD'
            confidence = 'LOW'
        else:
            action = 'BUY' if net_score > 0 else 'SELL'
            if abs(net_score) > 0.6:
                confidence = 'HIGH'
            elif abs(net_score) > 0.4:
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
        trend_signals = ['HTF_TREND', 'MULTI_TIMEFRAME']
        trend_scores = [s['final_score'] for s in weighted_signals if s['name'] in trend_signals]
        
        if not trend_scores:
            return 'WEAK'
        
        avg_trend_score = np.mean(trend_scores)
        
        if context == 'STRONG_TREND' and avg_trend_score > 0.8:
            return 'STRONG'
        elif avg_trend_score > 0.6:
            return 'MEDIUM'
        else:
            return 'WEAK'
    
    def _calculate_consolidation_level(self, weighted_signals: List[Dict[str, Any]], context: str) -> str:
        """통합 수준 계산"""
        sr_signal = next((s for s in weighted_signals if s['name'] == 'SUPPORT_RESISTANCE'), None)
        
        if not sr_signal:
            return 'NEUTRAL'
        
        if context == 'CONSOLIDATION' and sr_signal['final_score'] > 0.8:
            return 'HIGH'
        elif sr_signal['final_score'] > 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
