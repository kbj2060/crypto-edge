# long_term_synergy.py
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime, timedelta

@dataclass 
class LongTermConfig:
    """장기 전략 설정 (4-24시간 보유)"""
    # 신뢰도 기반 가중치
    confidence_weights: Dict[str, float] = None
    
    # 시장 상황별 가중치 배수
    market_context_multipliers: Dict[str, Dict[str, float]] = None
    
    # 충돌 페널티
    conflict_penalties: Dict[str, float] = None
    
    # 임계값들
    oi_delta_threshold: float = 0.8
    funding_rate_threshold: float = 0.75
    ichimoku_cloud_threshold: float = 0.7
    vpvr_level_threshold: float = 0.75
    
    # 시너지 보너스
    institutional_flow_bonus: float = 0.3      # 기관 자금 흐름 일치
    macro_alignment_bonus: float = 0.25        # 거시적 정렬
    cloud_confirmation_bonus: float = 0.2      # 이치모쿠 클라우드 확인
    volume_profile_bonus: float = 0.2          # 거래량 프로파일 지지

    def __post_init__(self):
        if self.confidence_weights is None:
            self.confidence_weights = {
                'OI_DELTA': 1.6,            # 기관 자금 흐름, 가장 신뢰도 높음
                'VPVR': 1.4,               # 거래량 프로파일, 장기 지지/저항
                'ICHIMOKU': 1.3,           # 종합적 기술 분석
                'FUNDING_RATE': 1.2        # 시장 심리, 보조 지표
            }
        
        if self.market_context_multipliers is None:
            self.market_context_multipliers = {
                'INSTITUTIONAL_ACCUMULATION': {    # 기관 매집 단계
                    'OI_DELTA': 1.5,              # 기관 자금 흐름 매우 중요
                    'VPVR': 1.3,                  # 거래량 축적 중요
                    'ICHIMOKU': 1.1,              # 기술적 분석 보조
                    'FUNDING_RATE': 0.9           # 개인 심리는 상대적 덜 중요
                },
                'DISTRIBUTION_PHASE': {           # 분산 단계
                    'OI_DELTA': 1.4,              # 기관 자금 유출 감지
                    'FUNDING_RATE': 1.3,          # 개인 투자자 심리 중요
                    'VPVR': 1.2,                  # 거래량 분산 패턴
                    'ICHIMOKU': 1.0               # 기술적 분석 보조
                },
                'MACRO_TREND': {                  # 거시적 트렌드
                    'ICHIMOKU': 1.4,              # 클라우드 트렌드 매우 중요
                    'VPVR': 1.3,                  # 거래량 지지/저항
                    'OI_DELTA': 1.2,              # 기관 방향성
                    'FUNDING_RATE': 1.1           # 시장 심리 확인
                },
                'RANGE_BOUND': {                  # 박스권 시장
                    'VPVR': 1.4,                  # 거래량 기반 지지/저항 핵심
                    'FUNDING_RATE': 1.2,          # 극단적 심리 반전 신호
                    'ICHIMOKU': 1.1,              # 클라우드 내 움직임
                    'OI_DELTA': 0.9               # 기관 활동 제한적
                },
                'VOLATILITY_EXPANSION': {         # 변동성 확대
                    'OI_DELTA': 1.3,              # 기관 포지션 변화
                    'ICHIMOKU': 1.2,              # 클라우드 이탈
                    'FUNDING_RATE': 1.1,          # 심리적 극단
                    'VPVR': 1.0                   # 거래량 급증
                }
            }
        
        if self.conflict_penalties is None:
            self.conflict_penalties = {
                'oi_funding_divergence': 0.7,        # OI와 펀딩비 분기 (기관 vs 개인)
                'vpvr_ichimoku_conflict': 0.75,      # 거래량 vs 기술적 분석 충돌
                'institutional_retail_split': 0.6,   # 기관 vs 개인 투자자 분기
                'macro_micro_conflict': 0.8,         # 거시 vs 미시 충돌
                'cloud_volume_divergence': 0.8       # 클라우드 vs 거래량 분기
            }

class LongTermSynergyEngine:
    """장기 전략 시너지 엔진 (4-24시간 보유)
    
    기관 투자자 행동, 거시적 트렌드, 장기 지지/저항에 중점
    """
    
    def __init__(self, config: LongTermConfig = LongTermConfig()):
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
        """장기 시장 상황 감지
        
        기관 자금 흐름, 거래량 프로파일, 기술적 분석을 종합하여
        장기적 시장 상황을 판단
        """
        signal_dict = {s['name']: s for s in signals}
        
        # 1. 기관 자금 흐름 분석 (OI_DELTA)
        oi_delta = signal_dict.get('OI_DELTA')
        institutional_flow_strength = 0
        institutional_direction = None
        
        if oi_delta and oi_delta['score'] > self.config.oi_delta_threshold:
            institutional_flow_strength = oi_delta['score']
            institutional_direction = oi_delta['action']
        
        # 2. 펀딩비 상태 분석 (시장 심리)
        funding = signal_dict.get('FUNDING_RATE')
        funding_extreme = False
        funding_direction = None
        
        if funding and funding['score'] > self.config.funding_rate_threshold:
            funding_extreme = True
            funding_direction = funding['action']
        
        # 3. VPVR 레벨 강도 (거래량 기반 지지/저항)
        vpvr = signal_dict.get('VPVR')
        volume_support_strength = 0
        volume_direction = None
        
        if vpvr and vpvr['score'] > self.config.vpvr_level_threshold:
            volume_support_strength = vpvr['score']
            volume_direction = vpvr['action']
        
        # 4. 이치모쿠 클라우드 상태
        ichimoku = signal_dict.get('ICHIMOKU')
        cloud_clear = False
        cloud_direction = None
        
        if ichimoku and ichimoku['score'] > self.config.ichimoku_cloud_threshold:
            cloud_clear = True
            cloud_direction = ichimoku['action']
        
        # 5. 종합 상황 판단
        # 기관 매집 단계: 강한 OI 유입 + 펀딩비 정상
        if (institutional_flow_strength > 0.8 and 
            institutional_direction == 'BUY' and 
            not funding_extreme):
            return 'INSTITUTIONAL_ACCUMULATION'
        
        # 분산 단계: 강한 OI 유출 + 극단적 펀딩비
        elif (institutional_flow_strength > 0.8 and 
              institutional_direction == 'SELL' and 
              funding_extreme):
            return 'DISTRIBUTION_PHASE'
        
        # 거시적 트렌드: 클라우드 명확 + 거래량 지지
        elif (cloud_clear and volume_support_strength > 0.7 and 
              cloud_direction == volume_direction):
            return 'MACRO_TREND'
        
        # 박스권: 강한 거래량 지지/저항 + 클라우드 불명확
        elif (volume_support_strength > 0.8 and not cloud_clear):
            return 'RANGE_BOUND'
        
        # 변동성 확대: 기관 활동 + 클라우드 이탈 + 극단 심리
        elif (institutional_flow_strength > 0.7 and 
              funding_extreme and 
              cloud_clear):
            return 'VOLATILITY_EXPANSION'
        
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
            adjusted_signal['context_multiplier'] = mult
            adjusted_signals.append(adjusted_signal)
            
        return adjusted_signals
    
    def detect_conflicts(self, signals: List[Dict[str, Any]]) -> Dict[str, float]:
        """장기 전략 간 충돌 감지"""
        conflicts = {}
        signal_dict = {s['name']: s for s in signals}
        
        # 1. OI vs 펀딩비 분기 (기관 vs 개인 심리)
        oi = signal_dict.get('OI_DELTA')
        funding = signal_dict.get('FUNDING_RATE')
        
        if (oi and funding and 
            oi['action'] != funding['action'] and 
            oi['score'] > 0.6 and funding['score'] > 0.6):
            conflicts['oi_funding_divergence'] = min(oi['score'], funding['score'])
        
        # 2. VPVR vs 이치모쿠 충돌 (거래량 vs 기술적 분석)
        vpvr = signal_dict.get('VPVR')
        ichimoku = signal_dict.get('ICHIMOKU')
        
        if (vpvr and ichimoku and 
            vpvr['action'] != ichimoku['action'] and 
            vpvr['score'] > 0.6 and ichimoku['score'] > 0.6):
            conflicts['vpvr_ichimoku_conflict'] = min(vpvr['score'], ichimoku['score'])
        
        # 3. 기관 vs 개인 투자자 전반적 분기
        institutional_signals = ['OI_DELTA', 'VPVR']  # 기관 관련
        retail_signals = ['FUNDING_RATE']             # 개인 관련
        
        inst_actions = []
        retail_actions = []
        
        for name in institutional_signals:
            if name in signal_dict and signal_dict[name]['score'] > 0.6:
                inst_actions.append(signal_dict[name]['action'])
        
        for name in retail_signals:
            if name in signal_dict and signal_dict[name]['score'] > 0.6:
                retail_actions.append(signal_dict[name]['action'])
        
        if inst_actions and retail_actions:
            # 기관 컨센서스
            inst_consensus = max(set(inst_actions), key=inst_actions.count) if len(set(inst_actions)) == 1 else None
            # 개인 컨센서스  
            retail_consensus = max(set(retail_actions), key=retail_actions.count) if len(set(retail_actions)) == 1 else None
            
            if (inst_consensus and retail_consensus and 
                inst_consensus != retail_consensus):
                avg_inst_score = np.mean([signal_dict[name]['score'] for name in institutional_signals if name in signal_dict])
                avg_retail_score = np.mean([signal_dict[name]['score'] for name in retail_signals if name in signal_dict])
                conflicts['institutional_retail_split'] = min(avg_inst_score, avg_retail_score)
        
        # 4. 거시 vs 미시 충돌
        macro_signals = ['ICHIMOKU', 'VPVR']     # 거시적
        micro_signals = ['OI_DELTA', 'FUNDING_RATE']  # 미시적
        
        macro_strong = [signal_dict[name] for name in macro_signals 
                       if name in signal_dict and signal_dict[name]['score'] > 0.6]
        micro_strong = [signal_dict[name] for name in micro_signals 
                       if name in signal_dict and signal_dict[name]['score'] > 0.6]
        
        if len(macro_strong) >= 1 and len(micro_strong) >= 1:
            macro_actions = [s['action'] for s in macro_strong]
            micro_actions = [s['action'] for s in micro_strong]
            
            # 방향이 혼재되어 있는 경우
            all_actions = macro_actions + micro_actions
            if len(set(all_actions)) > 1:
                conflicts['macro_micro_conflict'] = 0.6
        
        # 5. 클라우드 vs 거래량 분기
        if (ichimoku and vpvr and 
            ichimoku['action'] != vpvr['action'] and 
            ichimoku['score'] > 0.7 and vpvr['score'] > 0.7):
            conflicts['cloud_volume_divergence'] = min(ichimoku['score'], vpvr['score'])
        
        return conflicts
    
    def calculate_synergy_bonuses(self, signals: List[Dict[str, Any]]) -> Dict[str, float]:
        """장기 시너지 보너스 계산"""
        bonuses = {}
        signal_dict = {s['name']: s for s in signals}
        
        # 1. 기관 자금 흐름 일치 보너스
        oi = signal_dict.get('OI_DELTA')
        vpvr = signal_dict.get('VPVR')
        
        if (oi and vpvr and 
            oi['action'] == vpvr['action'] and 
            oi['score'] > 0.8 and vpvr['score'] > 0.7):
            bonuses['institutional_flow'] = self.config.institutional_flow_bonus
        
        # 2. 거시적 정렬 보너스 (이치모쿠 + 기타 확인)
        ichimoku = signal_dict.get('ICHIMOKU')
        if ichimoku and ichimoku['score'] > 0.8:
            aligned_signals = [s for s in signals 
                             if (s['name'] != 'ICHIMOKU' and 
                                 s['action'] == ichimoku['action'] and 
                                 s['score'] > 0.6)]
            if len(aligned_signals) >= 2:
                bonuses['macro_alignment'] = self.config.macro_alignment_bonus
        
        # 3. 클라우드 확인 보너스 (이치모쿠 매우 강함 + 확인)
        if ichimoku and ichimoku['score'] > 0.9:
            strong_confirmations = [s for s in signals 
                                  if (s['name'] != 'ICHIMOKU' and 
                                      s['action'] == ichimoku['action'] and 
                                      s['score'] > 0.7)]
            if len(strong_confirmations) >= 1:
                bonuses['cloud_confirmation'] = self.config.cloud_confirmation_bonus
        
        # 4. 거래량 프로파일 지지 보너스
        if vpvr and vpvr['score'] > 0.85:
            volume_confirmations = [s for s in signals 
                                  if (s['name'] != 'VPVR' and 
                                      s['action'] == vpvr['action'] and 
                                      s['score'] > 0.6)]
            if len(volume_confirmations) >= 1:
                bonuses['volume_profile'] = self.config.volume_profile_bonus
        
        return bonuses
    
    def calculate_synergy_score(self, signals) -> Dict[str, Any]:
        """장기 시너지 점수 계산"""
        signals = self._convert_signal_format(signals)
        
        if not signals:
            return self._create_hold_result('LONG_TERM')
        
        # 1. 시장 상황 감지
        market_context = self.detect_market_context(signals)
        
        # 2. 상황별 가중치 적용
        adjusted_signals = self.apply_context_weights(signals, market_context)
        
        # 3. 최소 임계값 필터링 (장기는 0.6 이상)
        filtered_signals = [s for s in adjusted_signals if s['score'] >= 0.6]
        
        if not filtered_signals:
            return self._create_hold_result('LONG_TERM', market_context)
        
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
                'penalty_applied': penalty,
                'base_weight': base_weight
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
                                       total_bonus_applied, 'LONG_TERM')
    
    def _get_affected_strategies(self, conflict_type: str) -> List[str]:
        """충돌 유형별 영향받는 전략들"""
        mapping = {
            'oi_funding_divergence': ['OI_DELTA', 'FUNDING_RATE'],
            'vpvr_ichimoku_conflict': ['VPVR', 'ICHIMOKU'],
            'institutional_retail_split': ['OI_DELTA', 'VPVR', 'FUNDING_RATE'],
            'macro_micro_conflict': ['ICHIMOKU', 'VPVR', 'OI_DELTA', 'FUNDING_RATE'],
            'cloud_volume_divergence': ['ICHIMOKU', 'VPVR']
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
            }
        }
    
    def _create_final_result(self, net_score: float, buy_score: float, sell_score: float,
                           context: str, conflicts: Dict, bonuses: Dict, signals: List, 
                           weighted_signals: List, total_bonus: float, category: str) -> Dict[str, Any]:
        """최종 결과 생성"""
        # 임계값: 장기는 0.25 (보수적)
        min_threshold = 0.25
        
        if abs(net_score) < min_threshold:
            action = 'HOLD'
            confidence = 'LOW'
        else:
            action = 'BUY' if net_score > 0 else 'SELL'
            # 장기 신뢰도는 더 엄격하게
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
                'buy_signals': [{'name': s['name'], 'score': s['final_score'], 
                               'base_weight': s['base_weight'], 'penalty': s['penalty_applied']} 
                               for s in weighted_signals if s['action'] == 'BUY'],
                'sell_signals': [{'name': s['name'], 'score': s['final_score'],
                                'base_weight': s['base_weight'], 'penalty': s['penalty_applied']} 
                                for s in weighted_signals if s['action'] == 'SELL']
            },
            'meta': {
                'total_bonus_applied': total_bonus,
                'conflict_count': len(conflicts),
                'strong_signals': [s['name'] for s in signals if s['score'] > 0.8],
                'institutional_bias': self._calculate_institutional_bias(weighted_signals),
                'macro_trend_strength': self._calculate_macro_strength(weighted_signals, context)
            }
        }
    
    def _calculate_institutional_bias(self, weighted_signals: List[Dict[str, Any]]) -> str:
        """기관 투자자 편향 계산"""
        institutional_strategies = ['OI_DELTA', 'VPVR']
        inst_signals = [s for s in weighted_signals if s['name'] in institutional_strategies]
        
        if not inst_signals:
            return 'NEUTRAL'
        
        buy_inst = [s for s in inst_signals if s['action'] == 'BUY']
        sell_inst = [s for s in inst_signals if s['action'] == 'SELL']
        
        if len(buy_inst) > len(sell_inst):
            avg_buy_score = np.mean([s['final_score'] for s in buy_inst])
            return 'BULLISH' if avg_buy_score > 0.7 else 'WEAK_BULLISH'
        elif len(sell_inst) > len(buy_inst):
            avg_sell_score = np.mean([s['final_score'] for s in sell_inst])
            return 'BEARISH' if avg_sell_score > 0.7 else 'WEAK_BEARISH'
        else:
            return 'NEUTRAL'
    
    def _calculate_macro_strength(self, weighted_signals: List[Dict[str, Any]], context: str) -> str:
        """거시적 트렌드 강도 계산"""
        ichimoku_signal = next((s for s in weighted_signals if s['name'] == 'ICHIMOKU'), None)
        
        if not ichimoku_signal:
            return 'WEAK'
        
        if context in ['MACRO_TREND', 'INSTITUTIONAL_ACCUMULATION'] and ichimoku_signal['final_score'] > 0.8:
            return 'STRONG'
        elif ichimoku_signal['final_score'] > 0.6:
            return 'MEDIUM'
        else:
            return 'WEAK'

