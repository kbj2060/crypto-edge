"""
데이터 평면화 유틸리티
- decision_data를 Parquet 저장에 최적화된 형태로 변환
- 여러 모듈에서 공통으로 사용
"""

import numpy as np
from typing import Dict, Any


def flatten_decision_data(decision_data: Dict[str, Any]) -> Dict[str, Any]:
    """복잡한 중첩 구조를 평면화하여 Parquet 저장에 최적화"""
    
    def safe_convert(value):
        """numpy 타입을 Python 기본 타입으로 안전하게 변환"""
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value
    
    flattened = {}
    
    # 기본 정보
    flattened['timestamp'] = decision_data.get('timestamp')
    
    # indicators 정보
    indicators = decision_data.get('indicators', {})
    for key, value in indicators.items():
        flattened[f'indicator_{key}'] = safe_convert(value)
    
    # decisions 정보를 각 카테고리별로 평면화
    decisions = decision_data.get('decisions', {})
    
    for category_name, category_data in decisions.items():
        prefix = f"{category_name.lower()}_"
        
        # RL 변환된 데이터에서 기본 정보 추출
        flattened[f'{prefix}action'] = safe_convert(category_data.get('action_value'))
        flattened[f'{prefix}net_score'] = safe_convert(category_data.get('net_score'))
        flattened[f'{prefix}leverage'] = safe_convert(category_data.get('leverage'))
        flattened[f'{prefix}max_holding_minutes'] = safe_convert(category_data.get('max_holding_minutes'))
        
        # 신뢰도 및 시장 상황
        flattened[f'{prefix}confidence'] = safe_convert(category_data.get('confidence_value'))
        flattened[f'{prefix}market_context'] = safe_convert(category_data.get('market_context_value'))
        
        # 점수 정보
        flattened[f'{prefix}buy_score'] = safe_convert(category_data.get('buy_score'))
        flattened[f'{prefix}sell_score'] = safe_convert(category_data.get('sell_score'))
        flattened[f'{prefix}signals_used'] = safe_convert(category_data.get('signals_used'))
        
        # 충돌 및 보너스 정보
        flattened[f'{prefix}conflicts_detected_count'] = safe_convert(category_data.get('conflicts_detected_count'))
        flattened[f'{prefix}bonuses_applied_count'] = safe_convert(category_data.get('bonuses_applied_count'))
        
        # 포지션 크기 정보
        flattened[f'{prefix}risk_multiplier'] = safe_convert(category_data.get('risk_multiplier'))
        flattened[f'{prefix}risk_usd'] = safe_convert(category_data.get('risk_usd'))
        
        # 전략 사용 정보
        flattened[f'{prefix}strategies_count'] = safe_convert(category_data.get('strategies_count'))
        
        # 카테고리별 특화 정보
        if category_name == 'short_term':
            flattened[f'{prefix}momentum_strength'] = safe_convert(category_data.get('momentum_strength'))
            flattened[f'{prefix}reversion_potential'] = safe_convert(category_data.get('reversion_potential'))
        elif category_name == 'medium_term':
            flattened[f'{prefix}trend_strength'] = safe_convert(category_data.get('trend_strength'))
            flattened[f'{prefix}consolidation_level'] = safe_convert(category_data.get('consolidation_level'))
        elif category_name == 'long_term':
            flattened[f'{prefix}institutional_bias'] = safe_convert(category_data.get('institutional_bias'))
            flattened[f'{prefix}macro_trend_strength'] = safe_convert(category_data.get('macro_trend_strength'))
    
    # conflicts 정보 (딕셔너리 형태로 처리)
    conflicts = decision_data.get('conflicts', {})
    if conflicts and isinstance(conflicts, dict):
        # conflicts의 각 키-값 쌍을 개별 컬럼으로 저장
        for key, value in conflicts.items():
            flattened[f'conflict_{key}'] = safe_convert(value)
    elif conflicts and isinstance(conflicts, list):
        # 리스트 형태인 경우
        flattened['conflicts_count'] = len(conflicts)
        flattened['conflicts_details'] = str(conflicts)
    
    return flattened


def is_flattened_data(data: Dict[str, Any]) -> bool:
    """데이터가 이미 평면화되었는지 확인"""
    # 평면화된 데이터의 특징: indicator_ 접두사가 있는 키들이 있음
    return any(key.startswith('indicator_') for key in data.keys())


def ensure_flattened_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """데이터가 평면화되지 않았다면 평면화하여 반환"""
    if is_flattened_data(data):
        return data
    else:
        return flatten_decision_data(data)
