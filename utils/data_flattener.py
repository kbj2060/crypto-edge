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
        """numpy 타입을 Python 기본 타입으로 안전하게 변환 (소수점 4자리 제한)"""
        if isinstance(value, (np.integer, np.floating)):
            return round(float(value), 4)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, float):
            return round(value, 4)
        return value
    
    flattened = {}
    
    # 기본 정보
    flattened['timestamp'] = decision_data.get('timestamp')
    
    # OHLC 데이터 추가
    ohlc_fields = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
    for field in ohlc_fields:
        if field in decision_data:
            flattened[field] = safe_convert(decision_data[field])
    
    # indicators 정보
    indicators = decision_data.get('indicators', {})
    for key, value in indicators.items():
        flattened[f'indicator_{key}'] = safe_convert(value)
    
    # 전략별 decision 정보를 평면화 (최상위 키들 중에서 전략 정보 찾기)
    strategy_keys = [k for k in decision_data.keys() if k not in ['timestamp', 'indicators', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'decisions']]
    
    for strategy_name in strategy_keys:
        strategy_data = decision_data.get(strategy_name, {})
        if isinstance(strategy_data, dict):
            prefix = f"{strategy_name.lower()}_"
            
            # 기본 전략 정보
            flattened[f'{prefix}action'] = safe_convert(strategy_data.get('action'))
            flattened[f'{prefix}score'] = safe_convert(strategy_data.get('score'))
            flattened[f'{prefix}confidence'] = safe_convert(strategy_data.get('confidence'))
            flattened[f'{prefix}entry'] = safe_convert(strategy_data.get('entry'))
            flattened[f'{prefix}stop'] = safe_convert(strategy_data.get('stop'))
    
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
