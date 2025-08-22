#!/usr/bin/env python3
"""
VWAP (Volume Weighted Average Price) 향상 지표
- 세션 앵커드 VWAP
- VWAP 편차의 표준편차
- 동적 바닥값 적용
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional


def calculate_session_vwap(df: pd.DataFrame, 
                          session_start_time: int,
                          current_time: int) -> Tuple[float, float]:
    """
    세션 앵커드 VWAP 계산
    
    Args:
        df: 3분봉 OHLCV 데이터 (timestamp, open, high, low, close, volume)
        session_start_time: 세션 시작 시간 (timestamp)
        current_time: 현재 시간 (timestamp)
    
    Returns:
        Tuple[float, float]: (VWAP, 누적 거래량)
    """
    try:
        if df.empty:
            return 0.0, 0.0
        
        # 세션 시작부터 현재까지의 데이터 필터링
        session_data = df[
            (df['timestamp'] >= session_start_time) & 
            (df['timestamp'] <= current_time)
        ]
        
        if session_data.empty:
            return 0.0, 0.0
        
        # Typical Price 계산: (H + L + C) / 3
        session_data = session_data.copy()
        session_data['typical_price'] = (session_data['high'] + session_data['low'] + session_data['close']) / 3
        
        # VWAP 계산: Σ(typical_price × volume) / Σ(volume)
        price_volume_sum = (session_data['typical_price'] * session_data['volume']).sum()
        volume_sum = session_data['volume'].sum()
        
        if volume_sum == 0:
            return 0.0, 0.0
        
        vwap = price_volume_sum / volume_sum
        
        return float(vwap), float(volume_sum)
        
    except Exception as e:
        print(f"❌ 세션 VWAP 계산 오류: {e}")
        return 0.0, 0.0


def calculate_vwap_series(df: pd.DataFrame, 
                         session_start_time: int,
                         current_time: int) -> pd.Series:
    """
    VWAP 시계열 계산 (각 시점의 VWAP)
    
    Args:
        df: 3분봉 OHLCV 데이터
        session_start_time: 세션 시작 시간 (timestamp)
        current_time: 현재 시간 (timestamp)
    
    Returns:
        pd.Series: VWAP 시계열
    """
    try:
        if df.empty:
            return pd.Series(dtype=float)
        
        # 세션 데이터 필터링
        session_data = df[
            (df['timestamp'] >= session_start_time) & 
            (df['timestamp'] <= current_time)
        ].copy()
        
        if session_data.empty:
            return pd.Series(dtype=float)
        
        # Typical Price 계산
        session_data['typical_price'] = (session_data['high'] + session_data['low'] + session_data['close']) / 3
        
        # 누적 계산
        session_data['cumulative_pv'] = (session_data['typical_price'] * session_data['volume']).cumsum()
        session_data['cumulative_volume'] = session_data['volume'].cumsum()
        
        # VWAP 시계열 계산
        vwap_series = session_data['cumulative_pv'] / session_data['cumulative_volume']
        
        return vwap_series
        
    except Exception as e:
        print(f"❌ VWAP 시계열 계산 오류: {e}")
        return pd.Series(dtype=float)


def calculate_vwap_std(df: pd.DataFrame, 
                      session_start_time: int,
                      current_time: int,
                      lookback: int = 120,
                      min_std_pct: float = 0.001) -> Tuple[float, float, float]:
    """
    VWAP와 표준편차 계산
    
    Args:
        df: 3분봉 OHLCV 데이터
        session_start_time: 세션 시작 시간 (timestamp)
        current_time: 현재 시간 (timestamp)
        lookback: 표준편차 계산용 룩백 기간 (기본: 120봉 ≈ 6시간)
        min_std_pct: 최소 표준편차 비율 (기본: 0.1%)
    
    Returns:
        Tuple[float, float, float]: (VWAP, VWAP 표준편차, 현재가)
    """
    try:
        if df.empty:
            return 0.0, 0.0, 0.0
        
        # VWAP 시계열 계산
        vwap_series = calculate_vwap_series(df, session_start_time, current_time)
        
        if vwap_series.empty:
            return 0.0, 0.0, 0.0
        
        # 현재 VWAP
        current_vwap = float(vwap_series.iloc[-1])
        
        # 현재가 (마지막 캔들의 종가)
        session_data = df[
            (df['timestamp'] >= session_start_time) & 
            (df['timestamp'] <= current_time)
        ]
        
        if session_data.empty:
            return current_vwap, 0.0, 0.0
        
        current_price = float(session_data['close'].iloc[-1])
        
        # VWAP 대비 편차 계산
        price_data = session_data['close']
        deviations = price_data - vwap_series
        
        # 표준편차 계산 (최근 lookback 기간)
        if len(deviations) >= lookback:
            recent_deviations = deviations.tail(lookback)
        else:
            recent_deviations = deviations
        
        # NaN 제거
        recent_deviations = recent_deviations.dropna()
        
        if len(recent_deviations) == 0:
            # 기본값: 현재가의 0.5%
            default_std = current_price * 0.005
            return current_vwap, default_std, current_price
        
        # 표준편차 계산 (ddof=0: 모집단 정의)
        std = float(recent_deviations.std(ddof=0))
        
        # 최소 바닥값 적용
        min_std = current_price * min_std_pct
        final_std = max(std, min_std)
        
        return current_vwap, final_std, current_price
        
    except Exception as e:
        print(f"❌ VWAP 표준편차 계산 오류: {e}")
        return 0.0, 0.0, 0.0


def get_vwap_levels(df: pd.DataFrame, 
                    session_start_time: int,
                    current_time: int,
                    lookback: int = 120) -> Dict[str, float]:
    """
    VWAP 관련 모든 레벨 계산
    
    Args:
        df: 3분봉 OHLCV 데이터
        session_start_time: 세션 시작 시간 (timestamp)
        current_time: 현재 시간 (timestamp)
        lookback: 표준편차 계산용 룩백
    
    Returns:
        Dict[str, float]: VWAP 레벨 정보
    """
    try:
        # VWAP와 표준편차 계산
        vwap, vwap_std, current_price = calculate_vwap_std(
            df, session_start_time, current_time, lookback
        )
        
        if vwap == 0 or vwap_std == 0:
            return {}
        
        # VWAP 밴드 레벨
        vwap_upper_1 = vwap + vwap_std
        vwap_upper_2 = vwap + 2 * vwap_std
        vwap_upper_3 = vwap + 3 * vwap_std
        
        vwap_lower_1 = vwap - vwap_std
        vwap_lower_2 = vwap - 2 * vwap_std
        vwap_lower_3 = vwap - 3 * vwap_std
        
        # 현재가 대비 VWAP 거리
        vwap_distance = current_price - vwap
        vwap_distance_pct = (vwap_distance / vwap) * 100 if vwap > 0 else 0
        
        # VWAP 대비 현재가 위치 (표준편차 단위)
        vwap_position_sigma = vwap_distance / vwap_std if vwap_std > 0 else 0
        
        result = {
            'vwap': vwap,
            'vwap_std': vwap_std,
            'current_price': current_price,
            'vwap_upper_1': vwap_upper_1,
            'vwap_upper_2': vwap_upper_2,
            'vwap_upper_3': vwap_upper_3,
            'vwap_lower_1': vwap_lower_1,
            'vwap_lower_2': vwap_lower_2,
            'vwap_lower_3': vwap_lower_3,
            'vwap_distance': vwap_distance,
            'vwap_distance_pct': vwap_distance_pct,
            'vwap_position_sigma': vwap_position_sigma,
            'is_above_vwap': current_price > vwap,
            'is_below_vwap': current_price < vwap
        }
        
        return result
        
    except Exception as e:
        print(f"❌ VWAP 레벨 계산 오류: {e}")
        return {}


def get_vwap_breakout_status(vwap_levels: Dict[str, float], 
                            tolerance_sigma: float = 0.1) -> Dict[str, any]:
    """
    VWAP 돌파 상태 확인
    
    Args:
        vwap_levels: VWAP 레벨 정보
        tolerance_sigma: 돌파 확인용 허용 오차 (표준편차 단위)
    
    Returns:
        Dict: VWAP 돌파 상태
    """
    try:
        if not vwap_levels:
            return {}
        
        current_price = vwap_levels.get('current_price', 0)
        vwap = vwap_levels.get('vwap', 0)
        vwap_std = vwap_levels.get('vwap_std', 0)
        position_sigma = vwap_levels.get('vwap_position_sigma', 0)
        
        if vwap == 0 or vwap_std == 0:
            return {}
        
        # 돌파 상태 확인
        is_above_1sigma = position_sigma > 1 + tolerance_sigma
        is_above_2sigma = position_sigma > 2 + tolerance_sigma
        is_above_3sigma = position_sigma > 3 + tolerance_sigma
        
        is_below_1sigma = position_sigma < -1 - tolerance_sigma
        is_below_2sigma = position_sigma < -2 - tolerance_sigma
        is_below_3sigma = position_sigma < -3 - tolerance_sigma
        
        # 돌파 강도
        breakout_strength = abs(position_sigma)
        
        # 돌파 방향
        if position_sigma > tolerance_sigma:
            breakout_direction = 'ABOVE'
        elif position_sigma < -tolerance_sigma:
            breakout_direction = 'BELOW'
        else:
            breakout_direction = 'INSIDE'
        
        result = {
            'current_price': current_price,
            'vwap': vwap,
            'vwap_std': vwap_std,
            'position_sigma': position_sigma,
            'is_above_1sigma': is_above_1sigma,
            'is_above_2sigma': is_above_2sigma,
            'is_above_3sigma': is_above_3sigma,
            'is_below_1sigma': is_below_1sigma,
            'is_below_2sigma': is_below_2sigma,
            'is_below_3sigma': is_below_3sigma,
            'breakout_strength': breakout_strength,
            'breakout_direction': breakout_direction,
            'is_extreme': breakout_strength > 3  # 3σ 이상은 극단적
        }
        
        return result
        
    except Exception as e:
        print(f"❌ VWAP 돌파 상태 확인 오류: {e}")
        return {}
