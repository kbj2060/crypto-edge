#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
전역 지표 관리자
- 모든 지표들을 중앙에서 관리
- 새로운 3분봉 데이터로 전체 지표 자동 업데이트
- 싱글톤 패턴으로 전역 접근
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import threading

import pandas as pd


# 지표 클래스들 import
from indicators.opening_range import OpeningRange
from indicators.vpvr import SessionVPVR
from indicators.atr import ATR3M
from indicators.daily_levels import DailyLevels
from indicators.vwap import SessionVWAP
from data.data_manager import get_data_manager


class GlobalIndicatorManager:
    """
    전역 지표 관리자
    - 모든 지표들을 중앙에서 관리
    - 새로운 3분봉 데이터로 전체 지표 자동 업데이트
    - 싱글톤 패턴으로 전역 접근
    """
    
    def __init__(self):
        """글로벌 지표 관리자 초기화"""
        self._indicators = {}
        self._initialized = False
        self._lock = threading.Lock()  # 스레드 안전성
        self._data_manager = None  # DataManager 인스턴스 (지연 초기화)
        
        # 지표 설정
        self.indicator_configs = {
            'vpvr': {
                'class': SessionVPVR,
                'bins': 50,
                'price_bin_size': 0.05,
                'lookback': 100
            },
            'atr': {
                'class': ATR3M,
                'length': 14,
                'max_candles': 100
            },
            'daily_levels': {
                'class': DailyLevels,
                'symbol': 'ETHUSDT',
            },
            'vwap': {
                'class': SessionVWAP,
                'symbol': 'ETHUSDT'
            },
            'opening_range': {
                'class': OpeningRange,
            }
        }
        

    
    def _initialize_indicator(self, name: str):
        """지표 초기화 - 공통 메서드"""
        try:
            config = self.indicator_configs[name]
            indicator_class = config['class']
            
            if name == 'vpvr':
                self._indicators[name] = indicator_class(
                    bins=config['bins'],
                    price_bin_size=config['price_bin_size'],
                    lookback=config['lookback']
                )
            elif name == 'atr':
                self._indicators[name] = indicator_class(
                    length=config['length'],
                    max_candles=config['max_candles']
                )
            elif name == 'vwap':
                self._indicators[name] = indicator_class(
                    symbol=config['symbol']
                )
            elif name == 'opening_range':
                self._indicators[name] = indicator_class(or_minutes=30)
            else:
                # 기본 초기화 (매개변수 없음)
                self._indicators[name] = indicator_class()
                
        except Exception as e:
            import traceback
            print(f"❌ {name} 지표 초기화 오류: {e}")
            print(f"❌ 상세 오류: {traceback.format_exc()}")
            self._indicators[name] = None

    def initialize_indicators(self):
        """모든 지표 초기화"""
        with self._lock:
            if self._initialized:
                return
            
            try:
                data_manager = self.get_data_manager()
                if not data_manager.is_ready():
                    return
                                
                # 🚀 2단계: 나머지 지표들 초기화 (DataManager 완료 후)
                print("\n🔥 2단계: 나머지 지표들 초기화 시작...")
                
                # 모든 지표를 순차적으로 초기화
                for indicator_name in self.indicator_configs.keys():
                    self._initialize_indicator(indicator_name)
                
                self._initialized = True
                print("🎯 모든 전역 지표 초기화 완료!")
            
            except Exception as e:
                self._initialized = False
    
    def update_all_indicators(self, candle_data: pd.Series):
        """
        새로운 3분봉 데이터로 모든 지표 업데이트
        
        Args:
            candle_data: 3분봉 캔들 데이터프레임 (1개 행) 
        """
        if not self._initialized:
            return

        # 1. ATR 업데이트 (가장 먼저 - 다른 지표들이 사용)
        if 'atr' in self._indicators and self._indicators['atr'] is not None:
            self._indicators['atr'].update_with_candle(candle_data)
        
        # 2. VPVR 업데이트
        if 'vpvr' in self._indicators and self._indicators['vpvr'] is not None:
            self._indicators['vpvr'].update_with_candle(candle_data)
        
        # 3. VWAP 업데이트
        if 'vwap' in self._indicators and self._indicators['vwap'] is not None:
            self._indicators['vwap'].update_with_candle(candle_data)
        
        # 4. Daily Levels는 자동 업데이트 (어제 데이터이므로)
        if 'daily_levels' in self._indicators and self._indicators['daily_levels'] is not None:
            self._indicators['daily_levels'].update_with_candle(candle_data)
        
        if 'opening_range' in self._indicators and self._indicators['opening_range'] is not None:
            self._indicators['opening_range'].update_with_candle(candle_data)
            
        print(f"✅ 전체 지표 업데이트 완료: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
        print(f"")
    def get_indicator(self, name: str):
        """특정 지표 반환"""
        if not self._initialized:
            return None
        
        return self._indicators.get(name)
    
    def get_data_manager(self):
        """DataManager 반환 (지연 초기화)"""
        if self._data_manager is None:
            self._data_manager = get_data_manager()
        return self._data_manager

    def get_all_indicators(self) -> Dict[str, Any]:
        """모든 지표 반환"""
        if not self._initialized:
            return {}
        
        return self._indicators.copy()
    
    def list_indicators(self) -> List[str]:
        """등록된 지표 목록 반환"""
        return list(self._indicators.keys())
    
    def is_initialized(self) -> bool:
        """지표들이 초기화되었는지 확인"""
        return self._initialized


# 전역 인스턴스 (싱글톤)
_global_indicator_manager = None


def get_global_indicator_manager() -> GlobalIndicatorManager:
    """
    전역 지표 관리자 인스턴스 반환 (싱글톤 패턴)
    
    Returns:
        GlobalIndicatorManager: 전역 지표 관리자 인스턴스
    """
    global _global_indicator_manager
    
    if _global_indicator_manager is None:
        _global_indicator_manager = GlobalIndicatorManager()
    
    return _global_indicator_manager


def initialize_global_indicators():
    """전역 지표들 초기화 (편의 함수)"""
    manager = get_global_indicator_manager()
    manager.initialize_indicators()
    return manager


def update_all_indicators_with_candle(candle_data: Dict[str, Any]):
    """새로운 3분봉으로 모든 지표 업데이트 (편의 함수)"""
    manager = get_global_indicator_manager()
    manager.update_all_indicators(candle_data)

def get_vwap() -> Tuple[Optional[float], Optional[float]]:
    """VWAP 값 바로 가져오기"""
    manager = get_global_indicator_manager()
    vwap_indicator = manager.get_indicator('vwap')
    return (vwap_indicator.get_status().get('vwap'), vwap_indicator.get_status().get('vwap_std'))

def get_atr() -> Optional[float]:
    """ATR 값 바로 가져오기"""
    manager = get_global_indicator_manager()
    atr_indicator = manager.get_indicator('atr')
    return atr_indicator.get_status().get('atr') if atr_indicator else None

def get_daily_levels() -> Tuple[Optional[float], Optional[float]]:
    """어제 고가 바로 가져오기"""
    manager = get_global_indicator_manager()
    daily_indicator = manager.get_indicator('daily_levels')
    return (daily_indicator.get_status().get('prev_day_high'), daily_indicator.get_status().get('prev_day_low'))

def get_opening_range() -> Tuple[Optional[float], Optional[float]]:
    """개장 범위 고가 바로 가져오기"""
    manager = get_global_indicator_manager()
    opening_indicator = manager.get_indicator('opening_range')
    return (opening_indicator.get_status().get('high'), opening_indicator.get_status().get('low'))

def get_vpvr() -> Optional[int]:
    """VPVR 활성 구간 수 바로 가져오기"""
    manager = get_global_indicator_manager()
    vpvr_indicator = manager.get_indicator('vpvr')
    return (
        vpvr_indicator.get_status().get('poc'), 
        vpvr_indicator.get_status().get('hvn'), 
        vpvr_indicator.get_status().get('lvn')
        )
def get_vpvr_status() -> Optional[Dict[str, Any]]:
    """VPVR 상태 바로 가져오기"""
    manager = get_global_indicator_manager()
    vpvr_indicator = manager.get_indicator('vpvr')
    return vpvr_indicator.get_status()