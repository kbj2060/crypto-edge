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
from indicators.vpvr import SessionVPVR
from indicators.atr import ATR3M
from indicators.daily_levels import DailyLevels
from indicators.vwap import SessionVWAP
from managers.data_manager import get_data_manager
from managers.time_manager import get_time_manager


class GlobalIndicatorManager:
    """
    전역 지표 관리자
    - 모든 지표들을 중앙에서 관리
    - 새로운 3분봉 데이터로 전체 지표 자동 업데이트
    - 싱글톤 패턴으로 전역 접근
    """
    
    def __init__(self, target_time: Optional[datetime] = None):
        """글로벌 지표 관리자 초기화"""
        self._indicators = {}
        self._initialized = False
        self._lock = threading.Lock()  # 스레드 안전성
        self._data_manager = None  # DataManager 인스턴스 (지연 초기화)
        self.target_time = target_time if target_time is not None else datetime.now(timezone.utc)
        self.time_manager = get_time_manager(self.target_time)

        # 지표 설정
        self.indicator_configs = {
            'vpvr': {
                'class': SessionVPVR,
                'target_time': self.target_time,
            },
            'atr': {
                'class': ATR3M,
                'target_time': self.target_time,
            },
            'daily_levels': {
                'class': DailyLevels,
                'target_time': self.target_time,
            },
            'vwap': {
                'class': SessionVWAP,
                'target_time': self.target_time,
            }
        }
        
    def _initialize_indicator(self, name: str):
        """지표 초기화 - 공통 메서드"""
        try:
            config = self.indicator_configs[name]
            indicator_class = config['class']

            if name == 'vpvr':
                self._indicators[name] = indicator_class(
                    target_time=self.target_time,
                )
            elif name == 'atr':
                self._indicators[name] = indicator_class(
                    target_time=self.target_time,
                )
            elif name == 'vwap':
                self._indicators[name] = indicator_class(
                    target_time=self.target_time,
                )
            elif name == 'daily_levels':
                self._indicators[name] = indicator_class(
                    target_time=self.target_time,
                )
            else:
                # 기본 초기화 (매개변수 없음)
                self._indicators[name] = indicator_class(
                    target_time=self.target_time,
                )
                
        except Exception as e:
            import traceback
            print(f"❌ {name} 지표 초기화 오류: {e}")
            print(f"❌ 상세 오류: {traceback.format_exc()}")
            self._indicators[name] = None

    def initialize_indicators(self):
        """모든 지표 초기화"""
        with self._lock:
            if self._initialized:
                print("🔄 전역 지표 이미 초기화됨")
                return
            
            self.time_manager = get_time_manager(self.target_time)

            try:
                data_manager = self.get_data_manager()
                if not data_manager.is_ready():
                    print("🔄 DataManager 준비 안됨")
                    return

                # 🚀 2단계: 나머지 지표들 초기화 (DataManager 완료 후)
                print("\n🔥 2단계: 나머지 지표들 초기화 시작...")
                
                # 모든 지표를 순차적으로 초기화
                for indicator_name in self.indicator_configs.keys():
                    self._initialize_indicator(indicator_name)
                
                self._initialized = True
                print("🎯 모든 전역 지표 초기화 완료!")
            
            except Exception:
                self._initialized = False
    
    def update_all_indicators(self, candle_data: pd.Series):
        """
        새로운 3분봉 데이터로 모든 지표 업데이트
        
        Args:
            candle_data: 3분봉 캔들 데이터프레임 (1개 행) 
        """
        if not self._initialized:
            return

        self.time_manager.update_with_candle(candle_data)
        
        # 1. ATR 업데이트 (가장 먼저 - 다른 지표들이 사용)
        if 'atr' in self._indicators and self._indicators['atr'] is not None:
            self._indicators['atr'].update_with_candle(candle_data)
            #print(f"✅ [{candle_data.name.strftime('%H:%M:%S')}]ATR 업데이트 완료: {self._indicators['atr'].get_status().get('atr')} ")

        # 2. VPVR 업데이트
        if 'vpvr' in self._indicators and self._indicators['vpvr'] is not None:
            self._indicators['vpvr'].update_with_candle(candle_data)
            #print(f"✅ [{candle_data.name.strftime('%H:%M:%S')}]VPVR 업데이트 완료: {self._indicators['vpvr'].get_status().get('poc')} {self._indicators['vpvr'].get_status().get('hvn')} {self._indicators['vpvr'].get_status().get('lvn')}")
        
        # 3. VWAP 업데이트
        if 'vwap' in self._indicators and self._indicators['vwap'] is not None:
            self._indicators['vwap'].update_with_candle(candle_data)
            #print(f"✅ [{candle_data.name.strftime('%H:%M:%S')}]VWAP 업데이트 완료: {self._indicators['vwap'].get_status().get('vwap')} {self._indicators['vwap'].get_status().get('vwap_std')}")
        
        # 4. Daily Levels는 자동 업데이트 (어제 데이터이므로)
        if 'daily_levels' in self._indicators and self._indicators['daily_levels'] is not None:
            self._indicators['daily_levels'].update_with_candle(candle_data)
            #print(f"✅ [{candle_data.name.strftime('%H:%M:%S')}]Daily Levels 업데이트 완료: {self._indicators['daily_levels'].get_status().get('prev_day_high')} {self._indicators['daily_levels'].get_status().get('prev_day_low')}")

        #print(f"✅ 전체 지표 업데이트 완료: {candle_data.name.strftime('%H:%M:%S')}")
        #print(f"")
        
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


def get_global_indicator_manager(target_time: Optional[datetime] = None) -> GlobalIndicatorManager:
    """
    전역 지표 관리자 인스턴스 반환 (싱글톤 패턴)
    
    Returns:
        GlobalIndicatorManager: 전역 지표 관리자 인스턴스
    """
    global _global_indicator_manager
    
    if _global_indicator_manager is None:
        _global_indicator_manager = GlobalIndicatorManager(target_time)
    
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

def get_all_indicators() -> Dict[str, Any]:
    # 튜플 언패킹으로 안전하게 처리
    poc, hvn, lvn = get_vpvr()
    vwap, vwap_std = get_vwap()
    atr = get_atr() 
    prev_day_high, prev_day_low = get_daily_levels()
    
    return {
        "poc" : round(float(poc), 3) if poc is not None else None,
        "hvn"  : round(float(hvn), 3) if hvn is not None else None,
        "lvn" : round(float(lvn), 3) if lvn is not None else None,
        "vwap" : round(float(vwap), 3) if vwap is not None else None,
        "vwap_std" : round(float(vwap_std), 3) if vwap_std is not None else None,
        "atr" : round(float(atr), 3) if atr is not None else None,
        "prev_day_high" : round(float(prev_day_high), 3) if prev_day_high is not None else None,
        "prev_day_low" : round(float(prev_day_low), 3) if prev_day_low is not None else None,
    }