#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
전역 지표 관리자
- 모든 지표들을 중앙에서 관리
- 새로운 3분봉 데이터로 전체 지표 자동 업데이트
- 싱글톤 패턴으로 전역 접근
"""

from typing import Dict, Any, Optional, List
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
        
        print("🚀 GlobalIndicatorManager 초기화 완료")
    
    def _initialize_vpvr_indicator(self):
        """VPVR 지표 초기화 - 공통 데이터 사용"""
        vpvr_config = self.indicator_configs['vpvr']
        self._indicators['vpvr'] = vpvr_config['class'](
            bins=vpvr_config['bins'],
            price_bin_size=vpvr_config['price_bin_size'],
            lookback=vpvr_config['lookback'],
        )
        
        # DataManager에서 데이터 가져와서 VPVR에 전달
        data_manager = self.get_data_manager()
        if data_manager.is_ready():
            df = data_manager.get_dataframe()
            if not df.empty:
                print(f"   📊 DataIndicator에서 데이터 로드: {len(df)}개 캔들")
                # self._indicators['vpvr'].update_with_dataframe(df)
            else:
                print("   ⚠️ DataManager에 데이터가 없습니다")
        else:
            print("   ⚠️ DataManager가 준비되지 않았습니다")
        
        print("   ✅ VPVR 지표 초기화 완료")

    def _initialize_atr_indicator(self):
        """ATR 지표 초기화 및 초기 데이터 로딩"""
        atr_config = self.indicator_configs['atr']
        self._indicators['atr'] = atr_config['class'](
            length=atr_config['length'],
            max_candles=atr_config['max_candles']
        )
        
        print("   ✅ ATR 지표 초기화 완료")

    def _initialize_daily_levels_indicator(self):
        """Daily Levels 지표 초기화"""
        daily_config = self.indicator_configs['daily_levels']
        self._indicators['daily_levels'] = daily_config['class']()
        print("   ✅ Daily Levels 지표 초기화 완료")

    def _initialize_vwap_indicator(self):
        """VWAP 지표 초기화 - DataIndicator에서 데이터 가져오기"""
        try:
            vwap_config = self.indicator_configs['vwap']
            self._indicators['vwap'] = vwap_config['class'](
                symbol=vwap_config['symbol']
            )
            
            # DataManager에서 데이터 가져오기
            data_manager = self.get_data_manager()
            if data_manager.is_ready():
                df = data_manager.get_dataframe()
                if not df.empty:
                    print(f"   📊 DataManager에서 데이터 로드: {len(df)}개 캔들")
                    self._indicators['vwap'].update_with_dataframe(df)
                else:
                    print("   ⚠️ DataManager에 데이터가 없습니다")
            else:
                print("   ⚠️ DataManager가 준비되지 않았습니다")
            
            print("   ✅ VWAP 지표 초기화 완료")
            
        except Exception as e:
            print(f"❌ VWAP 지표 초기화 오류: {e}")
            self._indicators['vwap'] = None

    def _initialize_opening_range_indicator(self):
        """Opening Range 지표 초기화 - DataManager에서 데이터 가져오기"""
        try:
            print("🚀 OpeningRange 초기화 시작...")
            
            self._indicators['opening_range'] = OpeningRange(or_minutes=30)
            print(self._indicators['opening_range'].get_status())
            
            print("   ✅ Opening Range 지표 초기화 완료")
            
        except Exception as e:
            print(f"❌ Opening Range 지표 초기화 오류: {e}")
            self._indicators['opening_range'] = None

    def initialize_indicators(self):
        """모든 지표 초기화"""
        with self._lock:
            if self._initialized:
                return
            
            print("🔧 전역 지표들 초기화 시작...")
            
            try:
                # 🚀 1단계: DataManager 상태 확인 (이미 smart_trader에서 초기화됨)
                print("📊 1단계: DataManager 상태 확인...")
                
                data_manager = self.get_data_manager()
                if not data_manager.is_ready():
                    print("❌ DataManager가 아직 준비되지 않음. smart_trader에서 먼저 초기화하세요.")
                    return
                
                print("✅ DataManager가 이미 준비됨 - 중앙 데이터 저장소 사용 가능")
                
            
                # 🚀 2단계: 나머지 지표들 초기화 (DataManager 완료 후)
                print("\n🔥 2단계: 나머지 지표들 초기화 시작...")
                self._initialize_atr_indicator()
                self._initialize_daily_levels_indicator()
                self._initialize_vpvr_indicator()
                self._initialize_vwap_indicator()
                self._initialize_opening_range_indicator()
                
                self._initialized = True
                print("🎯 모든 전역 지표 초기화 완료!")
                
                
            except Exception as e:
                print(f"❌ 전역 지표 초기화 오류: {e}")
                import traceback
                traceback.print_exc()
                self._initialized = False
    
    def update_all_indicators(self, candle_data: pd.Series):
        """
        새로운 3분봉 데이터로 모든 지표 업데이트
        
        Args:
            candle_data: 3분봉 캔들 데이터프레임 (1개 행) 
        """
        if not self._initialized:
            print("⚠️ 지표들이 아직 초기화되지 않음. 먼저 초기화하세요.")
            return
        
        timestamp = candle_data.get('timestamp', datetime.now(timezone.utc))
        print(f"🔄 전체 지표 업데이트 시작...")
        
        data_manager = self.get_data_manager()
        data_manager.update_with_candle(candle_data)
        print(f"   📊 DataManager 업데이트")

        # 1. ATR 업데이트 (가장 먼저 - 다른 지표들이 사용)
        if 'atr' in self._indicators:
            self._indicators['atr'].update_with_candle(candle_data)
            atr_value = self._indicators['atr'].get_status().get('current_atr')
            print(f"   📊 ATR 업데이트: {atr_value:.3f}")
        
        # 2. VPVR 업데이트
        if 'vpvr' in self._indicators:
            self._indicators['vpvr'].update_with_candle(candle_data)
            vpvr_status = self._indicators['vpvr'].get_status()
            active_bins = vpvr_status.get('active_bins')
            print(f"   📈 VPVR 업데이트: 활성 구간 {active_bins}개")
        
        # 3. VWAP 업데이트
        if 'vwap' in self._indicators:
            self._indicators['vwap'].update_with_candle(candle_data)
            vwap_status = self._indicators['vwap'].get_status()
            current_vwap = vwap_status.get('current_vwap')
            print(f"   📊 VWAP 업데이트: ${current_vwap:.2f}")
        
        # 4. Daily Levels는 자동 업데이트 (어제 데이터이므로)
        if 'daily_levels' in self._indicators:
            self._indicators['daily_levels'].update_with_candle(candle_data)
            daily_status = self._indicators['daily_levels'].get_status()
            print(f"   📅 Daily Levels 상태: {'로드됨' if daily_status else '로드 안됨'}")
        
        if 'opening_range' in self._indicators:
            self._indicators['opening_range'].update_with_candle(candle_data)
            opening_range_status = self._indicators['opening_range'].get_status()
            is_open = opening_range_status.get('is_open', False)
            print(f"   🌅 Opening Range 업데이트: {'개장 중' if is_open else '폐장'}")
        
        print(f"✅ 전체 지표 업데이트 완료: {timestamp.strftime('%H:%M:%S')}")
            
    
    def get_indicator(self, name: str):
        """특정 지표 반환"""
        if not self._initialized:
            print("⚠️ 지표들이 아직 초기화되지 않음")
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
            print("⚠️ 지표들이 아직 초기화되지 않음")
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


def get_indicator(name: str):
    """특정 지표 반환 (편의 함수)"""
    manager = get_global_indicator_manager()
    return manager.get_indicator(name)


def get_indicators_status():
    """모든 지표 상태 반환 (편의 함수)"""
    manager = get_global_indicator_manager()
    return manager.get_indicators_status()
