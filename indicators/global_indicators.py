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

# 지표 클래스들 import
from .vpvr import SessionVPVR
from .atr import ATR3M
from .daily_levels import DailyLevels
from .vwap import SessionVWAP


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
        
        # 지표 설정
        self.indicator_configs = {
            'vpvr': {
                'class': SessionVPVR,
                'auto_load': True,
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
                'auto_load': True
            },
            'vwap': {
                'class': SessionVWAP,
                'symbol': 'ETHUSDT',
                'auto_load': True
            }
        }
        
        print("🚀 GlobalIndicatorManager 초기화 완료")
    
    def initialize_indicators(self):
        """모든 지표 초기화"""
        with self._lock:
            if self._initialized:
                return
            
            print("🔧 전역 지표들 초기화 시작...")
            
            try:
                # VPVR 지표 초기화
                vpvr_config = self.indicator_configs['vpvr']
                self._indicators['vpvr'] = vpvr_config['class'](
                    bins=vpvr_config['bins'],
                    price_bin_size=vpvr_config['price_bin_size'],
                    lookback=vpvr_config['lookback'],
                    auto_load=vpvr_config['auto_load']
                )
                print("   ✅ VPVR 지표 초기화 완료")
                
                # ATR 지표 초기화
                atr_config = self.indicator_configs['atr']
                self._indicators['atr'] = atr_config['class'](
                    length=atr_config['length'],
                    max_candles=atr_config['max_candles']
                )
                
                # ATR 초기 데이터 로딩 (연속 롤링을 위해 필요)
                print("🚀 ATR 초기 데이터 자동 로딩 시작...")
                try:
                    from data.binance_dataloader import BinanceDataLoader
                    
                    dataloader = BinanceDataLoader()
                    # 최근 24시간 데이터로 ATR 초기화 (연속 롤링을 위해)
                    df = dataloader.fetch_recent_3m('ETHUSDT', hours=24)
                    
                    if df is not None and not df.empty:
                        print(f"✅ ATR 초기 데이터 로드 성공: {len(df)}개 캔들")
                        
                        # ATR에 캔들 데이터 주입 (연속 롤링 시작)
                        for _, row in df.iterrows():
                            candle_data = {
                                'timestamp': row.name,  # 인덱스가 timestamp
                                'open': row['open'],
                                'high': row['high'],
                                'low': row['low'],
                                'close': row['close'],
                                'volume': row['volume']
                            }
                            self._indicators['atr'].update_with_candle(candle_data)
                        
                        # ATR 상태 확인
                        atr_value = self._indicators['atr'].get_atr()
                        atr_status = self._indicators['atr'].get_status()
                        is_ready = atr_status.get('is_ready', False)
                        is_mature = atr_status.get('is_mature', False)
                        candles_count = atr_status.get('candles_count', 0)
                        
                        print(f"   📊 ATR 초기화 완료: {atr_value:.3f}")
                        print(f"   ✅ 준비 상태: {is_ready}")
                        print(f"   🎯 성숙 상태: {is_mature}")
                        print(f"   📊 캔들 개수: {candles_count}개")
                        print(f"   🔄 연속 롤링 모드 활성화")
                    else:
                        print("⚠️ ATR 초기 데이터 로드 실패")
                        
                except Exception as e:
                    print(f"❌ ATR 초기 데이터 로딩 오류: {e}")
                
                print("   ✅ ATR 지표 초기화 완료")
                
                # Daily Levels 지표 초기화
                daily_config = self.indicator_configs['daily_levels']
                self._indicators['daily_levels'] = daily_config['class'](
                    symbol=daily_config['symbol'],
                    auto_load=daily_config['auto_load']
                )
                print("   ✅ Daily Levels 지표 초기화 완료")
                
                # VWAP 지표 초기화
                vwap_config = self.indicator_configs['vwap']
                self._indicators['vwap'] = vwap_config['class'](
                    symbol=vwap_config['symbol'],
                    auto_load=vwap_config['auto_load']
                )
                print("   ✅ VWAP 지표 초기화 완료")
                
                self._initialized = True
                print("🎯 모든 전역 지표 초기화 완료!")
                
            except Exception as e:
                print(f"❌ 전역 지표 초기화 오류: {e}")
                import traceback
                traceback.print_exc()
                self._initialized = False
    
    def update_all_indicators(self, candle_data: Dict[str, Any]):
        """
        새로운 3분봉 데이터로 모든 지표 업데이트
        
        Args:
            candle_data: 3분봉 캔들 데이터 {
                'timestamp': datetime,
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float
            }
        """
        if not self._initialized:
            print("⚠️ 지표들이 아직 초기화되지 않음. 먼저 초기화하세요.")
            return
        
        try:
            timestamp = candle_data.get('timestamp', datetime.now(timezone.utc))
            print(f"🔄 {timestamp.strftime('%H:%M:%S')} - 전체 지표 업데이트 시작...")
            
            # 1. ATR 업데이트 (가장 먼저 - 다른 지표들이 사용)
            if 'atr' in self._indicators:
                self._indicators['atr'].update_with_candle(candle_data)
                atr_value = self._indicators['atr'].get_atr()
                print(f"   📊 ATR 업데이트: {atr_value:.3f}")
            
            # 2. VPVR 업데이트
            if 'vpvr' in self._indicators:
                self._indicators['vpvr'].update_with_candle(candle_data)
                vpvr_status = self._indicators['vpvr'].get_vpvr_status()
                active_bins = vpvr_status.get('active_bins', 0)
                print(f"   📈 VPVR 업데이트: 활성 구간 {active_bins}개")
            
            # 3. VWAP 업데이트
            if 'vwap' in self._indicators:
                self._indicators['vwap'].update_with_candle(candle_data)
                vwap_status = self._indicators['vwap'].get_vwap_status()
                current_vwap = vwap_status.get('current_vwap', 0)
                print(f"   📊 VWAP 업데이트: ${current_vwap:.2f}")
            
            # 4. Daily Levels는 자동 업데이트 (어제 데이터이므로)
            if 'daily_levels' in self._indicators:
                daily_status = self._indicators['daily_levels'].is_loaded()
                print(f"   📅 Daily Levels 상태: {'로드됨' if daily_status else '로드 안됨'}")
            
            print(f"✅ 전체 지표 업데이트 완료: {timestamp.strftime('%H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ 전체 지표 업데이트 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def update_all_indicators_with_candle(self, candle_data: Dict[str, Any]):
        """
        새로운 3분봉 데이터로 모든 지표 업데이트 (편의 메서드)
        
        Args:
            candle_data: 3분봉 캔들 데이터 (Dict 또는 pandas Series)
        """
        # pandas Series인 경우 Dict로 변환
        if hasattr(candle_data, 'to_dict'):
            candle_dict = candle_data.to_dict()
            # timestamp가 인덱스인 경우 별도 처리
            if hasattr(candle_data, 'name') and candle_data.name:
                candle_dict['timestamp'] = candle_data.name
        else:
            candle_dict = candle_data
        
        self.update_all_indicators(candle_dict)
    
    def get_indicator(self, name: str):
        """특정 지표 반환"""
        if not self._initialized:
            print("⚠️ 지표들이 아직 초기화되지 않음")
            return None
        
        return self._indicators.get(name)
    
    def get_all_indicators(self) -> Dict[str, Any]:
        """모든 지표 반환"""
        if not self._initialized:
            print("⚠️ 지표들이 아직 초기화되지 않음")
            return {}
        
        return self._indicators.copy()
    
    def list_indicators(self) -> List[str]:
        """등록된 지표 목록 반환"""
        return list(self._indicators.keys())
    
    def get_indicators_status(self) -> Dict[str, Any]:
        """모든 지표의 상태 정보 반환"""
        if not self._initialized:
            return {'status': 'not_initialized'}
        
        status = {
            'status': 'initialized',
            'indicators': {},
            'last_update': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # VPVR 상태
            if 'vpvr' in self._indicators:
                vpvr_status = self._indicators['vpvr'].get_vpvr_status()
                status['indicators']['vpvr'] = {
                    'active_bins': vpvr_status.get('active_bins', 0),
                    'total_volume': vpvr_status.get('total_volume', 0),
                    'data_count': vpvr_status.get('data_count', 0),
                    'session_status': vpvr_status.get('session_status', 'UNKNOWN')
                }
            
            # ATR 상태
            if 'atr' in self._indicators:
                atr = self._indicators['atr']
                atr_status = atr.get_status()
                status['indicators']['atr'] = {
                    'current_atr': atr_status.get('current_atr', 0),
                    'is_ready': atr_status.get('is_ready', False),
                    'is_mature': atr_status.get('is_mature', False),
                    'candles_count': atr_status.get('candles_count', 0)
                }
            
            # Daily Levels 상태
            if 'daily_levels' in self._indicators:
                daily_levels = self._indicators['daily_levels'].get_prev_day_high_low()
                status['indicators']['daily_levels'] = {
                    'is_loaded': self._indicators['daily_levels'].is_loaded(),
                    'prev_day_high': daily_levels.get('high', 0),
                    'prev_day_low': daily_levels.get('low', 0)
                }
            
            # VWAP 상태
            if 'vwap' in self._indicators:
                vwap_status = self._indicators['vwap'].get_vwap_status()
                status['indicators']['vwap'] = {
                    'current_vwap': vwap_status.get('current_vwap', 0),
                    'current_vwap_std': vwap_status.get('current_vwap_std', 0),
                    'data_count': vwap_status.get('data_count', 0),
                    'mode': vwap_status.get('mode', 'unknown')
                }
                
        except Exception as e:
            print(f"❌ 지표 상태 수집 오류: {e}")
            status['error'] = str(e)
        
        return status
    
    def reset_all_indicators(self):
        """모든 지표 리셋"""
        with self._lock:
            print("🔄 모든 전역 지표 리셋 시작...")
            
            try:
                # VPVR 리셋
                if 'vpvr' in self._indicators:
                    self._indicators['vpvr'].reset_session()
                    print("   ✅ VPVR 리셋 완료")
                
                # ATR 리셋
                if 'atr' in self._indicators:
                    self._indicators['atr'].reset()
                    print("   ✅ ATR 리셋 완료")
                
                # VWAP 리셋
                if 'vwap' in self._indicators:
                    self._indicators['vwap'].reset_session()
                    print("   ✅ VWAP 리셋 완료")
                
                # Daily Levels는 리셋하지 않음 (어제 데이터이므로)
                print("   ⏭️ Daily Levels 리셋 건너뜀 (어제 데이터)")
                
                print("🎯 모든 전역 지표 리셋 완료!")
                
            except Exception as e:
                print(f"❌ 지표 리셋 오류: {e}")
    
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
