#!/usr/bin/env python3
"""
VWAP (Volume Weighted Average Price) 지표
- 세션 기반 VWAP 계산
- 실시간 업데이트
- 세션 외 시간 지원
"""

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from managers.data_manager import get_data_manager
from utils.time_manager import get_time_manager
from utils.session_manager import get_session_manager

class SessionVWAP:
    """세션 기반 VWAP 관리 클래스"""
    
    def __init__(self, symbol: str = "ETHUSDC", target_time: Optional[datetime] = None):
        """VWAP 초기화"""
        self.symbol = symbol
        self.time_manager = get_time_manager()
        self.session_manager = get_session_manager()
        
        # VWAP 데이터
        self.current_vwap = 0.0
        self.current_vwap_std = 0.0
        self.session_data = []
        self.processed_candle_count = 0
        self.initial_data_count = 0
        self.target_time = target_time if target_time is not None else self.time_manager.get_current_time()

        # 캐시 및 상태
        self.cached_result = {}
        self.last_update_time = None
        self.last_session_name = None
        
        # 초기 데이터 자동 로딩
        self._initialize_vwap()
    
    def _initialize_vwap(self):
        """초기 데이터 자동 로딩"""
        is_active = self.session_manager.is_session_active()
        session_config = self.session_manager.get_indicator_mode_config()

        if is_active and session_config.get('start_time') + timedelta(minutes=30) <= self.time_manager.get_current_time():
            self._load_session_data(session_config)
        else:
            self._load_recent_data()
        
        # 초기 세션 이름 설정
        self.last_session_name = session_config.get('session_name', 'UNKNOWN')
    
    def _load_session_data(self, session_config: Dict[str, Any]):
        """세션 시작부터 현재까지 데이터 로딩"""
        try:
            data_manager = get_data_manager()
            session_start = session_config.get('start_time')

            session_start = self.time_manager.ensure_utc(session_start)
            self.target_time = self.time_manager.ensure_utc(self.target_time)
            
            df = data_manager.get_data_range(session_start, self.target_time)

            if df is None or df.empty:
                print("❌ 세션 데이터 로드 실패")
                return
            
            session_data = df[df.index >= session_start]
            self._calculate_session_vwap(session_data)
                
        except Exception as e:
            print(f"❌ 세션 데이터 로드 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_recent_data(self):
        """세션 외 시간용 데이터 로딩 - 이전 세션 종료 시점부터 현재까지"""
        try:
            
            data_manager = get_data_manager()
            previous_session_end = self.session_manager.get_previous_session_close()

            if previous_session_end:
                print(f"📊 세션 외 시간: 이전 세션 종료 시점({self.time_manager.format_datetime(previous_session_end)})부터 현재({self.time_manager.format_datetime(self.target_time)})까지 데이터 로딩")
                df = data_manager.get_data_range(previous_session_end, self.target_time)

            if df is None or df.empty:
                print("❌ 세션 외 시간 데이터 로드 실패")
                return
            
            # 초기 로딩된 데이터 수 저장
            self.initial_data_count = len(df)
            
            # 세션 외 시간 데이터로 VWAP 계산
            self._calculate_session_vwap(df)
        
        except Exception as e:
            print(f"❌ 세션 외 시간 데이터 로드 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_session_vwap(self, df: pd.DataFrame):
        """세션 데이터로 VWAP 계산"""
        try:
            if df.empty:
                return
            
            # 데이터 복사 및 전처리
            df = df.copy()
            
            # 필요한 컬럼이 있는지 확인
            for col in ['high', 'low', 'close', 'volume']:
                if col not in df.columns:
                    print(f"❌ 필수 컬럼 누락: {col}")
                    return
            
            # NaN 값 제거
            df = df.dropna(subset=['high', 'low', 'close', 'volume'])
            if df.empty:
                return
            
            # VWAP 계산
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            volume_price = typical_price * df['volume']
            total_volume = df['volume'].sum()
            
            if total_volume > 0:
                self.current_vwap = float(volume_price.sum() / total_volume)
                
                # 표준편차 계산 (2개 이상의 캔들이 있을 때)
                if len(df) > 1:
                    vwap_diff = typical_price - self.current_vwap
                    vwap_variance = (vwap_diff ** 2 * df['volume']).sum() / total_volume
                    self.current_vwap_std = float(vwap_variance ** 0.5)
                else:
                    self.current_vwap_std = 0.0
            else:
                # 단일 캔들의 경우 고가-저가 범위의 절반을 표준편차로 사용
                price_range = df['high'].iloc[0] - df['low'].iloc[0]
                self.current_vwap_std = float(price_range * 0.5)
            
            # 세션 데이터 업데이트
            self.session_data = df.to_dict('records')
            self.processed_candle_count = len(df)
            
            # VWAP 결과 업데이트
            self._update_vwap_result()
            
        except Exception as e:
            print(f"❌ 세션 VWAP 계산 오류: {e}")

    def update_with_candle(self, candle_data: pd.Series):
        """새로운 캔들로 VWAP 업데이트"""
        try:
            self.target_time = self.time_manager.ensure_utc(candle_data.name)
            # 세션 상태 업데이트
            session_config = self.session_manager.get_indicator_mode_config()
            
            # 세션 변경 확인 및 리셋
            self._check_session_reset(session_config)
            
            # 새로운 캔들 추가
            self.session_data.append(candle_data)
            self.processed_candle_count += 1
            
            # VWAP 재계산
            df = pd.DataFrame(self.session_data)
            self._calculate_session_vwap(df)
            
            # 세션 정보 출력
            session_config.get('elapsed_minutes', 0)

        except Exception as e:
            print(f"❌ VWAP 업데이트 오류: {e}")
    
    def _check_session_reset(self, session_config: Dict[str, Any]):
        """세션 변경 시 VWAP 리셋 확인"""
        try:
            current_session = session_config.get('session_name', 'UNKNOWN')
            
            # 이전 세션과 다른 경우 리셋
            if hasattr(self, 'last_session_name') and self.last_session_name != current_session:
                print(f"🔄 세션 변경 감지: {self.last_session_name} → {current_session}")
                print("🔄 VWAP 세션 데이터 리셋")
                self.reset_session()
            
            # 현재 세션 이름 저장
            self.last_session_name = current_session
            
        except Exception as e:
            print(f"❌ 세션 리셋 확인 오류: {e}")
    
    def _update_vwap_result(self):
        """VWAP 결과 업데이트"""
        try:
            session_config = self.session_manager.get_indicator_mode_config()
            
        
            result = {
                "vwap": self.current_vwap,
                "vwap_std": self.current_vwap_std,
                "total_volume": sum([candle.get('volume', 0) for candle in self.session_data]),
                "data_count": self.processed_candle_count,
                "last_update": self.target_time.isoformat(),
                "mode": "session" if session_config['use_session_mode'] else "outside_session"
            }
            
            # 세션 정보 추가
            if session_config['use_session_mode']:
                result.update({
                    "session": session_config.get('session_name'),
                    "session_start": session_config.get('start_time').isoformat() if session_config.get('start_time') else None,
                    "elapsed_minutes": session_config.get('elapsed_minutes', 0)
                })
            
            self.cached_result = result
            self.last_update_time = self.target_time
        
        except Exception as e:
            print(f"❌ VWAP 결과 업데이트 오류: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """현재 VWAP 결과 반환"""
        return self.cached_result
    
    def reset_session(self):
        """세션 데이터 초기화"""
        self.session_data.clear()
        self.processed_candle_count = 0
        self.initial_data_count = 0
        self.current_vwap = 0.0
        self.current_vwap_std = 0.0
        self.cached_result = {}
        self.last_update_time = None
        print("📊 VWAP 세션 초기화 완료")
