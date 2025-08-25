"""
Volume Profile Visible Range (VPVR) 지표 모듈

주요 기능:
- 세션 기반 실시간 VPVR 관리
- 동적 bin 크기 계산 (ATR 기반)
- POC, HVN, LVN 계산
- 공용 데이터를 사용한 효율적인 업데이트
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Any
import datetime as dt
from utils.time_manager import get_time_manager
from data.data_manager import get_data_manager
from indicators.atr import ATR3M

class SessionVPVR:
    """
    세션 기반 실시간 VPVR 관리 클래스
    
    세션 시작 시 리셋하고, 3분봉 닫힐 때마다 실시간 업데이트합니다.
    공용 데이터를 사용하여 효율적으로 VPVR을 계산합니다.
    """
    
    def __init__(self, bins: int = 50, price_bin_size: float = 0.05, 
                    lookback: int = 100):
        """
        SessionVPVR 초기화
        
        Args:
            bins: 가격 bin 개수
            price_bin_size: 기본 가격 bin 크기
            lookback: 세션 외 시간대 처리용 캔들 개수
            auto_load: 초기 데이터 자동 로딩 여부
        """
        self.bins = bins
        self.price_bin_size = price_bin_size
        self.lookback = lookback
        
        # 핵심 데이터 구조
        self.price_bins = {}
        self.volume_histogram = {}
        self.cached_result = None
        self.last_update_time = None
        self.processed_candle_count = 0
        
        # 세션 외 시간대 처리용
        self.lookback_data = []
        
        # 동적 빈 사이즈 관리
        self.bin_size = None
        
        # 의존성 객체들
        self.time_manager = get_time_manager()
        self.atr = ATR3M(length=14)
        
        self._initialize_vpvr()
    
    def _initialize_vpvr(self):
        """초기화 시 자동으로 적절한 데이터 로딩"""
        print("🚀 VPVR 초기 데이터 자동 로딩 시작...")
        
        session_config = self.time_manager.get_indicator_mode_config()
        
        if session_config['use_session_mode']:
            print(f"📊 세션 모드: {session_config['session_name']} 세션 시작부터 현재까지 데이터 로딩")
            self._load_session_data(session_config)
            self._update_vpvr_result(session_config)
        else:
            print(f"📊 룩백 모드: 최근 {self.lookback}개 3분봉 데이터 로딩")
            self._load_lookback_data()
            self._update_vpvr_result()

        self.last_update_time = dt.datetime.now(dt.timezone.utc)
        self.last_session_name = session_config.get('session_name', 'UNKNOWN')
        print("✅ VPVR 초기 데이터 로딩 완료")
            
    
    def _load_session_data(self, session_config: Dict[str, any]):
        """세션 시작부터 현재까지의 데이터 로딩"""
        try:
            data_manager = get_data_manager()
            session_start = session_config.get('session_start_time')
            
            if not session_start:
                print("⚠️ 세션 시작 시간을 찾을 수 없습니다")
                return
            
            # 세션 시작 시간을 datetime 객체로 변환
            if isinstance(session_start, str):
                session_start = dt.datetime.fromisoformat(session_start.replace('Z', '+00:00'))
            
            # 세션 시작 이후 데이터만 필터링
            df = data_manager.get_data_range(session_start, dt.datetime.now(dt.timezone.utc))
            
            if df.empty:
                print("⚠️ 세션 시작 이후 데이터가 없습니다")
                return
            
            print(f"📊 세션 데이터 로드: {len(df)}개 캔들")
            
            # VPVR에 데이터 직접 누적
            for timestamp, row in df.iterrows():
                self._process_candle_data(row, timestamp)
            
            # 처리된 캔들 개수 저장 및 VPVR 결과 업데이트
            self.processed_candle_count = len(df)
            
            print(f"✅ 세션 데이터 VPVR 업데이트 완료: {len(df)}개 캔들")
            print(f"   📊 활성 가격 구간: {len(self.price_bins)}개")
            print(f"   📊 총 거래량: {sum(self.volume_histogram.values()):.2f}")
            print(f"   📊 처리된 캔들: {self.processed_candle_count}개")
            
        except Exception as e:
            print(f"❌ 세션 데이터 로딩 오류: {e}")
    
    def _load_lookback_data(self):
        """lookback 기간만큼 과거부터 현재까지 데이터 로딩"""
        # lookback 기간만큼 충분한 데이터 확보
        hours_needed = 5
        data_manager = get_data_manager()
        df = data_manager.get_data_range(
            dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours_needed),
            dt.datetime.now(dt.timezone.utc)
            )
        
        # lookback 기간만큼만 사용 (최신 데이터부터)
        if len(df) > self.lookback: 
            df = df.tail(self.lookback)
        
        print(f"📊 룩백 데이터 로드: {len(df)}개 캔들 (요청: {self.lookback}개)")
        
        # VPVR에 데이터 직접 누적
        for timestamp, row in df.iterrows():
            self._process_candle_data(row, timestamp)
        
        # 처리된 캔들 개수 저장 및 VPVR 결과 업데이트
        self.processed_candle_count = len(df)
        
        print(f"✅ 룩백 데이터 VPVR 업데이트 완료: {len(df)}개 캔들")
        print(f"   📊 활성 가격 구간: {len(self.price_bins)}개")
        print(f"   📊 총 거래량: {sum(self.volume_histogram.values()):.2f}")
        print(f"   📊 처리된 캔들: {self.processed_candle_count}개")
            

    def _update_vpvr_result(self, session_config: Dict[str, any] = None):
        """현재 누적된 데이터로 VPVR 결과 업데이트"""
        try:
            if not self.volume_histogram:
                return
            
            active_bins = {k: v for k, v in self.volume_histogram.items() if v > 0}
            
            if not active_bins:
                return
            
            # POC (Point of Control) - 최대 거래량 가격대
            max_volume_bin = max(active_bins, key=active_bins.get)
            poc = self.price_bins[max_volume_bin]
            
            # 전체 거래량 대비 비율 계산
            total_volume = sum(active_bins.values())
            volume_ratios = {k: v / total_volume for k, v in active_bins.items()}
            
            # HVN (High Volume Node) - 고거래량 가격대
            mean_ratio = np.mean(list(volume_ratios.values()))
            std_ratio = np.std(list(volume_ratios.values()))
            
            hvn_candidates = {k: v for k, v in volume_ratios.items() if v > mean_ratio + std_ratio}
            if hvn_candidates:
                hvn_bin = max(hvn_candidates, key=lambda x: active_bins[x])
                hvn = self.price_bins[hvn_bin]
            else:
                hvn = poc
            
            # LVN (Low Volume Node) - 저거래량 가격대
            lvn_candidates = {k: v for k, v in volume_ratios.items() if v < mean_ratio - std_ratio}
            if lvn_candidates:
                lvn_bin = min(lvn_candidates, key=lambda x: active_bins[x])
                lvn = self.price_bins[lvn_bin]
            else:
                lvn = poc
            
            # 세션 정보와 함께 VPVR 결과 저장
            result = {
                "poc": poc,
                "hvn": hvn,
                "lvn": lvn,
                "total_volume": total_volume,
                "active_bins": len(active_bins),
                "data_count": len(self.volume_histogram),
                "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
                "mode": "session"
            }
            
            # SessionManager에서 세션 정보 추가
            if session_config:
                result.update({
                    "session": session_config.get('session_name'),
                    "session_start": session_config.get('session_start_time').isoformat() if session_config.get('session_start_time') else None,
                    "elapsed_minutes": session_config.get('elapsed_minutes', 0)
                })
            
            self.cached_result = result
            
        except Exception as e:
            print(f"❌ VPVR 결과 업데이트 오류: {e}")

    def update_with_candle(self, candle_data: pd.Series):
        """새로운 캔들 데이터로 VPVR 업데이트"""
        session_config = self.time_manager.get_indicator_mode_config()
        self._check_session_reset(session_config)

        # ATR 업데이트
        self.atr.update_with_candle(candle_data)
        
        # 가격 bin에 거래량 누적
        close_price = float(candle_data['close'])
        quote_volume = float(candle_data['quote_volume'])
        
        bin_key = self._get_price_bin_key(close_price)
        
        if bin_key not in self.volume_histogram:
            self.volume_histogram[bin_key] = 0
            self.price_bins[bin_key] = close_price
        
        self.volume_histogram[bin_key] += quote_volume
        
        # 처리된 캔들 개수 증가
        self.processed_candle_count += 1
        
        # VPVR 결과 업데이트
        self._update_vpvr_result()
        
        # 마지막 업데이트 시간 갱신
        self.last_update_time = dt.datetime.now(dt.timezone.utc)

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

    def _process_candle_data(self, row: pd.Series, timestamp):
        """캔들 데이터 처리 및 VPVR 업데이트"""
        close_price = float(row['close'])
        quote_volume = float(row['quote_volume'])

        self.atr.update_with_candle(row)

        # 가격 bin에 거래량 누적
        bin_key = self._get_price_bin_key(close_price)
        
        if bin_key not in self.volume_histogram:
            self.volume_histogram[bin_key] = 0
            self.price_bins[bin_key] = close_price
        
        self.volume_histogram[bin_key] += quote_volume
            

    def reset_session(self):
        """새 세션 시작 시 VPVR 리셋"""
        try:
            session_config = self.time_manager.get_indicator_mode_config()
            
            # 세션 VPVR 데이터 초기화
            self.price_bins = {}
            self.volume_histogram = {}
            self.cached_result = None
            self.last_update_time = None
            
            session_name = session_config.get('session_name', 'UNKNOWN')
            print(f"🔄 {session_name} 세션 VPVR 리셋 완료")
            
        except Exception as e:
            print(f"❌ 세션 리셋 오류: {e}")

    def _get_price_bin_key(self, price: float) -> str:
        """가격을 bin 키로 변환 (동적 bin 크기 사용)"""
        bin_size = self._calculate_dynamic_bin_size(price)
        bin_index = int(price / bin_size)
        bin_key = f"bin_{bin_index}"
        
        # price_bins에 실제 가격 저장
        if bin_key not in self.price_bins:
            self.price_bins[bin_key] = price
        else:
            # 이미 존재하는 경우 평균 가격으로 업데이트
            self.price_bins[bin_key] = (self.price_bins[bin_key] + price) / 2
        
        return bin_key
    
    def _calculate_dynamic_bin_size(self, price: float) -> float:
        """동적 bin 크기 계산"""
        try:
            # 1. Tick size (ETHUSDT는 0.01)
            tick_size = 0.01
            
            # 2. 0.05% = 5bp
            price_based_size = 0.0005 * price
            
            # 3. 3분 ATR의 20% (ATR 객체에서 직접 가져오기)
            atr_value = self.atr.get_status().get('atr')
            atr_size = atr_value * 0.2
            
            # 4. 최종 bin 크기 계산
            bin_size = max(
                10 * tick_size,        # 0.1 (노이즈 억제)
                price_based_size,      # 가격 비례
                atr_size               # 변동성 반영
            )
            
            return bin_size
            
        except Exception as e:
            print(f"❌ 동적 bin 크기 계산 오류: {e}")
            # 기본값 반환
            return max(0.1, price * 0.001)
    
    def get_current_vpvr(self) -> Optional[Dict[str, any]]:
        """현재 VPVR 결과 반환"""
        return self.cached_result
        
    def _get_processed_candle_count(self) -> int:
        """처리된 캔들 개수 반환"""
        return self.processed_candle_count
    
    def get_status(self) -> Dict[str, any]:
        """현재 VPVR 상태 정보 반환 (POC 포함)"""
        try:
            # 기존 세션 정보
            session_config = self.time_manager.get_indicator_mode_config()
            status = {
                'is_session_active': session_config['use_session_mode'],
                'current_session': session_config.get('session_name'),
                'session_start': session_config.get('session_start_time').isoformat() if session_config.get('session_start_time') else None,
                'mode': session_config['mode'],
                'data_count': self._get_processed_candle_count(),
                'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
                'elapsed_minutes': session_config.get('elapsed_minutes'),
                'session_status': session_config.get('session_status', 'UNKNOWN')
            }
            
            # VPVR 핵심 데이터 추가
            if self.cached_result:
                status.update({
                    'poc': self.cached_result.get('poc'),
                    'hvn': self.cached_result.get('hvn'),
                    'lvn': self.cached_result.get('lvn'),
                    'total_volume': self.cached_result.get('total_volume'),
                    'active_bins': self.cached_result.get('active_bins'),
                    'data_count': self.cached_result.get('data_count'),
                    'last_update': self.cached_result.get('last_update'),
                    'mode': self.cached_result.get('mode')
                })
            
            # 기존 ATR 정보
            status['atr_status'] = {
                'atr': self.atr.get_status(),
                'is_ready': self.atr.is_ready(),
                'is_mature': len(self.atr.true_ranges) >= self.atr.length,
                'candles_count': len(self.atr.candles)
            }
            
            return status
            
        except Exception as e:
            print(f"❌ VPVR 상태 확인 오류: {e}")
            return {
                'is_session_active': False,
                'mode': 'error',
                'data_count': 0
            }