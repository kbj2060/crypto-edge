import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime, timezone


class SessionVPVR:
    """
    세션 기반 실시간 VPVR 관리 클래스
    - 세션 시작 시 리셋
    - 3분봉 닫힐 때마다 실시간 업데이트
    - 세션별 최종 결과 저장
    - 세션 외 시간대에는 lookback 길이만큼만 처리
    """
    
    def __init__(self, bins: int = 50, price_bin_size: float = 0.05, lookback: int = 100):
        self.bins = bins
        self.price_bin_size = price_bin_size
        self.lookback = lookback
        
        # 현재 세션 VPVR 상태
        self.current_session = None
        self.current_session_start = None
        self.price_bins = {}
        self.volume_histogram = {}
        
        # 세션별 최종 결과 저장
        self.session_results = {}
        
        # VPVR 계산 결과 캐시
        self.cached_result = None
        self.last_update_time = None
        
        # 세션 외 시간대 처리용
        self.is_session_active = False
        self.lookback_data = []
    
    def reset_session(self, session_name: str, session_start_time: datetime):
        """새 세션 시작 시 VPVR 리셋"""
        try:
            # 이전 세션 결과 저장
            if self.current_session and self.cached_result:
                self.session_results[self.current_session] = {
                    'start_time': self.current_session_start,
                    'end_time': datetime.now(timezone.utc),
                    'vpvr_result': self.cached_result.copy()
                }
            
            # 새 세션 초기화
            self.current_session = session_name
            self.current_session_start = session_start_time
            self.price_bins = {}
            self.volume_histogram = {}
            self.cached_result = None
            self.last_update_time = None
            self.is_session_active = True
            
            print(f"🔄 {session_name} 세션 VPVR 리셋 완료")
            
        except Exception as e:
            print(f"❌ 세션 리셋 오류: {e}")
    
    def set_session_inactive(self):
        """세션 종료 시 비활성 상태로 설정"""
        try:
            if self.current_session:
                # 현재 세션 결과 저장
                if self.cached_result:
                    self.session_results[self.current_session] = {
                        'start_time': self.current_session_start,
                        'end_time': datetime.now(timezone.utc),
                        'vpvr_result': self.cached_result.copy()
                    }
                
                print(f"🌙 {self.current_session} 세션 종료, VPVR을 lookback 모드로 전환")
            
            self.is_session_active = False
            self.current_session = None
            self.current_session_start = None
            
        except Exception as e:
            print(f"❌ 세션 비활성화 오류: {e}")
    
    def update_with_candle(self, candle_data: Dict[str, any]):
        """3분봉 닫힐 때마다 VPVR 업데이트"""
        try:
            if self.is_session_active:
                self._update_session_vpvr(candle_data)
            else:
                self._update_lookback_vpvr(candle_data)
                
        except Exception as e:
            print(f"❌ 캔들 업데이트 오류: {e}")
    
    def _update_session_vpvr(self, candle_data: Dict[str, any]):
        """세션 활성 상태에서의 VPVR 업데이트"""
        close_price = float(candle_data['close'])
        volume = float(candle_data['volume'])
        
        bin_key = self._get_price_bin_key(close_price)
        
        if bin_key not in self.volume_histogram:
            self.volume_histogram[bin_key] = 0
            self.price_bins[bin_key] = close_price
        
        self.volume_histogram[bin_key] += volume
        self._update_vpvr_result()
        self.last_update_time = datetime.now(timezone.utc)
    
    def _update_lookback_vpvr(self, candle_data: Dict[str, any]):
        """세션 외 시간대에서의 VPVR 업데이트 (lookback 길이만큼만)"""
        self.lookback_data.append(candle_data)
        
        if len(self.lookback_data) > self.lookback:
            self.lookback_data.pop(0)
        
        if len(self.lookback_data) >= 5:
            self._calculate_lookback_vpvr()
            self.last_update_time = datetime.now(timezone.utc)
    
    def _calculate_lookback_vpvr(self):
        """lookback 데이터로 VPVR 계산"""
        try:
            if not self.lookback_data:
                return
            
            df_data = []
            for candle in self.lookback_data:
                df_data.append({
                    'timestamp': candle['timestamp'],
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle['volume'])
                })
            
            df = pd.DataFrame(df_data)
            vpvr_result = self._calculate_vpvr_from_data(df)
            
            if vpvr_result:
                self.cached_result = {
                    **vpvr_result,
                    'total_volume': df['volume'].sum(),
                    'active_bins': len(vpvr_result),
                    'session': 'LOOKBACK',
                    'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
                    'data_count': len(df),
                    'mode': 'lookback'
                }
            
        except Exception as e:
            print(f"❌ lookback VPVR 계산 오류: {e}")
    
    def _get_price_bin_key(self, price: float) -> str:
        """가격을 bin 키로 변환"""
        bin_size = price * self.price_bin_size
        bin_index = round(price / bin_size)
        return f"bin_{bin_index}"
    
    def _update_vpvr_result(self):
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
            
            self.cached_result = {
                "poc": poc,
                "hvn": hvn,
                "lvn": lvn,
                "total_volume": total_volume,
                "active_bins": len(active_bins),
                "session": self.current_session,
                "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
                "mode": "session"
            }
            
        except Exception as e:
            print(f"❌ VPVR 결과 업데이트 오류: {e}")
    
    def _calculate_vpvr_from_data(self, df: pd.DataFrame) -> Dict[str, float]:
        """DataFrame에서 VPVR 계산"""
        try:
            if df.empty or len(df) < 5:
                return {}
            
            # 가격 범위 계산
            price_min = float(df['low'].min())
            price_max = float(df['high'].max())
            
            # 가격 bin 생성
            bin_size = (price_max - price_min) / self.bins
            if bin_size == 0:
                bin_size = price_min * 0.001  # 최소 bin 크기
            
            # 거래량 히스토그램 생성
            volume_histogram = {}
            for _, row in df.iterrows():
                close_price = float(row['close'])
                volume = float(row['volume'])
                
                bin_key = self._get_price_bin_key(close_price)
                if bin_key not in volume_histogram:
                    volume_histogram[bin_key] = 0
                volume_histogram[bin_key] += volume
            
            if not volume_histogram:
                return {}
            
            # POC (Point of Control) - 최대 거래량 가격대
            max_volume_bin = max(volume_histogram, key=volume_histogram.get)
            poc = float(max_volume_bin.replace('bin_', '')) * bin_size + price_min
            
            # 전체 거래량 대비 비율 계산
            total_volume = sum(volume_histogram.values())
            volume_ratios = {k: v / total_volume for k, v in volume_histogram.items()}
            
            # HVN (High Volume Node) - 고거래량 가격대
            mean_ratio = np.mean(list(volume_ratios.values()))
            std_ratio = np.std(list(volume_ratios.values()))
            
            hvn_candidates = {k: v for k, v in volume_ratios.items() if v > mean_ratio + std_ratio}
            if hvn_candidates:
                hvn_bin = max(hvn_candidates, key=lambda x: volume_histogram[x])
                hvn = float(hvn_bin.replace('bin_', '')) * bin_size + price_min
            else:
                hvn = poc
            
            # LVN (Low Volume Node) - 저거래량 가격대
            lvn_candidates = {k: v for k, v in volume_ratios.items() if v < mean_ratio - std_ratio}
            if lvn_candidates:
                lvn_bin = min(lvn_candidates, key=lambda x: volume_histogram[x])
                lvn = float(lvn_bin.replace('bin_', '')) * bin_size + price_min
            else:
                lvn = poc
            
            return {
                "poc": poc,
                "hvn": hvn,
                "lvn": lvn
            }
            
        except Exception as e:
            print(f"❌ VPVR 계산 오류: {e}")
            return {}
    
    def get_current_vpvr(self) -> Optional[Dict[str, any]]:
        """현재 VPVR 결과 반환"""
        return self.cached_result
    
    def get_session_history(self) -> Dict[str, any]:
        """모든 세션의 VPVR 결과 반환"""
        return self.session_results.copy()
    
    def get_session_summary(self) -> Dict[str, any]:
        """현재 세션 요약 정보"""
        if self.is_session_active and self.current_session:
            return {
                'current_session': self.current_session,
                'session_start': self.current_session_start.isoformat() if self.current_session_start else None,
                'total_bins': len(self.volume_histogram),
                'active_bins': len([v for v in self.volume_histogram.values() if v > 0]),
                'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
                'current_vpvr': self.cached_result,
                'mode': 'session'
            }
        else:
            return {
                'current_session': 'LOOKBACK',
                'session_start': None,
                'total_bins': 0,
                'active_bins': 0,
                'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
                'current_vpvr': self.cached_result,
                'mode': 'lookback',
                'lookback_data_count': len(self.lookback_data),
                'lookback_length': self.lookback
            }
    
    def get_status_info(self) -> Dict[str, any]:
        """현재 VPVR 상태 정보"""
        return {
            'is_session_active': self.is_session_active,
            'current_session': self.current_session,
            'current_mode': 'session' if self.is_session_active else 'lookback',
            'lookback_data_count': len(self.lookback_data),
            'lookback_length': self.lookback,
            'has_vpvr_result': self.cached_result is not None
        }

