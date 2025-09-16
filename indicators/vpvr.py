"""
Production VPVR (Volume Profile Visible Range) Module
실전용 VPVR 지표 - 간단하고 안정적인 bin_size 계산
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
import datetime as dt
from data.data_manager import get_data_manager
from utils.time_manager import get_time_manager

class SessionVPVR:
    """
    실전용 세션 기반 VPVR 관리 클래스
    - 실제 바이낸스 3분봉 데이터 사용
    - 간단하고 안정적인 bin_size 계산
    - 실시간 업데이트 지원
    """

    def __init__(
        self,
        target_time: Optional[dt.datetime] = None,
        bins: int = 50,
        lookback: int = 150,
        volume_field: str = "quote_volume",
        hvn_sigma_factor: float = 0.5,
        lvn_sigma_factor: float = 0.5,
        top_n: int = 3,
        bottom_n: int = 3,
        min_vol_pct: float = 0.005,
    ):
        # 설정
        self.bins = bins
        self.lookback = lookback
        self.volume_field = volume_field
        self.hvn_sigma_factor = hvn_sigma_factor
        self.lvn_sigma_factor = lvn_sigma_factor
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.min_vol_pct = min_vol_pct

        # 핵심 데이터 구조
        self.price_bins: Dict[str, float] = {}
        self.volume_histogram: Dict[str, float] = {}
        self.cached_result: Optional[Dict[str, Any]] = None
        self.last_update_time: Optional[dt.datetime] = None
        self.processed_candle_count: int = 0

        # bin_size 관련
        self.bin_size: Optional[float] = None
        self.price_min: Optional[float] = None
        self.price_max: Optional[float] = None
        
        # 캔들 데이터
        self.candles = []
        self.time_manager = get_time_manager()
        self.target_time = target_time or self.time_manager.get_current_time()

        # 초기화
        self._initialize_vpvr()

    def _calculate_simple_bin_size(self, price_data: List[float]) -> float:
        """
        개선된 bin_size 계산
        가격 범위와 ATR을 고려한 안정적인 bin_size
        """
        if not price_data or len(price_data) < 2:
            current_price = price_data[0] if price_data else 50000
            return current_price * 0.001  # 0.1%
        
        price_min = min(price_data)
        price_max = max(price_data)
        price_range = price_max - price_min
        
        if price_range <= 0:
            return price_min * 0.001 if price_min > 0 else 1.0
        
        # 개선: ATR 기반 bin_size 계산
        try:
            # 간단한 ATR 계산
            prices = np.array(price_data)
            if len(prices) > 1:
                high_low = np.diff(prices)
                atr_estimate = np.mean(np.abs(high_low)) * 0.5
                atr_based_size = atr_estimate * 2  # ATR의 2배
            else:
                atr_based_size = price_min * 0.002
        except:
            atr_based_size = price_min * 0.002
        
        # 범위 기반 bin_size
        range_based_size = price_range / self.bins
        
        # 두 방법 중 더 작은 값 선택 (더 세밀한 분석)
        bin_size = min(range_based_size, atr_based_size)
        
        # 최소/최대 크기 제한
        min_bin_size = price_min * 0.0005 if price_min > 0 else 0.01
        max_bin_size = price_min * 0.01 if price_min > 0 else 1.0
        
        return max(min_bin_size, min(bin_size, max_bin_size))

    def _get_price_bin_key(self, price: float) -> str:
        """가격을 bin 키로 변환 - 개선된 버전"""
        if self.bin_size is None or self.bin_size <= 0:
            raise ValueError("bin_size가 초기화되지 않았습니다")
        
        # price_min 초기화 (한 번만 설정)
        if self.price_min is None:
            self.price_min = price
        
        # price_min 기준 상대적 위치 계산
        relative_price = price - self.price_min
        bin_index = int(math.floor(relative_price / self.bin_size))
        
        # 음수 인덱스 처리 - price_min을 동적으로 변경하지 않음
        if bin_index < 0:
            # 새로운 price_min으로 재계산하되, 기존 데이터는 유지
            new_price_min = price
            relative_price = price - new_price_min
            bin_index = int(math.floor(relative_price / self.bin_size))
            # price_min은 변경하지 않고 bin_index만 조정
            bin_index = max(0, bin_index)
        
        bin_key = f"bin_{bin_index}"
        
        # bin center 계산 - 안정적인 계산
        center_price = self.price_min + (bin_index * self.bin_size) + (self.bin_size / 2)
        
        # bin center 저장 (항상 업데이트)
        self.price_bins[bin_key] = center_price
        
        return bin_key

    def _initialize_vpvr(self):
        """초기 데이터 로드 및 bin_size 설정"""
        try:
            self._load_lookback_data()
            
            if not self.candles:
                self.bin_size = 1.0
                return
            
            # 가격 데이터 추출
            price_data = []
            for candle in self.candles:
                high = float(candle.get('high', 0))
                low = float(candle.get('low', 0))
                close = float(candle.get('close', 0))
                price_data.extend([high, low, close])
            
            # 유효한 가격만 필터링
            price_data = [p for p in price_data if p > 0]
            
            if price_data:
                self.price_min = min(price_data)
                self.price_max = max(price_data)
                self.bin_size = self._calculate_simple_bin_size(price_data)
            else:
                self.bin_size = 1.0
            
            # 초기 VPVR 계산
            self._recalculate_vpvr()
            
        except Exception as e:
            print(f"VPVR 초기화 오류: {e}")
            self.bin_size = 1.0

    def _load_lookback_data(self):
        """실제 데이터 로드"""
        try:
            hours_needed = max(1, int(self.lookback * 3 / 60))
            data_manager = get_data_manager()
            
            start_time = self.target_time - dt.timedelta(hours=hours_needed)
            end_time = self.target_time
            
            # 실제 바이낸스 데이터 가져오기
            df = data_manager.get_data_range(start_time, end_time)

            if df is None or df.empty:
                return

            if len(df) > self.lookback:
                df = df.tail(self.lookback)

            # DataFrame을 candles 리스트로 변환
            for timestamp, row in df.iterrows():
                row_with_timestamp = row.copy()
                row_with_timestamp.name = timestamp
                self.candles.append(row_with_timestamp)
            
            self.processed_candle_count = len(df)
            
        except Exception as e:
            print(f"데이터 로드 오류: {e}")

    def _recalculate_vpvr(self):
        """전체 VPVR 재계산"""
        try:
            self.volume_histogram.clear()
            self.price_bins.clear()
            self.processed_candle_count = 0
            
            for candle in self.candles:
                self._process_single_candle(candle)
            
            self._update_vpvr_result()
            
        except Exception as e:
            print(f"VPVR 재계산 오류: {e}")

    def _process_single_candle(self, candle: pd.Series):
        """단일 캔들 처리"""
        try:
            close_price = float(candle.get('close', 0))
            if close_price <= 0:
                return
            
            # 볼륨 데이터
            volume = float(candle.get(self.volume_field, 0))
            if volume <= 0:
                volume = float(candle.get('volume', 0)) * close_price
            
            if volume <= 0:
                return
            
            # bin 키 계산 및 히스토그램 업데이트
            bin_key = self._get_price_bin_key(close_price)
            
            if bin_key not in self.volume_histogram:
                self.volume_histogram[bin_key] = 0.0
            
            self.volume_histogram[bin_key] += volume
            self.processed_candle_count += 1
            
        except Exception as e:
            print(f"캔들 처리 오류: {e}")

    def update_with_candle(self, candle_data: pd.Series):
        """새로운 캔들로 업데이트"""
        try:
            self.target_time = candle_data.name if hasattr(candle_data, 'name') and candle_data.name else dt.datetime.now(dt.timezone.utc)
            
            # 새 캔들 추가
            self.candles.append(candle_data)
            
            # lookback 제한
            if len(self.candles) > self.lookback:
                self.candles = self.candles[-self.lookback:]
            
            # 전체 재계산
            self._recalculate_vpvr()
            
        except Exception as e:
            print(f"캔들 업데이트 오류: {e}")

    def _update_vpvr_result(self):
        """VPVR 결과 계산"""
        try:
            if not self.volume_histogram:
                self.cached_result = None
                return
            
            active_bins = {k: v for k, v in self.volume_histogram.items() if v > 0}
            if not active_bins:
                self.cached_result = None
                return
            
            total_volume = sum(active_bins.values())
            if total_volume <= 0:
                self.cached_result = None
                return
            
            # POC (Point of Control) - 가장 높은 볼륨
            poc_bin = max(active_bins, key=active_bins.get)
            poc_price = self.price_bins.get(poc_bin)
            
            # POC 가격이 없으면 현재 가격 사용
            if poc_price is None:
                # 현재 가격 추정 (가장 최근 캔들의 종가)
                poc_price = float(self.candles[-1].get('close'))
            
            # HVN/LVN 계산
            volume_ratios = {k: (v / total_volume) for k, v in active_bins.items()}
            ratios_arr = np.array(list(volume_ratios.values()))
            
            if len(ratios_arr) > 1:
                mean_ratio = float(np.mean(ratios_arr))
                std_ratio = float(np.std(ratios_arr))
                
                hvn_threshold = mean_ratio + (self.hvn_sigma_factor * std_ratio)
                lvn_threshold = mean_ratio - (self.lvn_sigma_factor * std_ratio)
                
                # HVN 후보 (POC 제외)
                hvn_candidates = [k for k, r in volume_ratios.items() 
                                if (r > hvn_threshold and k != poc_bin)]
                
                # LVN 후보 (POC 제외)
                lvn_candidates = [k for k, r in volume_ratios.items() 
                                if (r < lvn_threshold and k != poc_bin)]
                
                # fallback: sigma 기반으로 찾지 못하면 상위/하위 N개 사용
                if not hvn_candidates:
                    sorted_desc = sorted(((k, v) for k, v in active_bins.items() if k != poc_bin), 
                                        key=lambda x: x[1], reverse=True)
                    hvn_candidates = [k for k, _ in sorted_desc[:self.top_n]]
                
                if not lvn_candidates:
                    sorted_asc = sorted(((k, v) for k, v in active_bins.items() if k != poc_bin), 
                                        key=lambda x: x[1])
                    lvn_candidates = [k for k, _ in sorted_asc[:self.bottom_n]]
                
                # 최종 HVN/LVN 선택
                hvn_bin = hvn_candidates[0] if hvn_candidates else poc_bin
                hvn_price = self.price_bins.get(hvn_bin, poc_price)
                
                lvn_bin = (min(lvn_candidates, key=lambda k: active_bins.get(k, float('inf'))) 
                            if lvn_candidates else poc_bin)
                lvn_price = self.price_bins.get(lvn_bin, poc_price)
                
            else:
                # 단일 bin인 경우
                hvn_price = poc_price
                lvn_price = poc_price
                hvn_bin = poc_bin
                lvn_bin = poc_bin
            
            self.cached_result = {
                "poc": poc_price,
                "poc_bin": poc_bin,
                "hvn": hvn_price,
                "hvn_bin": hvn_bin,
                "lvn": lvn_price,
                "lvn_bin": lvn_bin,
                "total_volume": total_volume,
                "active_bins": len(active_bins),
                "processed_candles": self.processed_candle_count,
                "bin_size": self.bin_size,
                "price_range": [self.price_min, self.price_max],
                "last_update": self.target_time.isoformat(),
            }
            print(f"VPVR 결과: {poc_price:.2f}, {hvn_price:.2f}, {lvn_price:.2f}")
            self.last_update_time = self.target_time
            
        except Exception as e:
            print(f"VPVR 결과 업데이트 오류: {e}")
            self.cached_result = None

    def get_current_vpvr(self) -> Optional[Dict[str, Any]]:
        """현재 VPVR 결과 반환"""
        return self.cached_result

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 정보 반환"""
        status = {
            'bin_size': self.bin_size,
            'price_range': [self.price_min, self.price_max] if self.price_min and self.price_max else None,
            'data_count': self.processed_candle_count,
            'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
            'active_bins': len(self.volume_histogram) if self.volume_histogram else 0,
            'lookback': self.lookback,
            'volume_field': self.volume_field
        }
        
        if self.cached_result:
            status.update({
                'poc': self.cached_result.get('poc'),
                'hvn': self.cached_result.get('hvn'),
                'lvn': self.cached_result.get('lvn'),
                'total_volume': self.cached_result.get('total_volume')
            })
        
        return status

    def _get_processed_candle_count(self) -> int:
        """처리된 캔들 수 반환"""
        return self.processed_candle_count

    def reset(self):
        """VPVR 데이터 초기화"""
        self.price_bins.clear()
        self.volume_histogram.clear()
        self.cached_result = None
        self.candles.clear()
        self.processed_candle_count = 0
        self.bin_size = None
        self.price_min = None
        self.price_max = None
        self.last_update_time = None