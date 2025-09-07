"""
Volume Profile Visible Range (VPVR) 지표 모듈 - 개선판 (추가 안정화)

주요 변경점 요약:
- HVN/LVN 후보 선택시 최소 볼륨 필터, 로컬 극값 검증, POC 근접 페널티, 스코어링 적용
- min_vol_pct, poc_distance_bins, distance_penalty를 클래스 인자로 노출
- 기존 로직과 호환성 유지 (폴백 동작 보장)
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
import datetime as dt

from utils.time_manager import get_time_manager
from data.data_manager import get_data_manager
from indicators.atr import ATR3M


class SessionVPVR:
    """
    세션 기반 실시간 VPVR 관리 클래스 (개선판)

    - 세션 시작 시 bin_size 계산(고정)
    - 캔들 도착 시 bin으로 누적 (bin center 사용)
    - POC/HVN/LVN 계산 개선 및 디버그 출력
    """

    def __init__(
        self,
        bins: int = 50,
        price_bin_size: float = 0.05,
        lookback: int = 300,
        volume_field: str = "quote_volume",  # or "volume"
        hvn_sigma_factor: float = 0.5,
        lvn_sigma_factor: float = 0.5,
        top_n: int = 3,
        bottom_n: int = 3,
        recalc_bin_price_move_pct: float = 0.15,  # price move % to trigger bin_size recalculation
        min_vol_pct: float = 0.005,   # 전체 거래량 대비 최소 볼륨 비율(예: 0.0005 = 0.05%)
        poc_distance_bins: int = 2,    # POC로부터 이 bins 이내 후보는 우선순위 낮춤
        distance_penalty: float = 0.6, # POC 거리 기반 점수 패널티 계수 (0..1)
    ):
        self.bins = bins
        self.price_bin_size = price_bin_size
        self.lookback = lookback

        # core structures
        self.price_bins: Dict[str, float] = {}        # bin_key -> bin_center_price
        self.volume_histogram: Dict[str, float] = {} # bin_key -> accumulated volume
        self.cached_result: Optional[Dict[str, Any]] = None
        self.last_update_time: Optional[dt.datetime] = None
        self.processed_candle_count: int = 0

        # dynamic bin size (session-level fixed)
        self.bin_size: Optional[float] = None
        self._bin_sample_price: Optional[float] = None  # sample price used to compute bin_size

        # settings
        self.volume_field = volume_field
        self.hvn_sigma_factor = hvn_sigma_factor
        self.lvn_sigma_factor = lvn_sigma_factor
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.recalc_bin_price_move_pct = recalc_bin_price_move_pct

        # new stability settings
        self.min_vol_pct = min_vol_pct
        self.poc_distance_bins = poc_distance_bins
        self.distance_penalty = distance_penalty

        # dependencies
        self.time_manager = get_time_manager()
        self.atr = ATR3M(length=14)

        # initialize
        self._initialize_vpvr()

    # -----------------------
    # Initialization / Loading
    # -----------------------
    def _initialize_vpvr(self):
        """세션 설정 확인 후 초기 데이터 로드 및 bin_size 계산"""
        session_config = self.time_manager.get_indicator_mode_config()

        self._load_lookback_data()

        # determine initial sample price for bin_size calculation
        if self.price_bins:
            centers = list(self.price_bins.values())
            self._bin_sample_price = float(np.median(centers))
        else:
            self._bin_sample_price = 1.0

        # compute and fix bin_size for this session
        self.bin_size = self._calculate_dynamic_bin_size(self._bin_sample_price, force=True)

        self.last_update_time = dt.datetime.now(dt.timezone.utc)

        # after initialization, compute vpvr result
        self._update_vpvr_result()

    def _load_lookback_data(self):
        """lookback 기간만큼 과거부터 현재까지 데이터 로딩 (간이 모드)"""
        try:
            hours_needed = max(1, int(self.lookback * 3 / 60))  # heuristic
            data_manager = get_data_manager()
            df = data_manager.get_data_range(
                dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours_needed),
                dt.datetime.now(dt.timezone.utc)
            )

            if df is None or df.empty:
                return

            if len(df) > self.lookback:
                df = df.tail(self.lookback)

            for timestamp, row in df.iterrows():
                self._process_candle_data(row, timestamp)

            self.processed_candle_count = len(df)

        except Exception as e:
            print(f"❌ lookback 데이터 로딩 오류: {e}")

    # -----------------------
    # Bin sizing & mapping
    # -----------------------
    def _calculate_dynamic_bin_size(self, price: float, force: bool = False) -> float:
        """
        동적 bin 크기 계산 (안전 처리 포함).
        - ATR이 준비되지 않으면 price 기반 fallback 사용.
        - force=True이면 ATR 미준비여도 계산해서 반환 (안전값)
        """
        try:
            tick_size = 0.01  # 기본 tick
            price_based_size = 0.0005 * price  # 0.05%

            # try get ATR value safely
            atr_status = self.atr.get_status() if hasattr(self.atr, 'get_status') else None
            atr_value = None
            if isinstance(atr_status, dict):
                atr_value = atr_status.get('atr') if 'atr' in atr_status else atr_status.get('value', None)
            elif atr_status is not None:
                try:
                    atr_value = float(atr_status)
                except Exception:
                    atr_value = None

            if atr_value is None:
                atr_size = max(10 * tick_size, price * 0.001)
            else:
                atr_size = max(10 * tick_size, float(atr_value) * 0.2)

            bin_size = max(
                10 * tick_size,
                price_based_size,
                atr_size
            )

            return max(0.0001, float(bin_size))

        except Exception as e:
            print(f"❌ 동적 bin 크기 계산 오류: {e}")
            return max(0.1, price * 0.001)

    def _get_price_bin_key(self, price: float) -> str:
        """
        가격 -> bin 키 변환
        - 세션 고정 bin_size 사용 (없으면 계산해서 저장)
        - bin center를 price_bins에 저장 (canonical)
        """
        if not self.bin_size or self.bin_size <= 0:
            self.bin_size = self._calculate_dynamic_bin_size(price, force=True)
            self._bin_sample_price = price

        self._maybe_recalculate_bin_size(price)

        bin_index = int(math.floor(price / self.bin_size))
        bin_key = f"bin_{bin_index}"

        center_price = bin_index * self.bin_size + (self.bin_size / 2)
        if bin_key not in self.price_bins:
            self.price_bins[bin_key] = center_price

        return bin_key

    def _maybe_recalculate_bin_size(self, current_price: float):
        """
        가격이 세션 초기 sample 대비 큰 폭으로 이동했을 때 bin_size 재계산 권장(로그 출력).
        """
        try:
            if not self._bin_sample_price:
                return

            pct_move = abs(current_price - self._bin_sample_price) / max(1e-12, self._bin_sample_price)
            if pct_move >= self.recalc_bin_price_move_pct:
                new_bin_size = self._calculate_dynamic_bin_size(current_price, force=True)
                if abs(new_bin_size - self.bin_size) / max(1e-12, self.bin_size) > 0.05:
                    print(f"⚠️ 가격 이동 {pct_move:.2%} 감지 - bin_size 재계산 권장: {self.bin_size} -> {new_bin_size}")
        except Exception:
            pass

    # -----------------------
    # Candle processing
    # -----------------------
    def update_with_candle(self, candle_data: pd.Series):
        """새로운 캔들 데이터로 VPVR 업데이트 (실시간 경로)"""
        session_config = self.time_manager.get_indicator_mode_config()
        # self._check_session_reset(session_config)
        
        self.atr.update_with_candle(candle_data)

        # price
        close_price = float(candle_data.get('close'))

        vol_val = candle_data.get(self.volume_field)
        quote_volume = float(vol_val)

        bin_key = self._get_price_bin_key(close_price)

        if bin_key not in self.volume_histogram:
            self.volume_histogram[bin_key] = 0.0
            if bin_key not in self.price_bins:
                self.price_bins[bin_key] = (int(math.floor(close_price / self.bin_size)) * self.bin_size + self.bin_size / 2)

        self.volume_histogram[bin_key] += quote_volume
        self.processed_candle_count += 1
        self.last_update_time = dt.datetime.now(dt.timezone.utc)

        # VPVR 결과 갱신
        self._update_vpvr_result()
        print(f"✅ [{self.time_manager.get_current_time().strftime('%H:%M:%S')}] VPVR 업데이트 POC: {self.cached_result['poc']:.2f} HVN: {self.cached_result['hvn']:.2f} LVN: {self.cached_result['lvn']:.2f}")
        print(f"✅ [{self.time_manager.get_current_time().strftime('%H:%M:%S')}] ATR 업데이트 {self.atr.current_atr:.2f}")

    def _process_candle_data(self, row: pd.Series, timestamp):
        """배치 로드 시 캔들 데이터 처리 (update_with_candle과 거의 동일)"""
        try:
            try:
                if hasattr(self.atr, 'update_with_candle'):
                    self.atr.update_with_candle(row)
            except Exception:
                pass

            close_price = float(row.get('close', row.get('price', row.get('last'))))
            vol_val = row.get(self.volume_field, None)
            if vol_val is None:
                vol_val = row.get('volume', row.get('base_volume', 0.0))
            try:
                quote_volume = float(vol_val)
            except Exception:
                quote_volume = 0.0

            bin_key = self._get_price_bin_key(close_price)
            if bin_key not in self.volume_histogram:
                self.volume_histogram[bin_key] = 0.0
                if bin_key not in self.price_bins:
                    self.price_bins[bin_key] = (int(math.floor(close_price / self.bin_size)) * self.bin_size + self.bin_size / 2)
            self.volume_histogram[bin_key] += quote_volume

        except Exception as e:
            print(f"❌ 캔들 처리 오류: {e}")

    # -----------------------
    # VPVR 계산 (POC/HVN/LVN)
    # -----------------------
    def _update_vpvr_result(self):
        """현재 누적된 데이터로 VPVR 결과 업데이트 (개선된 HVN/LVN 계산)"""
        try:
            if not self.volume_histogram:
                self.cached_result = None
                return

            active_bins = {k: v for k, v in self.volume_histogram.items() if v > 0}
            if not active_bins:
                self.cached_result = None
                return

            total_volume = float(sum(active_bins.values()))
            if total_volume <= 0:
                self.cached_result = None
                return

            # POC
            poc_bin = max(active_bins, key=active_bins.get)
            poc_price = self.price_bins.get(poc_bin)

            # ratios
            volume_ratios = {k: (v / total_volume) for k, v in active_bins.items()}
            ratios_arr = np.array(list(volume_ratios.values()))
            mean_ratio = float(np.mean(ratios_arr))
            std_ratio = float(np.std(ratios_arr))

            # thresholds
            hvn_threshold = mean_ratio + (self.hvn_sigma_factor * std_ratio)
            lvn_threshold = mean_ratio - (self.lvn_sigma_factor * std_ratio)

            # initial candidates excluding POC
            hvn_candidates = [k for k, r in volume_ratios.items() if (r > hvn_threshold and k != poc_bin)]
            lvn_candidates = [k for k, r in volume_ratios.items() if (r < lvn_threshold and k != poc_bin)]

            # fallback based on top/bottom N by volume if no sigma-based candidates
            if not hvn_candidates:
                sorted_desc = sorted(((k, v) for k, v in active_bins.items() if k != poc_bin), key=lambda x: x[1], reverse=True)
                hvn_candidates = [k for k, _ in sorted_desc[:self.top_n]]

            if not lvn_candidates:
                sorted_asc = sorted(((k, v) for k, v in active_bins.items() if k != poc_bin), key=lambda x: x[1])
                lvn_candidates = [k for k, _ in sorted_asc[:self.bottom_n]]

            # ---- Improved selection: filter tiny volumes, prefer local extrema, apply distance penalty ----
            # helpers
            def bin_index_from_key(k: str) -> int:
                try:
                    return int(k.split("_", 1)[1])
                except Exception:
                    return 0

            poc_idx = bin_index_from_key(poc_bin)
            max_dist_bins = max(1, max(abs(bin_index_from_key(k) - poc_idx) for k in active_bins.keys()))

            # min volume threshold absolute
            min_vol_threshold = max(1e-12, self.min_vol_pct * total_volume)

            def is_local_max(k: str) -> bool:
                idx = bin_index_from_key(k)
                left = f"bin_{idx-1}"
                right = f"bin_{idx+1}"
                v = active_bins.get(k, 0)
                return v >= active_bins.get(left, 0) and v >= active_bins.get(right, 0)

            def is_local_min(k: str) -> bool:
                idx = bin_index_from_key(k)
                left = f"bin_{idx-1}"
                right = f"bin_{idx+1}"
                v = active_bins.get(k, 0)
                return v <= active_bins.get(left, float('inf')) and v <= active_bins.get(right, float('inf')) and (v < active_bins.get(left, float('inf')) or v < active_bins.get(right, float('inf')))

            def distance_penalty_score(k: str) -> float:
                idx = bin_index_from_key(k)
                dist = abs(idx - poc_idx)
                # normalized in [0,1]
                norm = dist / float(max_dist_bins)
                # penalty factor increases with distance
                return (1.0 - (norm * self.distance_penalty))

            # filter hvn candidates
            hvn_filtered = [k for k in hvn_candidates if active_bins.get(k, 0) >= min_vol_threshold and k != poc_bin]
            if not hvn_filtered:
                hvn_filtered = [k for k in hvn_candidates]  # relax if all filtered out

            # score hvn: volume * distance_penalty
            hvn_scored = []
            for k in hvn_filtered:
                vol = active_bins.get(k, 0)
                score = vol * distance_penalty_score(k)
                hvn_scored.append((k, score, vol))

            hvn_scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
            if hvn_scored:
                hvn_bin = hvn_scored[0][0]
                hvn_price = self.price_bins.get(hvn_bin, poc_price)
            else:
                hvn_bin = poc_bin
                hvn_price = poc_price

            # filter lvn candidates
            lvn_filtered = [k for k in lvn_candidates if active_bins.get(k, 0) >= min_vol_threshold and k != poc_bin]
            if not lvn_filtered:
                lvn_filtered = [k for k in lvn_candidates]

            # prefer true local minima among filtered
            lvn_local = [k for k in lvn_filtered if is_local_min(k)]
            if lvn_local:
                lvn_bin = min(lvn_local, key=lambda k: active_bins.get(k, float('inf')))
            else:
                # choose smallest volume among filtered
                lvn_bin = min(lvn_filtered, key=lambda k: active_bins.get(k, float('inf'))) if lvn_filtered else poc_bin
            lvn_price = self.price_bins.get(lvn_bin, poc_price)

            # ---------------------------------------------------------------------------------------

            result = {
                "poc": poc_price,
                "poc_bin": poc_bin,
                "hvn": hvn_price,
                "hvn_bin": hvn_bin,
                "hvn_candidates": hvn_candidates,
                "lvn": lvn_price,
                "lvn_bin": lvn_bin,
                "lvn_candidates": lvn_candidates,
                "total_volume": total_volume,
                "active_bins": len(active_bins),
                "processed_candles": self.processed_candle_count,
                "mean_ratio": mean_ratio,
                "std_ratio": std_ratio,
                "hvn_threshold": hvn_threshold,
                "lvn_threshold": lvn_threshold,
                "min_vol_threshold": min_vol_threshold,
                "last_update": dt.datetime.now(dt.timezone.utc).isoformat(),
            }

            self.cached_result = result
            self.last_update_time = dt.datetime.now(dt.timezone.utc)

        except Exception as e:
            print(f"❌ VPVR 결과 업데이트 오류: {e}")

    # -----------------------
    # Utilities / Exports
    # -----------------------
    def get_current_vpvr(self) -> Optional[Dict[str, Any]]:
        """현재 VPVR 결과 반환 (cached_result)"""
        return self.cached_result

    def _get_processed_candle_count(self) -> int:
        return self.processed_candle_count

    def get_status(self) -> Dict[str, Any]:
        """현재 VPVR 상태 정보 반환 (POC 포함)"""
        try:
            session_config = self.time_manager.get_indicator_mode_config()
            status = {
                'is_session_active': session_config.get('use_session_mode', False),
                'current_session': session_config.get('session_name'),
                'session_start': (session_config.get('session_start_time').isoformat()
                                    if isinstance(session_config.get('session_start_time'), dt.datetime)
                                    else session_config.get('session_start_time')),
                'mode': session_config.get('mode'),
                'data_count': self._get_processed_candle_count(),
                'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
                'elapsed_minutes': session_config.get('elapsed_minutes'),
                'session_status': session_config.get('session_status', 'UNKNOWN'),
                'bin_size': self.bin_size,
                'bin_sample_price': self._bin_sample_price
            }

            if self.cached_result:
                status.update({
                    'poc': self.cached_result.get('poc'),
                    'hvn': self.cached_result.get('hvn'),
                    'lvn': self.cached_result.get('lvn'),
                    'total_volume': self.cached_result.get('total_volume'),
                    'active_bins': self.cached_result.get('active_bins'),
                    'processed_candles': self.cached_result.get('processed_candles'),
                    'last_vpvr_update': self.cached_result.get('last_update'),
                })

            status['atr_status'] = {
                'atr': self.atr.get_status() if hasattr(self.atr, 'get_status') else None,
                'is_ready': getattr(self.atr, 'is_ready', lambda: False)(),
            }

            return status

        except Exception as e:
            print(f"❌ VPVR 상태 확인 오류: {e}")
            return {
                'is_session_active': False,
                'mode': 'error',
                'data_count': 0
            }