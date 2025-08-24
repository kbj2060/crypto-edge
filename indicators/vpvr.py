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
from datetime import datetime, timezone
from utils.time_manager import get_time_manager
from data.data_manager import get_data_manager


class SessionVPVR:
    """
    세션 기반 실시간 VPVR 관리 클래스
    
    세션 시작 시 리셋하고, 3분봉 닫힐 때마다 실시간 업데이트합니다.
    공용 데이터를 사용하여 효율적으로 VPVR을 계산합니다.
    """
    
    def __init__(self, bins: int = 50, price_bin_size: float = 0.05, 
                 lookback: int = 100, auto_load: bool = True):
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
        from indicators.atr import ATR3M
        self.atr = ATR3M(length=14)
        
        # 초기 데이터 로딩
        if auto_load:
            self._auto_load_initial_data()
    
    def _auto_load_initial_data(self):
        """초기화 시 자동으로 적절한 데이터 로딩"""
        try:
            print("🚀 VPVR 초기 데이터 자동 로딩 시작...")
            
            session_config = self.time_manager.get_indicator_mode_config()
            
            if session_config['use_session_mode']:
                print(f"📊 세션 모드: {session_config['session_name']} 세션 시작부터 현재까지 데이터 로딩")
                self._load_session_data(session_config)
            else:
                print(f"📊 룩백 모드: 최근 {self.lookback}개 3분봉 데이터 로딩")
                self._load_lookback_data()
                
            print("✅ VPVR 초기 데이터 로딩 완료")
            
        except Exception as e:
            print(f"❌ VPVR 초기 데이터 로딩 오류: {e}")
    
    def _load_session_data(self, session_config: Dict[str, any]):
        """세션 시작부터 현재까지의 데이터 로딩"""
        try:
            from data.binance_dataloader import BinanceDataLoader
            
            dataloader = BinanceDataLoader()
            session_start = session_config.get('session_start_time')
            
            if not session_start:
                print("⚠️ 세션 시작 시간을 찾을 수 없습니다")
                return
            
            # 세션 시작 시간을 datetime 객체로 변환
            if isinstance(session_start, str):
                session_start = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
            
            # 바이낸스에서 데이터 가져오기 (24시간)
            df = dataloader.fetch_recent_3m('ETHUSDT', hours=24)
            
            if df is None or df.empty:
                print("❌ 바이낸스 데이터 로드 실패")
                return
            
            # 세션 시작 이후 데이터만 필터링
            session_data = df[df.index >= session_start]
            
            if session_data.empty:
                print("⚠️ 세션 시작 이후 데이터가 없습니다")
                return
            
            print(f"📊 세션 데이터 로드: {len(session_data)}개 캔들")
            
            # VPVR에 데이터 직접 누적
            for timestamp, row in session_data.iterrows():
                self._process_candle_data(row, timestamp)
            
            # 처리된 캔들 개수 저장 및 VPVR 결과 업데이트
            self.processed_candle_count = len(session_data)
            self._update_vpvr_result(session_config)
            self.last_update_time = datetime.now(timezone.utc)
            
            print(f"✅ 세션 데이터 VPVR 업데이트 완료: {len(session_data)}개 캔들")
            print(f"   📊 활성 가격 구간: {len(self.price_bins)}개")
            print(f"   📊 총 거래량: {sum(self.volume_histogram.values()):.2f}")
            print(f"   📊 처리된 캔들: {self.processed_candle_count}개")
            
        except Exception as e:
            print(f"❌ 세션 데이터 로딩 오류: {e}")
    
    def _load_lookback_data(self):
        """lookback 기간만큼 과거부터 현재까지 데이터 로딩"""
        try:
            from data.binance_dataloader import BinanceDataLoader
            
            dataloader = BinanceDataLoader()
            
            # lookback 기간만큼 충분한 데이터 확보
            hours_needed = max(6, (self.lookback * 3) // 60 + 2)
            df = dataloader.fetch_recent_3m('ETHUSDT', hours=hours_needed)
            
            if df is None or df.empty:
                print("❌ 바이낸스 데이터 로드 실패")
                return
            
            # lookback 기간만큼만 사용 (최신 데이터부터)
            if len(df) > self.lookback:
                df = df.tail(self.lookback)
            
            print(f"📊 룩백 데이터 로드: {len(df)}개 캔들 (요청: {self.lookback}개)")
            
            # VPVR에 데이터 직접 누적
            for timestamp, row in df.iterrows():
                self._process_candle_data(row, timestamp)
            
            # 처리된 캔들 개수 저장 및 VPVR 결과 업데이트
            self.processed_candle_count = len(df)
            self._update_vpvr_result()
            self.last_update_time = datetime.now(timezone.utc)
            
            print(f"✅ 룩백 데이터 VPVR 업데이트 완료: {len(df)}개 캔들")
            print(f"   📊 활성 가격 구간: {len(self.price_bins)}개")
            print(f"   📊 총 거래량: {sum(self.volume_histogram.values()):.2f}")
            print(f"   📊 처리된 캔들: {self.processed_candle_count}개")
            
        except Exception as e:
            print(f"❌ 룩백 데이터 로딩 오류: {e}")
    
    def _process_candle_data(self, row: pd.Series, timestamp: datetime):
        """캔들 데이터 처리 및 VPVR 업데이트"""
        try:
            close_price = float(row['close'])
            volume = float(row.get('quote_volume', row.get('volume', 0)))
            
            # ATR 업데이트
            candle_data = {
                'timestamp': timestamp,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': close_price,
                'volume': float(row.get('volume', 0))
            }
            self.atr.update_with_candle(candle_data)
            
            # 가격 bin에 거래량 누적
            bin_key = self._get_price_bin_key(close_price)
            
            if bin_key not in self.volume_histogram:
                self.volume_histogram[bin_key] = 0
                self.price_bins[bin_key] = close_price
            
            self.volume_histogram[bin_key] += volume
            
        except Exception as e:
            print(f"❌ 캔들 데이터 처리 오류: {e}")
    
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
    
    def update_with_candle(self, candle_data: Dict[str, any]):
        """3분봉 닫힐 때마다 VPVR 업데이트"""
        try:
            session_config = self.time_manager.get_indicator_mode_config()
            
            if session_config['use_session_mode']:
                session_name = session_config.get('session_name', 'UNKNOWN')
                print(f"🔄 세션 진행 중 - {session_name} 세션 VPVR 업데이트")
                self._update_session_vpvr(candle_data, session_config)
            else:
                print(f"📊 세션 없음 - Lookback 데이터로 VPVR 업데이트")
                self._update_lookback_vpvr(candle_data)
                
        except Exception as e:
            print(f"❌ 캔들 업데이트 오류: {e}")
    
    def _update_session_vpvr(self, candle_data: Dict[str, Any], session_config: Dict[str, Any]):
        """세션 VPVR 업데이트"""
        try:
            # 캔들 데이터 처리
            row = pd.Series(candle_data)
            self._process_candle_data(row, candle_data.get('timestamp', datetime.now(timezone.utc)))
            
            # VPVR 결과 업데이트
            self._update_vpvr_result(session_config)
            self.last_update_time = datetime.now(timezone.utc)
            
        except Exception as e:
            print(f"❌ 세션 VPVR 업데이트 오류: {e}")

    def update_with_dataframe(self, df: pd.DataFrame):
        """DataFrame으로 VPVR 일괄 업데이트"""
        try:
            if df is None or df.empty:
                return
            
            print(f"📊 VPVR DataFrame 일괄 업데이트: {len(df)}개 캔들")
            
            # 세션 설정 (기본값)
            session_config = {
                'session_start': datetime.now(timezone.utc),
                'elapsed_minutes': 0
            }
            
            # 각 캔들을 순차적으로 처리
            for timestamp, row in df.iterrows():
                candle_data = {
                    'timestamp': timestamp,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                
                self._update_session_vpvr(candle_data, session_config)
            
            print(f"✅ VPVR DataFrame 업데이트 완료: {len(df)}개 캔들 처리됨")
            
        except Exception as e:
            print(f"❌ VPVR DataFrame 업데이트 오류: {e}")
    
    def _update_lookback_vpvr(self, candle_data: Dict[str, any]):
        """세션 외 시간대에서의 VPVR 업데이트"""
        try:
            # 새로운 캔들 데이터 추가
            self.lookback_data.append(candle_data)
            
            # lookback 길이 제한 (오래된 데이터 제거)
            if len(self.lookback_data) > self.lookback:
                self.lookback_data.pop(0)
            
            # 최소 5개 캔들이 모이면 VPVR 계산
            if len(self.lookback_data) >= 5:
                self._calculate_lookback_vpvr()
                self.last_update_time = datetime.now(timezone.utc)
                print(f"   ✅ Lookback VPVR 계산 완료")
            else:
                print(f"   ⏳ VPVR 계산 대기 중... ({len(self.lookback_data)}/5)")
                
        except Exception as e:
            print(f"❌ Lookback VPVR 업데이트 오류: {e}")
    
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
            atr_value = self.atr.get_atr()
            atr_size = atr_value * 0.2
            
            # 4. 최종 bin 크기 계산
            bin_size = max(
                10 * tick_size,        # 0.1 (노이즈 억제)
                price_based_size,      # 가격 비례
                atr_size               # 변동성 반영
            )
            
            # 디버깅용 로깅 (처음 몇 번만)
            if hasattr(self, '_bin_size_log_count'):
                self._bin_size_log_count += 1
            else:
                self._bin_size_log_count = 1
            
            if self._bin_size_log_count <= 3:  # 처음 3번만 로깅
                print(f"   🔍 동적 bin 크기 계산: tick={10*tick_size:.3f}, price={price_based_size:.3f}, atr={atr_size:.3f} (ATR={atr_value:.3f}) → 최종={bin_size:.3f}")
            
            return bin_size
            
        except Exception as e:
            print(f"❌ 동적 bin 크기 계산 오류: {e}")
            # 기본값 반환
            return max(0.1, price * 0.001)
    
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
    
    def _calculate_vpvr_from_data(self, df: pd.DataFrame) -> Dict[str, float]:
        """DataFrame에서 VPVR 계산"""
        try:
            if df.empty or len(df) < 5:
                return {}
            
            # 가격 범위 계산
            price_min = float(df['low'].min())
            price_max = float(df['high'].max())
            
            # 가격 bin 생성
            # bin_size = (price_max - price_min) / self.bins
            # if bin_size == 0:
            #     bin_size = price_min * 0.001  # 최소 bin 크기
            
            # 거래량 히스토그램 생성
            volume_histogram = {}
            for _, row in df.iterrows():
                close_price = float(row['close'])
                # VPVR은 quote volume (USDT) 사용, 없으면 base volume (ETH)로 폴백
                volume = float(row['quote_volume'])
                
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
        """모든 세션의 VPVR 결과 반환 (TimeManager 기반)"""
        return self.time_manager.get_session_history()
    
    def get_vpvr_status(self) -> Dict[str, any]:
        """현재 VPVR 상태 정보 반환 (TimeManager 기반)"""
        try:
            # TimeManager에서 세션 정보 가져오기
            session_config = self.time_manager.get_indicator_mode_config()
            
            status = {
                'is_session_active': session_config['use_session_mode'],
                'current_session': session_config.get('session_name'),
                'session_start': session_config.get('session_start_time').isoformat() if session_config.get('session_start_time') else None,
                'mode': session_config['mode'],
                'data_count': self._get_processed_candle_count(session_config),
                'last_update': self.last_update_time.isoformat() if self.last_update_time else None,
                'elapsed_minutes': session_config.get('elapsed_minutes', 0),
                'session_status': session_config.get('session_status', 'UNKNOWN')
            }
            
            if session_config['use_session_mode']:
                status.update({
                    'active_bins': len(self.volume_histogram),
                    'total_volume': sum(self.volume_histogram.values()) if self.volume_histogram else 0,
                    'atr_status': {
                        'current_atr': self.atr.get_atr(),
                        'is_ready': self.atr.is_ready(),
                        'is_mature': len(self.atr.true_ranges) >= self.atr.length,
                        'candles_count': len(self.atr.candles)
                    }
                })
            else:
                status.update({
                    'lookback_data_count': len(self.lookback_data),
                    'lookback_limit': self.lookback,
                    'atr_status': {
                        'current_atr': self.atr.get_atr(),
                        'is_ready': self.atr.is_ready(),
                        'is_mature': len(self.atr.true_ranges) >= self.atr.length,
                        'candles_count': len(self.atr.candles)
                    }
                })
            
            return status
            
        except Exception as e:
            print(f"❌ VPVR 상태 확인 오류: {e}")
            return {
                'is_session_active': False,
                'mode': 'error',
                'data_count': 0
            }
    
    def _get_processed_candle_count(self, session_config: Dict[str, any]) -> int:
        """처리된 캔들 개수 반환"""
        return self.processed_candle_count
    
    def load_data_from_manager(self, start_time: datetime, end_time: datetime) -> bool:
        """DataManager에서 지정된 시간 범위의 데이터를 가져와서 VPVR 계산"""
        try:
            data_manager = get_data_manager()
            if not data_manager.is_ready():
                print("⚠️ DataManager가 준비되지 않았습니다")
                return False
            
            print(f"📊 DataManager에서 VPVR 데이터 로드: {start_time} ~ {end_time}")
            
            # DataManager에서 데이터 가져오기
            data = data_manager.get_data_range(start_time, end_time)
            
            if not data:
                print("⚠️ DataManager에서 해당 기간의 데이터를 찾을 수 없습니다")
                return False
            
            # 딕셔너리 리스트를 DataFrame으로 변환
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ DataManager에서 {len(df)}개 캔들 데이터 로드 완료")
            
            # VPVR 계산 및 업데이트
            self.update_with_dataframe(df)
            
            return True
            
        except Exception as e:
            print(f"❌ DataManager에서 VPVR 데이터 로드 오류: {e}")
            return False
    
    def load_session_data_from_manager(self) -> bool:
        """DataManager에서 현재 세션 데이터를 가져와서 VPVR 계산"""
        try:
            session_config = self.time_manager.get_indicator_mode_config()
            
            if not session_config['use_session_mode']:
                print("⚠️ 세션 모드가 활성화되지 않았습니다")
                return False
            
            session_start = session_config.get('session_start_time')
            if not session_start:
                print("⚠️ 세션 시작 시간을 찾을 수 없습니다")
                return False
            
            # 세션 시작 시간을 datetime 객체로 변환
            if isinstance(session_start, str):
                session_start = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
            
            current_time = datetime.now(timezone.utc)
            
            print(f"📊 DataManager에서 세션 데이터 로드: {session_start} ~ {current_time}")
            
            return self.load_data_from_manager(session_start, current_time)
            
        except Exception as e:
            print(f"❌ DataManager에서 세션 데이터 로드 오류: {e}")
            return False



