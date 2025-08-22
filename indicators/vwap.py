#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VWAP (Volume Weighted Average Price) 지표
- 세션 기반 VWAP 계산
- VWAP 표준편차 계산
- 실시간 업데이트 지원
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from data.binance_dataloader import BinanceDataLoader
from .opening_range import get_session_manager

class SessionVWAP:
    """세션 기반 VWAP 관리 클래스"""
    
    def __init__(self, symbol: str = "ETHUSDT", auto_load: bool = True):
        self.symbol = symbol
        
        # 데이터 저장소
        self.session_data = []
        self.processed_candle_count = 0
        
        # VWAP 계산 결과
        self.current_vwap = 0.0
        self.current_vwap_std = 0.0
        self.cached_result = {}
        
        # 세션 관리
        self.session_manager = get_session_manager()
        self.last_session_name = None  # 세션 변경 감지용
        
        # 마지막 업데이트 시간
        self.last_update_time = None
        
        # 자동 데이터 로딩
        if auto_load:
            self._auto_load_initial_data()
        
        print(f"🚀 SessionVWAP 초기화 완료 ({symbol})")
    
    def _auto_load_initial_data(self):
        """초기 데이터 자동 로딩"""
        print("🚀 VWAP 초기 데이터 자동 로딩 시작...")
        
        session_config = self.session_manager.get_indicator_mode_config()
        
        if session_config['use_session_mode']:
            print("📊 세션 모드: 세션 시작부터 현재까지 데이터 로딩")
            self._load_session_data()
        else:
            print("📊 세션 외 시간: 최근 데이터로 VWAP 초기화")
            self._load_recent_data()
    
    def _load_session_data(self):
        """세션 시작부터 현재까지 데이터 로딩"""
        try:
            from data.binance_dataloader import BinanceDataLoader
            
            dataloader = BinanceDataLoader()
            session_config = self.session_manager.get_indicator_mode_config()
            session_start = session_config.get('session_start_time')
            
            if not session_start:
                print("⚠️ 세션 시작 시간을 찾을 수 없습니다")
                return
            
            # 세션 시작부터 현재까지의 3분봉 데이터 가져오기
            current_time = datetime.now(timezone.utc)
            
            # 세션 시작 시간을 datetime 객체로 변환
            if isinstance(session_start, str):
                session_start = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
            
            # 바이낸스에서 데이터 가져오기 (24시간)
            df = dataloader.fetch_recent_3m(self.symbol, hours=24)  # 충분한 데이터
            
            if df is None or df.empty:
                print("❌ 바이낸스 데이터 로드 실패")
                return
            
            print(f"✅ 바이낸스 데이터 로드 성공: {len(df)}개 캔들")
            print(f"📊 기간: {df.index[0]} ~ {df.index[-1]}")
            print(f"💰 평균 거래량: {df['volume'].mean():.2f} ETH")
            
            # 세션 시작 이후 데이터만 필터링 (인덱스가 close_time)
            session_data = df[df.index >= session_start]
            print(f"📊 세션 데이터 로드: {len(session_data)}개 캔들")
            
            # VWAP 계산
            self._calculate_session_vwap(session_data)
                
        except Exception as e:
            print(f"❌ 세션 데이터 로드 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_recent_data(self):
        """세션 외 시간용 데이터 로딩 - 이전 세션 종료 시점부터 현재까지"""
        try:
            from data.binance_dataloader import BinanceDataLoader
            
            dataloader = BinanceDataLoader()
            session_config = self.session_manager.get_indicator_mode_config()
            
            # 이전 세션 종료 시점 찾기
            previous_session_end = self._get_previous_session_end_time(session_config)
            
            if previous_session_end:
                print(f"📊 세션 외 시간: 이전 세션 종료 시점({previous_session_end.strftime('%H:%M')})부터 현재까지 데이터 로딩")
                
                # 이전 세션 종료 시점부터 현재까지의 데이터 가져오기
                df = dataloader.fetch_3m_data(
                    symbol=self.symbol,
                    start_time=previous_session_end,
                    end_time=datetime.now(timezone.utc)
                )
                
                if df is None or df.empty:
                    print("⚠️ 이전 세션 종료 시점부터 데이터가 없습니다. 최근 24시간 데이터 사용")
                    df = dataloader.fetch_recent_3m(self.symbol, hours=24)
            else:
                print("📊 세션 외 시간: 최근 24시간 데이터 로딩 (이전 세션 정보 없음)")
                df = dataloader.fetch_recent_3m(self.symbol, hours=24)
            
            if df is None or df.empty:
                print("❌ 바이낸스 데이터 로드 실패")
                return
            
            print(f"✅ 바이낸스 데이터 로드 성공: {len(df)}개 캔들")
            print(f"📊 기간: {df.index[0]} ~ {df.index[-1]}")
            print(f"💰 평균 거래량: {df['volume'].mean():.2f} ETH")
            
            # 세션 외 시간 데이터로 VWAP 계산
            self._calculate_session_vwap(df)
        
        except Exception as e:
            print(f"❌ 세션 외 시간 데이터 로드 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_previous_session_end_time(self, session_config: Dict[str, Any]) -> Optional[datetime]:
        """이전 세션 종료 시점 찾기"""
        try:
            # 현재 세션이 US인 경우, 이전 세션은 EU
            # 현재 세션이 EU인 경우, 이전 세션은 US
            # 세션 외 시간인 경우, 가장 최근에 끝난 세션 찾기
            
            current_session = session_config.get('session_name', 'NONE')
            current_time = datetime.now(timezone.utc)
            
            if current_session == 'US':
                # US 세션 중이면 이전 EU 세션 종료 시점
                # EU 세션은 보통 15:00 UTC에 끝남
                previous_end = current_time.replace(hour=15, minute=0, second=0, microsecond=0)
                if previous_end > current_time:
                    previous_end = previous_end - timedelta(days=1)
                return previous_end
                
            elif current_session == 'EU':
                # EU 세션 중이면 이전 US 세션 종료 시점
                # US 세션은 보통 22:00 UTC에 끝남
                previous_end = current_time.replace(hour=22, minute=0, second=0, microsecond=0)
                if previous_end > current_time:
                    previous_end = previous_end - timedelta(days=1)
                return previous_end
                
            else:
                # 세션 외 시간이면 가장 최근에 끝난 세션 찾기
                # 현재 시간이 15:00-22:00 UTC 사이면 EU 세션이 끝난 후
                # 현재 시간이 22:00-15:00 UTC 사이면 US 세션이 끝난 후
                current_hour = current_time.hour
                
                if 15 <= current_hour < 22:
                    # EU 세션이 끝난 후 (15:00 UTC)
                    previous_end = current_time.replace(hour=15, minute=0, second=0, microsecond=0)
                else:
                    # US 세션이 끝난 후 (22:00 UTC)
                    previous_end = current_time.replace(hour=22, minute=0, second=0, microsecond=0)
                    if previous_end > current_time:
                        previous_end = previous_end - timedelta(days=1)
                
                return previous_end
                
        except Exception as e:
            print(f"❌ 이전 세션 종료 시점 계산 오류: {e}")
            return None
    
    def _calculate_session_vwap(self, df: pd.DataFrame):
        """세션 데이터로 VWAP 계산"""
        try:
            if df.empty:
                return
            
            # 데이터 타입 확인 및 변환
            df = df.copy()
            for col in ['high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # NaN 값 제거
            df = df.dropna(subset=['high', 'low', 'close', 'volume'])
            
            if df.empty:
                print("⚠️ 유효한 데이터가 없습니다")
                return
            
            # VWAP 계산
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            volume_price = typical_price * df['volume']
            
            total_volume = df['volume'].sum()
            if total_volume > 0:
                self.current_vwap = float(volume_price.sum() / total_volume)
            else:
                self.current_vwap = 0.0
            
            # VWAP 표준편차 계산 (개선된 방식)
            if len(df) > 1:  # 최소 2개 캔들이 있어야 표준편차 계산 가능
                # 가격 변동성 기반 표준편차
                price_changes = df['close'].pct_change().dropna()
                if len(price_changes) > 0:
                    # ATR과 유사한 방식으로 변동성 계산
                    high_low_range = df['high'] - df['low']
                    typical_range = (df['high'] + df['low'] + df['close']) / 3
                    
                    # 가격 범위의 가중 평균을 표준편차로 사용
                    weighted_range = (high_low_range * df['volume']).sum() / total_volume
                    self.current_vwap_std = float(weighted_range * 0.5)  # 0.5 배수로 조정
                else:
                    self.current_vwap_std = 0.0
            else:
                # 단일 캔들의 경우 고가-저가 범위의 절반을 표준편차로 사용
                price_range = df['high'].iloc[0] - df['low'].iloc[0]
                self.current_vwap_std = float(price_range * 0.5)
            
            # 데이터 저장 (DataFrame 형태로 유지)
            self.session_data = df.to_dict('records')
            self.processed_candle_count = len(df)
            
            # 결과 업데이트
            self._update_vwap_result()
            
            print(f"✅ 세션 VWAP 계산 완료: {len(df)}개 캔들")
            print(f"   📊 VWAP: ${self.current_vwap:.2f}")
            print(f"   📊 VWAP 표준편차: ${self.current_vwap_std:.2f}")
            print(f"   📊 처리된 캔들: {self.processed_candle_count}개")
            print(f"   📊 데이터 범위: ${df['low'].min():.2f} ~ ${df['high'].max():.2f}")
            
        except Exception as e:
            print(f"❌ 세션 VWAP 계산 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def update_with_candle(self, candle_data: Dict[str, Any]):
        """새로운 캔들로 VWAP 업데이트"""
        try:
            # 세션 상태 업데이트
            self.session_manager.update_session_status()
            session_config = self.session_manager.get_indicator_mode_config()
            
            if session_config['use_session_mode']:
                print("🔄 세션 진행 중 - 세션 VWAP 업데이트")
                self._update_session_vwap(candle_data, session_config)
            else:
                print("🔄 세션 외 시간 - 세션 외 VWAP 업데이트")
                self._update_outside_session_vwap(candle_data, session_config)
                
        except Exception as e:
            print(f"❌ VWAP 업데이트 오류: {e}")
    
    def _update_session_vwap(self, candle_data: Dict[str, Any], session_config: Dict[str, Any]):
        """세션 VWAP 업데이트"""
        try:
            # 세션 변경 확인 및 리셋
            self._check_session_reset(session_config)
            
            # 새로운 캔들 추가
            self.session_data.append(candle_data)
            self.processed_candle_count += 1
            
            print(f"   📊 세션 데이터 누적: {len(self.session_data)}개 캔들")
            
            # VWAP 재계산
            df = pd.DataFrame(self.session_data)
            self._calculate_session_vwap(df)
            
            # 세션 정보 출력
            elapsed_minutes = session_config.get('elapsed_minutes', 0)
            print(f"   📊 세션 VWAP 업데이트 완료 - 거래량: {candle_data.get('volume', 0):.2f}, 가격: ${candle_data.get('close', 0):.2f}")
            print(f"   ⏱️  세션 진행 시간: {elapsed_minutes:.1f}분")
            print(f"   📊 누적 데이터: {len(self.session_data)}개 캔들")
        
        except Exception as e:
            print(f"❌ 세션 VWAP 업데이트 오류: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    def _update_outside_session_vwap(self, candle_data: Dict[str, Any], session_config: Dict[str, Any]):
        """세션 외 시간 VWAP 업데이트"""
        try:
            # 세션 변경 확인 및 리셋
            self._check_session_reset(session_config)
            
            # 새로운 캔들 추가
            self.session_data.append(candle_data)
            self.processed_candle_count += 1
            
            print(f"   📊 세션 외 데이터 누적: {len(self.session_data)}개 캔들")
            
            # VWAP 재계산
            df = pd.DataFrame(self.session_data)
            self._calculate_session_vwap(df)
            
            print(f"   📊 세션 외 VWAP 업데이트 완료 - 거래량: {candle_data.get('volume', 0):.2f}, 가격: ${candle_data.get('close', 0):.2f}")
            print(f"   📊 누적 데이터: {len(self.session_data)}개 캔들")
            
        except Exception as e:
            print(f"❌ 세션 외 VWAP 업데이트 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_vwap_result(self):
        """VWAP 결과 업데이트"""
        try:
            session_config = self.session_manager.get_indicator_mode_config()
        
            result = {
                "vwap": self.current_vwap,
                "vwap_std": self.current_vwap_std,
                "total_volume": sum([candle.get('volume', 0) for candle in self.session_data]),
                "data_count": self.processed_candle_count,
                "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
                "mode": "session" if session_config['use_session_mode'] else "outside_session"
            }
            
            # 세션 정보 추가
            if session_config['use_session_mode']:
                result.update({
                    "session": session_config.get('session_name'),
                    "session_start": session_config.get('session_start_time').isoformat() if session_config.get('session_start_time') else None,
                    "elapsed_minutes": session_config.get('elapsed_minutes', 0)
                })
            
            self.cached_result = result
            self.last_update_time = datetime.now(timezone.utc)
        
        except Exception as e:
            print(f"❌ VWAP 결과 업데이트 오류: {e}")
    
    def get_current_vwap(self) -> Dict[str, Any]:
        """현재 VWAP 결과 반환"""
        return self.cached_result
    
    def get_vwap_status(self) -> Dict[str, Any]:
        """VWAP 상태 정보 반환"""
        try:
            session_config = self.session_manager.get_indicator_mode_config()
            
            status = {
                "symbol": self.symbol,
                "current_vwap": self.current_vwap,
                "current_vwap_std": self.current_vwap_std,
                "data_count": self.processed_candle_count,
                "last_update": self.last_update_time.isoformat() if self.last_update_time else None,
                "session_status": session_config.get('session_status', 'UNKNOWN')
            }
            
            if session_config['use_session_mode']:
                status.update({
                    "session_name": session_config.get('session_name'),
                    "session_start": session_config.get('session_start_time').isoformat() if session_config.get('session_start_time') else None,
                    "elapsed_minutes": session_config.get('elapsed_minutes', 0),
                    "mode": "session"
                })
            else:
                status.update({
                    "mode": "outside_session",
                    "status": "세션 외 시간 VWAP 계산 중"
                })
            
            return status
        
        except Exception as e:
            print(f"❌ VWAP 상태 조회 오류: {e}")
            return {"error": str(e)}
    
    def reset_session(self):
        """세션 데이터 초기화"""
        self.session_data.clear()
        self.processed_candle_count = 0
        self.current_vwap = 0.0
        self.current_vwap_std = 0.0
        self.cached_result = {}
        self.last_update_time = None
        print("�� VWAP 세션 초기화 완료")
