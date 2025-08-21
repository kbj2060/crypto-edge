#!/usr/bin/env python3
"""
통합 스마트 자동 트레이더 (리팩토링 버전)
실시간 청산 전략 + 세션 기반 전략 + 고급 청산 전략을 활용합니다.
"""

import time
import datetime
import threading
import requests
import json
import sqlite3
from typing import Dict, Any, Optional, List, Tuple
from core.trader_core import TraderCore
from analyzers.liquidation_analyzer import LiquidationAnalyzer
from analyzers.technical_analyzer import TechnicalAnalyzer
from handlers.websocket_handler import WebSocketHandler
from handlers.display_handler import DisplayHandler
from utils.trader_utils import get_next_5min_candle_time, format_time_delta
from config.integrated_config import IntegratedConfig
import pandas as pd
import numpy as np


class ExternalDataLoader:
    """외부 IP 서버 데이터베이스에서 초기 청산 데이터를 로드하는 클래스"""
    
    def __init__(self, server_url: str = "http://158.180.82.65", api_key: str = None):
        self.server_url = server_url
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def fetch_initial_liquidation_data(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """
        외부 서버에서 초기 청산 데이터를 가져옵니다.
        
        Args:
            symbol: 거래 심볼 (예: 'BTCUSDT')
            hours_back: 몇 시간 전까지의 데이터를 가져올지 (기본값: 24시간)
        
        Returns:
            청산 데이터 리스트
        """
        try:
            print(f"🔄 외부 서버에서 {symbol} 청산 데이터를 가져오는 중...")
            
            # 1) 단순 엔드포인트 우선 시도: http://<ip>/liquidations
            endpoint_simple = f"{self.server_url.rstrip('/')}/liquidations"
            response = self.session.get(endpoint_simple, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # 데이터가 리스트라고 가정하고 유연 매핑
            if isinstance(data, list):
                mapped: List[Dict] = []
                cutoff_ts = int(time.time()) - hours_back * 3600
                for item in data:
                    # 타임스탬프 파싱 (int/float/ISO 문자열 대응)
                    ts = item.get('timestamp') or item.get('ts') or item.get('time')
                    if isinstance(ts, str):
                        try:
                            import datetime as _dt
                            dt = _dt.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=_dt.timezone.utc)
                            ts = int(dt.timestamp())
                        except Exception:
                            try:
                                ts = int(float(ts))
                            except Exception:
                                ts = None
                    elif isinstance(ts, (int, float)):
                        ts = int(ts)
                    
                    price_raw = item.get('price') or item.get('markPrice') or item.get('p')
                    try:
                        price_val = float(price_raw) if price_raw is not None else 0.0
                    except Exception:
                        price_val = 0.0

                    # qty_usd 직접 제공되지 않으면 size*price로 계산
                    qty_usd_raw = item.get('qty_usd') or item.get('quantity_usd') or item.get('usd') or item.get('amount_usd')
                    if qty_usd_raw is None:
                        size_raw = item.get('size') or item.get('qty') or item.get('quantity') or item.get('amount')
                        try:
                            size_val = float(size_raw) if size_raw is not None else None
                        except Exception:
                            size_val = None
                        if size_val is not None and price_val is not None:
                            qty_val = size_val * price_val
                        else:
                            qty_val = 0.0
                    else:
                        try:
                            qty_val = float(qty_usd_raw)
                        except Exception:
                            qty_val = 0.0
                    symbol_val = item.get('symbol') or item.get('S') or symbol
                    side_raw = item.get('side') or item.get('direction') or item.get('s') or ''
                    side_norm = str(side_raw).lower()
                    if side_norm in ['buy', 'long']:
                        side = 'long'
                    elif side_norm in ['sell', 'short']:
                        side = 'short'
                    else:
                        side = 'unknown'
                    
                    if ts is not None and ts >= cutoff_ts:
                        mapped.append({
                            'timestamp': ts,
                            'symbol': symbol_val,
                            'side': side,
                            'qty_usd': qty_val,
                            'price': price_val
                        })
                
                print(f"✅ 외부 서버에서 {len(mapped)}개의 청산 데이터를 성공적으로 가져왔습니다.")
                return mapped
            
            # 2) 레거시 엔드포인트 백업: /api/liquidation/history
            endpoint_legacy = f"{self.server_url.rstrip('/')}/api/liquidation/history"
            params = {'symbol': symbol, 'hours_back': hours_back, 'limit': 1000}
            response2 = self.session.get(endpoint_legacy, params=params, timeout=30)
            response2.raise_for_status()
            data2 = response2.json()
            if isinstance(data2, dict) and data2.get('success'):
                liquidation_data = data2.get('data', [])
                print(f"✅ 외부 서버(레거시)에서 {len(liquidation_data)}개 데이터를 가져왔습니다.")
                return liquidation_data
            
            print("❌ 외부 서버 응답 형식을 인식하지 못했습니다.")
            return []
        except requests.exceptions.RequestException as e:
            print(f"❌ 외부 서버 연결 오류: {e}")
            return []
        except Exception as e:
            print(f"❌ 데이터 로딩 중 오류: {e}")
            return []
    
    def save_to_local_database(self, liquidation_data: List[Dict], db_path: str = "liquidation_data.db"):
        """
        외부에서 가져온 청산 데이터를 로컬 SQLite 데이터베이스에 저장합니다.
        
        Args:
            liquidation_data: 청산 데이터 리스트
            db_path: 로컬 데이터베이스 경로
        """
        if not liquidation_data:
            print("⚠️ 저장할 청산 데이터가 없습니다.")
            return
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 청산 데이터 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS liquidation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty_usd REAL NOT NULL,
                    price REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 데이터 삽입
            for data in liquidation_data:
                cursor.execute('''
                    INSERT INTO liquidation_history (timestamp, symbol, side, qty_usd, price)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    data.get('timestamp', int(time.time())),
                    data.get('symbol', 'BTCUSDT'),
                    data.get('side', 'unknown'),
                    data.get('qty_usd', 0.0),
                    data.get('price', 0.0)
                ))
            
            conn.commit()
            print(f"💾 {len(liquidation_data)}개의 청산 데이터를 로컬 데이터베이스에 저장했습니다.")
            
        except Exception as e:
            print(f"❌ 로컬 데이터베이스 저장 오류: {e}")
        finally:
            if conn:
                conn.close()
    
    def load_from_local_database(self, symbol: str, hours_back: int = 24, db_path: str = "liquidation_data.db") -> List[Dict]:
        """
        로컬 SQLite 데이터베이스에서 청산 데이터를 로드합니다.
        
        Args:
            symbol: 거래 심볼
            hours_back: 몇 시간 전까지의 데이터를 가져올지
            db_path: 로컬 데이터베이스 경로
        
        Returns:
            청산 데이터 리스트
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 테이블 존재 확인
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='liquidation_history'")
            if not cursor.fetchone():
                print("⚠️ 로컬 데이터베이스에 청산 데이터 테이블이 없습니다.")
                return []
            
            # 지정된 시간 범위 내의 데이터 조회
            cutoff_time = int(time.time()) - (hours_back * 3600)
            
            cursor.execute('''
                SELECT timestamp, symbol, side, qty_usd, price
                FROM liquidation_history
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 1000
            ''', (symbol, cutoff_time))
            
            rows = cursor.fetchall()
            
            liquidation_data = []
            for row in rows:
                liquidation_data.append({
                    'timestamp': row[0],
                    'symbol': row[1],
                    'side': row[2],
                    'qty_usd': row[3],
                    'price': row[4]
                })
            
            print(f"📂 로컬 데이터베이스에서 {len(liquidation_data)}개의 청산 데이터를 로드했습니다.")
            return liquidation_data
            
        except Exception as e:
            print(f"❌ 로컬 데이터베이스 로드 오류: {e}")
            return []
        finally:
            if conn:
                conn.close()


class IntegratedSmartTrader:
    """통합 스마트 자동 트레이더 (리팩토링 버전)"""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self.running = False
        
        # 외부 데이터 로더 초기화
        self.external_data_loader = ExternalDataLoader(
            server_url=getattr(config, 'external_server_url', '158.180.82.65'),
            api_key=getattr(config, 'external_api_key', None)
        )
        
        # 초기 청산 데이터 로드
        self._load_initial_liquidation_data()
        
        # 핵심 컴포넌트 초기화
        self.core = TraderCore(config)
        
        # 분석 엔진 초기화
        self.liquidation_analyzer = LiquidationAnalyzer(self.core.get_websocket())
        self.technical_analyzer = TechnicalAnalyzer(config)
        
        # 핸들러 초기화
        self.websocket_handler = WebSocketHandler(self.core.get_websocket())
        self.display_handler = DisplayHandler(self.core.get_websocket())
        
        # 상태 관리
        self.running = False
        self.last_analysis_time = None
        self.last_liquidation_analysis = None
        
        # 상태 및 통계 초기화
        self._init_state_and_stats()
        
        # 콜백 설정
        self._setup_callbacks()
    
    def _load_initial_liquidation_data(self):
        """외부 서버에서 초기 청산 데이터를 로드하고 AdvancedLiquidationStrategy에 전달"""
        try:
            print("🔄 초기 청산 데이터 로딩을 시작합니다...")
            
            # 외부 서버에서 청산 데이터 가져오기
            liquidation_data = self.external_data_loader.fetch_initial_liquidation_data(
                symbol=self.config.symbol,
                hours_back=getattr(self.config, 'initial_data_hours', 24)
            )
            
            if liquidation_data:
                # 로컬 데이터베이스에 저장
                self.external_data_loader.save_to_local_database(liquidation_data)
                
                # AdvancedLiquidationStrategy 초기화 및 데이터 전달
                self._initialize_advanced_liquidation_strategy(liquidation_data)
                
                print(f"✅ 초기 청산 데이터 로딩 완료: {len(liquidation_data)}개 레코드")
            else:
                # 외부 서버 연결 실패 시 로컬 데이터베이스에서 로드 시도
                print("⚠️ 외부 서버에서 청산 데이터를 가져올 수 없습니다. 로컬 데이터베이스를 확인합니다...")
                
                local_data = self.external_data_loader.load_from_local_database(
                    symbol=self.config.symbol,
                    hours_back=getattr(self.config, 'initial_data_hours', 24)
                )
                
                if local_data:
                    # 로컬 데이터로 AdvancedLiquidationStrategy 초기화
                    self._initialize_advanced_liquidation_strategy(local_data)
                    print(f"✅ 로컬 데이터베이스에서 {len(local_data)}개 레코드를 로드했습니다.")
                else:
                    print("⚠️ 로컬 데이터베이스에도 데이터가 없습니다. 실시간 데이터만 사용합니다.")
                
        except Exception as e:
            print(f"❌ 초기 청산 데이터 로딩 중 오류: {e}")
            
            # 오류 발생 시에도 로컬 데이터베이스에서 로드 시도
            try:
                print("🔄 오류 발생으로 인해 로컬 데이터베이스에서 데이터를 로드합니다...")
                local_data = self.external_data_loader.load_from_local_database(
                    symbol=self.config.symbol,
                    hours_back=getattr(self.config, 'initial_data_hours', 24)
                )
                
                if local_data:
                    self._initialize_advanced_liquidation_strategy(local_data)
                    print(f"✅ 로컬 데이터베이스에서 {len(local_data)}개 레코드를 로드했습니다.")
                else:
                    print("⚠️ 로컬 데이터베이스에도 데이터가 없습니다. 실시간 데이터만 사용합니다.")
            except Exception as local_error:
                print(f"❌ 로컬 데이터베이스 로드 중에도 오류 발생: {local_error}")
    
    def _initialize_advanced_liquidation_strategy(self, liquidation_data: List[Dict]):
        """AdvancedLiquidationStrategy를 초기화하고 히스토리 데이터를 전달"""
        try:
            from signals.advanced_liquidation_strategy import AdvancedLiquidationStrategy, AdvancedLiquidationConfig
            
            # AdvancedLiquidationStrategy 초기화
            adv_config = AdvancedLiquidationConfig()
            self._adv_liquidation_strategy = AdvancedLiquidationStrategy(adv_config)
            
            # 히스토리 데이터를 전략에 전달
            print("🔄 AdvancedLiquidationStrategy에 히스토리 데이터를 전달하는 중...")
            
            processed_count = 0
            long_count = 0
            short_count = 0
            
            for data in liquidation_data:
                # 데이터 형식 변환
                liquidation_event = {
                    'ts': data.get('timestamp', int(time.time())),
                    'side': data.get('side', 'unknown'),
                    'qty_usd': data.get('qty_usd', 0.0)
                }
                
                # 사이드 카운팅
                if liquidation_event['side'] == 'long':
                    long_count += 1
                elif liquidation_event['side'] == 'short':
                    short_count += 1
                
                # AdvancedLiquidationStrategy에 이벤트 전달
                self._adv_liquidation_strategy.process_liquidation_event(liquidation_event)
                processed_count += 1
            
            print(f"✅ AdvancedLiquidationStrategy 초기화 완료: {processed_count}개 히스토리 이벤트 처리됨")
            
            # 워밍업 상태 확인
            if hasattr(self._adv_liquidation_strategy, 'get_warmup_status'):
                warmup_status = self._adv_liquidation_strategy.get_warmup_status()
                print(f"🔥 초기 워밍업 상태: {warmup_status}")
            
        except Exception as e:
            print(f"❌ AdvancedLiquidationStrategy 초기화 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _init_state_and_stats(self):
        """상태 및 통계 초기화"""
        # 거래량 급증 집계
        self.volume_spike_buffer = []
        self.last_volume_summary = None
        self.volume_summary_cooldown = 30
    
    def _setup_callbacks(self):
        """웹소켓 콜백 설정"""
        callbacks = {
            'liquidation': lambda data: self._handle_liquidation_event(data),
            'volume': lambda data: self._handle_volume_spike(data),
            'price': lambda data: self.websocket_handler.on_price_update(
                data, 
                self._analyze_realtime_liquidation  # 청산 분석만 실행
            ),
            'kline': lambda data: self.websocket_handler.on_kline(
                data, 
                self._analyze_realtime_liquidation  # 청산 분석만 실행
            )
        }
        self.websocket_handler.setup_callbacks(callbacks)
        

    
    def _handle_liquidation_event(self, data: Dict):
        """청산 이벤트 처리 및 AdvancedLiquidationStrategy에 전달"""
        try:
            # 기본 청산 분석 실행
            self._analyze_realtime_liquidation(data)
            
            # AdvancedLiquidationStrategy가 초기화되지 않은 경우에만 생성
            if not hasattr(self, '_adv_liquidation_strategy') or self._adv_liquidation_strategy is None:
                print("⚠️ AdvancedLiquidationStrategy가 초기화되지 않았습니다. 새로 생성합니다.")
                from signals.advanced_liquidation_strategy import AdvancedLiquidationStrategy, AdvancedLiquidationConfig
                adv_config = AdvancedLiquidationConfig()
                self._adv_liquidation_strategy = AdvancedLiquidationStrategy(adv_config)
            
            strategy = self._adv_liquidation_strategy
            
            # 바이낸스 청산 데이터 형식에 맞게 처리
            if 'side' in data and 'qty_usd' in data:
                # 바이낸스 청산 데이터 형식: BUY=숏청산, SELL=롱청산
                # BUY: 숏 포지션이 강제 청산됨 (숏 청산)
                # SELL: 롱 포지션이 강제 청산됨 (롱 청산)
                side = 'short' if data['side'] == 'BUY' else 'long'
                
                # 청산 이벤트를 딕셔너리로 구성
                liquidation_event = {
                    'ts': int(data.get('timestamp', datetime.datetime.now(datetime.timezone.utc)).timestamp()),
                    'side': side,
                    'qty_usd': data['qty_usd']
                }
                
                strategy.process_liquidation_event(liquidation_event)
                
                # 청산 데이터가 들어올 때마다 고급 청산 전략 분석 실행
                websocket = self.core.get_websocket()
                if websocket and websocket.price_history:
                    advanced_signal = self._analyze_advanced_liquidation_strategy(websocket)
                    # 신호가 있을 때만 출력
                    if advanced_signal:
                        self._process_integrated_signal({
                            'advanced_liquidation_signal': advanced_signal
                        })
                
        except Exception as e:
            print(f"❌ 청산 이벤트 처리 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_volume_spike(self, volume_data: Dict):
        """거래량 급증 처리"""
        self.last_volume_summary = self.websocket_handler.on_volume_spike(
            volume_data, 
            self.volume_spike_buffer, 
            self.last_volume_summary,
            self.volume_summary_cooldown,
            self.display_handler.print_volume_spike_summary,
            self._analyze_realtime_liquidation
        )
    
    def _analyze_realtime_technical(self):
        """실시간 기술적 분석"""
        try:
            # 세션 기반 전략과 고급 청산 전략만 실행
            websocket = self.core.get_websocket()
            
            # 세션 기반 전략 분석
            session_signal = self._analyze_session_strategy(websocket)
            if session_signal:
                # 세션 전략 신호 직접 처리 (중립 포함 모든 신호)
                self._process_integrated_signal({
                    'session_signal': session_signal
                })
            else:
                # 세션 전략 분석은 실행되었지만 신호가 없는 경우
                print(f"📊 세션 전략: 분석 완료, 신호 없음")
            
            # 고급 청산 전략 분석
            advanced_liquidation_signal = self._analyze_advanced_liquidation_strategy(websocket)
            if advanced_liquidation_signal:
                # 고급 청산 전략 신호 직접 처리
                self._process_integrated_signal({
                    'advanced_liquidation_signal': advanced_liquidation_signal
                })
            else:
                # 고급 청산 전략 분석은 실행되었지만 신호가 없는 경우
                print(f"📊 고급 청산 전략: 분석 완료, 신호 없음")
                
        except Exception as e:
            print(f"❌ 실시간 기술적 분석 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_realtime_liquidation(self, data=None):
        """실시간 통합 청산 신호 분석 (ENHANCED_LIQUIDATION + Prediction 통합)"""
        try:
            # 현재 가격 가져오기
            websocket = self.core.get_websocket()
            if not websocket.price_history:
                return
            
            current_price = websocket.price_history[-1]['price']
            
            # 청산 통계 분석
            liquidation_stats = websocket.get_liquidation_stats(self.config.liquidation_window_minutes)
            volume_analysis = websocket.get_volume_analysis(3)
            
            # 통합 청산 신호 분석 (ENHANCED_LIQUIDATION + Prediction)
            integrated_liquidation_signal = self._analyze_integrated_liquidation(
                liquidation_stats, volume_analysis, current_price, websocket
            )
            
            # 청산 신호만 처리 (세션 전략은 정각 1분마다 별도 실행)
            if integrated_liquidation_signal:
                self._process_integrated_signal({
                    'liquidation_signal': integrated_liquidation_signal
                })
            
        except Exception as e:
            print(f"❌ 실시간 청산 분석 오류: {e}")
    
    def _analyze_session_strategy(self, websocket) -> Optional[Dict]:
        """세션 기반 전략 분석"""
        try:
            if not self.config.enable_session_strategy:
                return None
            
            # 1분봉 데이터 로드
            df_1m = self.core.get_data_loader().load_klines(
                self.config.symbol, 
                self.config.session_timeframe, 
                1500  # 현재 시간까지 커버하기 위해 더 증가
            )
            
            if df_1m.empty:
                return None
            
            # 키 레벨 계산 (전일 H/L, 스윙 레벨 등)
            key_levels = self._calculate_session_key_levels(df_1m)
            
            # 현재 시간 (UTC 명시)
            current_time = datetime.datetime.now(datetime.timezone.utc)
            
            # 세션 전략 분석 (직접 SessionBasedStrategy 사용)
            from signals.session_based_strategy import SessionBasedStrategy, SessionConfig
            session_config = SessionConfig()  # 기본 설정으로 생성
            session_strategy = SessionBasedStrategy(session_config)
            
            session_signal = session_strategy.analyze_session_strategy(
                df_1m, key_levels, current_time
            )
            
            return session_signal
            
        except Exception as e:
            print(f"❌ 세션 전략 분석 오류: {e}")
            return None
    
    def _calculate_session_key_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """세션 전략용 키 레벨 계산"""
        try:
            if df.empty:
                return {}
            
            # 전일 고가/저가/종가
            daily_data = df.resample('D').agg({
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
            
            if len(daily_data) < 2:
                return {}
            
            prev_day = daily_data.iloc[-2]
            
            # 최근 스윙 고점/저점 (20봉 기준)
            lookback = min(20, len(df))
            recent_data = df.tail(lookback)
            
            swing_high = recent_data['high'].max()
            swing_low = recent_data['low'].min()
            
            return {
                'prev_day_high': prev_day['high'],
                'prev_day_low': prev_day['low'],
                'prev_day_close': prev_day['close'],
                'last_swing_high': swing_high,
                'last_swing_low': swing_low
            }
            
        except Exception as e:
            print(f"❌ 세션 키 레벨 계산 오류: {e}")
            return {}
    
    def _analyze_advanced_liquidation_strategy(self, websocket) -> Optional[Dict]:
        """고급 청산 전략 분석"""
        try:
            if not self.config.enable_advanced_liquidation:
                return None
            
            # 1분봉 데이터 로드
            df_1m = self.core.get_data_loader().load_klines(
                self.config.symbol, 
                "1m", 
                500  # 충분한 데이터
            )
            
            if df_1m.empty:
                return None
            
            # 키 레벨 계산
            key_levels = self._calculate_session_key_levels(df_1m)
            
            # 오프닝 레인지 계산
            opening_range = self._calculate_opening_range(df_1m)
            
            # VWAP 및 표준편차 계산
            vwap, vwap_std = self._calculate_vwap_and_std(df_1m)
            
            # ATR 계산
            from indicators.atr import calculate_atr
            atr = calculate_atr(df_1m, 14)
            if pd.isna(atr):
                atr = df_1m['close'].iloc[-1] * 0.02  # 기본값
            
            # 기존에 생성된 AdvancedLiquidationStrategy 인스턴스 사용
            if hasattr(self, '_adv_liquidation_strategy'):
                adv_strategy = self._adv_liquidation_strategy
            else:
                # 새로 생성
                from signals.advanced_liquidation_strategy import AdvancedLiquidationStrategy, AdvancedLiquidationConfig
                adv_config = AdvancedLiquidationConfig()
                self._adv_liquidation_strategy = AdvancedLiquidationStrategy(adv_config)
                adv_strategy = self._adv_liquidation_strategy
            
            # # 워밍업 상태 및 청산 데이터 상태 확인
            warmup_status = adv_strategy.get_warmup_status()
            
            # 현재 청산 메트릭 확인
            try:
                metrics = adv_strategy.get_current_liquidation_metrics()
                if metrics and warmup_status['long_samples'] > 0 or warmup_status['short_samples'] > 0:
                    pass  # 메트릭 확인 완료
            except Exception as e:
                print(f"   ❌ 청산 메트릭 확인 실패: {e}")
            
            # 현재 가격
            current_price = df_1m['close'].iloc[-1]
            
            # 고급 청산 전략 분석 실행
            advanced_signal = adv_strategy.analyze_all_strategies(
                df_1m, key_levels, opening_range, vwap, vwap_std, atr
            )
            
            return advanced_signal
            
        except Exception as e:
            print(f"❌ 고급 청산 전략 분석 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_opening_range(self, df: pd.DataFrame) -> Dict[str, float]:
        """오프닝 레인지 계산"""
        try:
            if df.empty:
                return {}
            
            # 첫 15분 데이터
            or_minutes = 15
            if len(df) < or_minutes:
                return {}
            
            or_data = df.head(or_minutes)
            
            return {
                'high': or_data['high'].max(),
                'low': or_data['low'].min(),
                'center': (or_data['high'].max() + or_data['low'].min()) / 2,
                'range': or_data['high'].max() - or_data['low'].min()
            }
            
        except Exception as e:
            print(f"❌ 오프닝 레인지 계산 오류: {e}")
            return {}
    
    def _calculate_vwap_and_std(self, df: pd.DataFrame) -> Tuple[float, float]:
        """VWAP 및 표준편차 계산"""
        try:
            if df.empty:
                return 0.0, 0.0
            
            # 가격과 거래량으로 VWAP 계산
            vwap = np.average(df['close'], weights=df['volume'])
            
            # 표준편차 계산
            std = np.std(df['close'])
            
            return vwap, std
            
        except Exception as e:
            print(f"❌ VWAP 및 표준편차 계산 오류: {e}")
            return 0.0, 0.0
    

    
    def _analyze_integrated_liquidation(self, liquidation_stats: Dict, volume_analysis: Dict, current_price: float, websocket) -> Optional[Dict]:
        """통합 청산 신호 분석 (ENHANCED_LIQUIDATION + Prediction)"""
        try:
            # 기본 청산 신호 분석
            basic_signal = self.liquidation_analyzer.analyze_liquidation_signal(
                liquidation_stats, volume_analysis, current_price
            )
            
            # 청산 예측 분석
            recent_liquidations = websocket.get_recent_liquidations(self.config.liquidation_window_minutes)
            prediction_signal = self.core.get_integrated_strategy().analyze_liquidation_prediction(
                recent_liquidations, current_price
            )
            
            # 두 신호를 통합하여 최종 신호 생성
            if basic_signal and prediction_signal:
                # 둘 다 신호가 있는 경우 - 통합 효과
                return self._create_liquidation_integrated_signal(basic_signal, prediction_signal, current_price)
            elif basic_signal:
                # 기본 청산 신호만 있는 경우
                return basic_signal
            elif prediction_signal:
                # 예측 신호만 있는 경우 - 예측 신호를 기본 형태로 변환
                return self._convert_prediction_to_liquidation_signal(prediction_signal, current_price)
            else:
                return None
                
        except Exception as e:
            print(f"❌ 통합 청산 신호 분석 오류: {e}")
            return None
    
    def _create_liquidation_integrated_signal(self, basic_signal: Dict, prediction_signal: Dict, current_price: float) -> Dict:
        """청산 통합 신호 생성"""
        try:
            # 기본 신호 정보
            action = basic_signal.get('action', 'NEUTRAL')
            confidence = basic_signal.get('confidence', 0)
            
            # 예측 신호 정보
            pred_type = prediction_signal.get('type', 'UNKNOWN')
            pred_confidence = prediction_signal.get('confidence', 0)
            target_price = prediction_signal.get('target_price', current_price)
            
            # 통합 신뢰도 계산 (기본 + 예측)
            integrated_confidence = min(0.95, (confidence + pred_confidence) / 2 + 0.1)
            
            # 리스크 관리 (기본 신호 기준)
            if action == 'BUY':
                stop_loss = basic_signal.get('stop_loss', current_price * 0.98)
                take_profit1 = basic_signal.get('take_profit1', current_price * 1.04)
                take_profit2 = basic_signal.get('take_profit2', current_price * 1.06)
            elif action == 'SELL':
                stop_loss = basic_signal.get('stop_loss', current_price * 1.02)
                take_profit1 = basic_signal.get('take_profit1', current_price * 0.96)
                take_profit2 = basic_signal.get('take_profit2', current_price * 0.94)
            else:
                return basic_signal
            
            # 리스크/보상 비율 계산
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit1 - current_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            # 통합 이유 생성
            integrated_reason = f"청산 급증 + {pred_type} 예측 일치 | 신뢰도: {confidence:.1%} + {pred_confidence:.1%}"
            
            return {
                'signal_type': 'INTEGRATED_LIQUIDATION',
                'action': action,
                'confidence': integrated_confidence,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit1': take_profit1,
                'take_profit2': take_profit2,
                'risk_reward': risk_reward,
                'liquidation_stats': basic_signal.get('liquidation_stats', {}),
                'volume_analysis': basic_signal.get('volume_analysis', {}),
                'prediction_info': {
                    'type': pred_type,
                    'target_price': target_price,
                    'confidence': pred_confidence
                },
                'timestamp': basic_signal.get('timestamp'),
                'reason': integrated_reason,
                'is_integrated': True
            }
            
        except Exception as e:
            print(f"❌ 청산 통합 신호 생성 오류: {e}")
            return basic_signal
    
    def _convert_prediction_to_liquidation_signal(self, prediction_signal: Dict, current_price: float) -> Dict:
        """예측 신호를 청산 신호 형태로 변환"""
        try:
            pred_type = prediction_signal.get('type', 'UNKNOWN')
            confidence = prediction_signal.get('confidence', 0)
            target_price = prediction_signal.get('target_price', current_price)
            
            # 예측 타입에 따른 액션 결정
            if pred_type == 'EXPLOSION_UP':
                action = 'BUY'
                stop_loss = current_price * 0.98
                take_profit1 = target_price
                take_profit2 = target_price * 1.02
            elif pred_type == 'EXPLOSION_DOWN':
                action = 'SELL'
                stop_loss = current_price * 1.02
                take_profit1 = target_price
                take_profit2 = target_price * 0.98
            else:
                return None
            
            # 리스크/보상 비율 계산
            risk = abs(current_price - stop_loss)
            reward = abs(take_profit1 - current_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            return {
                'signal_type': 'INTEGRATED_LIQUIDATION',
                'action': action,
                'confidence': confidence,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit1': take_profit1,
                'take_profit2': take_profit2,
                'risk_reward': risk_reward,
                'liquidation_stats': {},
                'volume_analysis': {},
                'prediction_info': {
                    'type': pred_type,
                    'target_price': target_price,
                    'confidence': confidence
                },
                'timestamp': prediction_signal.get('timestamp'),
                'reason': f"{pred_type} 예측 기반 {action} 신호 | 목표가: ${target_price:.2f}",
                'is_integrated': False
            }
            
        except Exception as e:
            print(f"❌ 예측 신호 변환 오류: {e}")
            return None
    
    def _run_periodic_analysis(self):
        """주기적 분석 (5분봉 기반)"""
        while self.running:
            try:
                # 5분봉 타이밍 계산
                next_candle = get_next_5min_candle_time()
                now = datetime.datetime.now()
                
                if now >= next_candle:
                    # 1초 후 분석 시작
                    time.sleep(1)
                    
                    print(f"\n⏰ {now.strftime('%H:%M:%S')} - 5분봉 주기적 분석 시작")
                    
                    # 세션 기반 전략과 고급 청산 전략 분석
                    websocket = self.core.get_websocket()
                    
                    session_signal = self._analyze_session_strategy(websocket)
                    advanced_liquidation_signal = self._analyze_advanced_liquidation_strategy(websocket)
                    
                    if session_signal or advanced_liquidation_signal:
                        print(f"\n{'='*50}")
                        print(f"🎯 5분봉 주기 분석 - 전략 신호 생성됨!")
                        print(f"{'='*50}")
                        self._process_integrated_signal({
                            'session_signal': session_signal,
                            'advanced_liquidation_signal': advanced_liquidation_signal
                        })
                    else:
                        # 신호가 없어도 분석 상태 출력 (간단하게)
                        current_price = websocket.price_history[-1]['price'] if websocket.price_history else 0
                        print(f"📊 5분봉 분석 완료 | ${current_price:.2f} | 다음: {(next_candle + datetime.timedelta(minutes=5)).strftime('%H:%M')}")
                    
                    self.last_5min_analysis = now
                    print(f"✅ {now.strftime('%H:%M')} - 5분봉 분석 완료")
                
                    # 다음 5분봉까지 대기 (더 짧은 간격으로 체크)
                    time.sleep(30)  # 30초마다 체크
                else:
                    # 다음 5분봉까지 대기 (더 짧은 간격으로 체크)
                    time.sleep(10)  # 10초마다 체크
                    
            except Exception as e:
                print(f"❌ 주기적 분석 오류: {e}")
                time.sleep(10)
    
    def _process_integrated_signal(self, signal: Dict):
        """개별 전략 신호 처리 - 명확하게 분리"""
        try:
            # 세션 신호와 고급 청산 신호 처리
            session_signal = signal.get('session_signal')
            advanced_liquidation_signal = signal.get('advanced_liquidation_signal')
            now = datetime.datetime.now()
            
            # 세션 신호 처리
            if session_signal:
                self._print_session_signal(session_signal, now)
            
            # 고급 청산 신호 처리 (신호가 있을 때만 출력)
            if advanced_liquidation_signal:
                self._print_advanced_liquidation_signal(advanced_liquidation_signal, now)
            
            # 통합 신호가 있는 경우
            if signal.get('signal_type'):
                self._print_integrated_signal(signal, now)
            
        except Exception as e:
            print(f"❌ 신호 처리 오류: {e}")
    
    def _print_session_signal(self, signal: Dict, now: datetime.datetime):
        """세션 신호 출력 - 명확하게 분리"""
        try:
            action = signal.get('action', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            signal_type = signal.get('signal_type', 'N/A')
            reason = signal.get('reason', 'N/A')
            
            # 중립 신호인지 확인
            is_neutral = action == 'NEUTRAL'
            
            if is_neutral:
                print(f"\n{'='*50}")
                print(f"📊 세션 전략 분석 결과 (중립)")
                print(f"{'='*50}")
                print(f"⏰ 시간: {now.strftime('%H:%M:%S')}")
                print(f"🎯 액션: {action}")
                print(f"📈 신호 타입: {signal_type}")
                print(f"💪 신뢰도: {confidence:.1%}")
                print(f"📝 이유: {reason}")
                print(f"{'='*50}\n")
            else:
                print(f"\n{'='*50}")
                print(f"📊 세션 전략 신호 감지!")
                print(f"{'='*50}")
                print(f"⏰ 시간: {now.strftime('%H:%M:%S')}")
                print(f"🎯 액션: {action}")
                print(f"📈 신호 타입: {signal_type}")
                print(f"💪 신뢰도: {confidence:.1%}")
                print(f"📝 이유: {reason}")
                
                # 추가 정보가 있는 경우 출력
                if 'entry_price' in signal:
                    print(f"💰 진입가: ${signal['entry_price']:.2f}")
                if 'stop_loss' in signal:
                    print(f"🛑 손절가: ${signal['stop_loss']:.2f}")
                if 'take_profit' in signal:
                    print(f"🎯 목표가: ${signal['take_profit']:.2f}")
                
                print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"❌ 세션 신호 출력 오류: {e}")
    
    def _print_advanced_liquidation_signal(self, signal: Dict, now: datetime.datetime):
        """고급 청산 신호 출력 - 명확하게 분리"""
        try:
            # 신호가 None인 경우 기본값 설정
            if signal is None:
                signal = {}
            
            action = signal.get('action', 'NEUTRAL')
            playbook = signal.get('playbook', 'NO_SIGNAL')
            tier = signal.get('tier', 'NEUTRAL')
            total_score = signal.get('total_score', 0.000)
            reason = signal.get('reason', '모든 전략에서 신호 없음')
            
            print(f"\n{'='*50}")
            print(f"⚡ 고급 청산 전략 신호 감지!")
            print(f"{'='*50}")
            print(f"⏰ 시간: {now.strftime('%H:%M:%S')}")
            print(f"🎯 액션: {action}")
            print(f"📚 플레이북: {playbook}")
            print(f"🏆 등급: {tier}")
            print(f"📊 총점: {total_score:.3f}")
            print(f"📝 이유: {reason}")
            
            # 추가 정보가 있는 경우 출력
            if 'entry_price' in signal:
                print(f"💰 진입가: ${signal['entry_price']:.2f}")
            if 'stop_loss' in signal:
                print(f"🛑 손절가: ${signal['stop_loss']:.2f}")
            if 'take_profit' in signal:
                print(f"🎯 목표가: ${signal['take_profit']:.2f}")
            
            print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"❌ 고급 청산 신호 출력 오류: {e}")
    
    def _print_integrated_signal(self, signal: Dict, now: datetime.datetime):
        """통합 신호 출력 - 명확하게 분리"""
        try:
            signal_type = signal.get('signal_type', 'UNKNOWN')
            action = signal.get('action', 'NEUTRAL')
            confidence = signal.get('confidence', 0)
            reason = signal.get('reason', 'N/A')
            
            signal_icon = self._get_signal_icon(signal_type)
            signal_name = self._get_signal_name(signal_type)
            
            print(f"\n{'='*50}")
            print(f"{signal_icon} {signal_name} 신호 감지!")
            print(f"{'='*50}")
            print(f"⏰ 시간: {now.strftime('%H:%M:%S')}")
            print(f"🎯 액션: {action}")
            print(f"💪 신뢰도: {confidence:.1%}")
            print(f"📝 이유: {reason}")
            
            # 추가 정보가 있는 경우 출력
            if 'entry_price' in signal:
                print(f"💰 진입가: ${signal['entry_price']:.2f}")
            if 'stop_loss' in signal:
                print(f"🛑 손절가: ${signal['stop_loss']:.2f}")
            if 'take_profit1' in signal:
                print(f"🎯 목표가1: ${signal['take_profit1']:.2f}")
            if 'take_profit2' in signal:
                print(f"🎯 목표가2: ${signal['take_profit2']:.2f}")
            
            print(f"{'='*50}\n")
            
        except Exception as e:
            print(f"❌ 통합 신호 출력 오류: {e}")
    
    def _get_signal_icon(self, signal_type: str) -> str:
        """신호 타입별 아이콘 반환"""
        icons = {
            'SESSION': '📊',
            'ADVANCED_LIQUIDATION': '⚡',
            'INTEGRATED_LIQUIDATION': '🎯',
            'INTEGRATED': '🎯',
            'UNKNOWN': '❓'
        }
        return icons.get(signal_type, '❓')
    
    def _get_signal_name(self, signal_type: str) -> str:
        """신호 타입별 이름 반환"""
        names = {
            'SESSION': '세션 전략',
            'ADVANCED_LIQUIDATION': '고급 청산 전략',
            'INTEGRATED_LIQUIDATION': '통합 청산 전략',
            'INTEGRATED': '통합 전략',
            'UNKNOWN': 'UNKNOWN'
        }
        return names.get(signal_type, 'UNKNOWN')
    
    def start(self):
        """트레이더 시작"""
        self._print_startup_info()
        
        self.running = True
        
        # 웹소켓 백그라운드 시작
        self.core.start_websocket()
        
        # 주기적 분석 스레드 (옵션)
        if self.config.use_periodic_hybrid:
            self.core.periodic_thread = threading.Thread(target=self._run_periodic_analysis, daemon=True)
            self.core.periodic_thread.start()
        
        # 메인 루프
        self._run_main_loop()
    
    def _print_startup_info(self):
        """시작 정보 출력"""
        print(f"🚀 {self.config.symbol} 통합 스마트 트레이더 시작!")
        print(f"📊 세션: {'활성' if self.config.enable_session_strategy else '비활성'}")
        print(f"⏰ 모드: {'주기(5m)' if self.config.use_periodic_hybrid else '실시간'}")
        print("=" * 60)
        print("💡 실시간 분석 중... 신호가 나올 때만 알림을 표시합니다.")
        print("=" * 60)
    
    def _run_main_loop(self):
        """메인 실행 루프"""
        try:
            last_technical_analysis = None
            api_call_count = 0
            last_api_reset = datetime.datetime.now()
            max_api_calls_per_minute = 2400
            
            while self.running:
                now = datetime.datetime.now()
                
                # API 호출 제한 체크 (1분마다 리셋)
                if (now - last_api_reset).total_seconds() >= 60:
                    api_call_count = 0
                    last_api_reset = now
                
                # 정각 1분마다 세션 전략 분석 (00초)
                if (now.second == 0 and 
                    (not last_technical_analysis or 
                        (now - last_technical_analysis).total_seconds() >= 60)):
                    
                    # API 호출 제한 체크
                    if api_call_count < max_api_calls_per_minute:
                        # 정각 1분마다 세션 전략 분석 실행
                        self._analyze_realtime_technical()
                        last_technical_analysis = now
                        api_call_count += 1
                        # print(f"📊 정각 1분 분석: {now.strftime('%H:%M')}")  # 조용한 모드
                    else:
                        # API 제한 도달 시 5초 대기
                        if not last_technical_analysis or (now - last_technical_analysis).total_seconds() > 5:
                            print(f"⚠️ API 제한 도달, 5초 대기...")
                            self._analyze_realtime_technical()
                            last_technical_analysis = now
                            api_call_count += 1
                
                time.sleep(1)  # 1초마다 체크
                    
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중지됨")
        finally:
            self.stop()
    
    def stop(self):
        """트레이더 중지"""
        self.running = False
        self.core.stop_websocket()
        print("🛑 통합 스마트 자동 트레이더 중지됨")


# ==================== 메인 실행 부분 ====================

def main():
    """메인 함수"""
    try:
        config = IntegratedConfig()
        trader = IntegratedSmartTrader(config)
        trader.start()
    except KeyboardInterrupt:
        print("\n⏹️ 프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 프로그램 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
