#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import requests
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta


class BinanceDataLoader:
    """
    바이낸스에서 3분봉 데이터를 가져오는 클래스
    - Futures API 사용
    - 날짜 범위 지정 가능
    - OHLCV + 추가 정보 제공
    """
    
    def __init__(self, base_url: str = "https://fapi.binance.com"):
        self.base_url = base_url
        self.klines_endpoint = f"{base_url}/fapi/v1/klines"
    
    def fetch_data(self, 
                        interval: int = 3,
                        symbol: str = "ETHUSDT",
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                    ) -> Optional[pd.DataFrame]:
        """
        3분봉 데이터 가져오기
        
        Args:
            symbol: 심볼 (기본값: ETHUSDT)
            start_time: 시작 시간 (UTC)
            end_time: 종료 시간 (UTC)
        
        Returns:
            DataFrame 또는 None (실패 시)
        """
        try:
            # 파라미터 구성
            params = {
                'symbol': symbol.upper(),
                'interval': f'{interval}m',
                'limit': 1500
            }
            
            # 시간 범위 지정
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            print(f"📡 바이낸스 API 요청: {symbol} {interval}분봉 데이터")
            if start_time and end_time:
                print(f"🕐 기간: {start_time.strftime('%Y-%m-%d %H:%M')} ~ {end_time.strftime('%Y-%m-%d %H:%M')} UTC")
            
            # API 요청
            response = requests.get(self.klines_endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                print("⚠️ 데이터가 비어있습니다")
                return None
            
            # DataFrame 생성
            print(f"📊 API 응답 데이터: {len(data)}개")
            df = self._parse_klines_data(data)
            
            if df.empty:
                print("⚠️ 파싱된 데이터가 비어있습니다")
                return None
    
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 네트워크 오류: {e}")
            return None
        except Exception as e:
            print(f"❌ 데이터 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def fetch_prev_day_3m(self, symbol: str = "ETHUSDT") -> Optional[pd.DataFrame]:
        """
        어제 하루의 3분봉 데이터 가져오기
        
        Args:
            symbol: 심볼 (기본값: ETHUSDT)
        
        Returns:
            DataFrame 또는 None (실패 시)
        """
        # UTC 기준 어제 날짜 계산
        utc_now = datetime.now(timezone.utc)
        prev_day = utc_now - timedelta(days=1)
        start_time = prev_day.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = prev_day.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        print(f"📅 어제 데이터 요청: {start_time.strftime('%Y-%m-%d')} UTC")
        
        return self.fetch_data(symbol, start_time, end_time)
    
    def fetch_recent_3m(self, symbol: str = "ETHUSDT", hours: int = 24) -> Optional[pd.DataFrame]:
        """
        최근 N시간의 3분봉 데이터 가져오기
        
        Args:
            symbol: 심볼 (기본값: ETHUSDT)
            hours: 최근 몇 시간 (기본값: 24시간)
        
        Returns:
            DataFrame 또는 None (실패 시)
        """
        utc_now = datetime.now(timezone.utc)
        start_time = utc_now - timedelta(hours=hours)
        
        # 3분봉 개수 계산 (1시간 = 20개)
        candle_count = hours * 20
        
        print(f"⏰ 최근 {hours}시간 데이터 요청")
        
        return self.fetch_data(interval=3, symbol=symbol, start_time=start_time, end_time=utc_now)
    
    def _parse_klines_data(self, data: List) -> pd.DataFrame:
        """바이낸스 Kline 데이터를 DataFrame으로 파싱"""
        try:
            if not data:
                return pd.DataFrame()

            # 표준 컬럼명으로 DataFrame 생성
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # 숫자 컬럼을 float로 변환
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 시간 컬럼을 datetime으로 변환 (밀리초 timestamp 처리)
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
            df['close_time'] = pd.to_datetime(df['close_time']+1, unit='ms', utc=True)
            
            # close_time을 인덱스로 설정 (3분봉 완료 시점)
            df.set_index('close_time', inplace=True)
            df.index.name = 'timestamp'  
            # 필요한 컬럼만 선택 (표준 OHLCV 구조)
            df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume']]
            
            # 3분봉 데이터 검증 및 필터링
            df = df.sort_index()
            
            # 현재 시간보다 미래의 close_time 제거
            current_time = datetime.now(timezone.utc)
            future_candles = df[df.index > current_time]

            if not future_candles.empty:
                print(f"⚠️ 미래 시간 캔들 {len(future_candles)}개 제거: {future_candles.index[0]} ~ {future_candles.index[-1]}")
                df = df[df.index <= current_time]

            return df
            
        except Exception as e:
            print(f"❌ Kline 데이터 파싱 오류: {e}")
            return pd.DataFrame()
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        데이터 요약 정보 반환
        
        Args:
            df: 3분봉 DataFrame
        
        Returns:
            데이터 요약 정보
        """
        if df is None or df.empty:
            return {}
        
        return {
            'symbol': 'ETHUSDT',  # 현재는 고정값
            'interval': '3m',
            'count': len(df),
            'start_time': df.index[0],
            'end_time': df.index[-1],
            'duration_hours': (df.index[-1] - df.index[0]).total_seconds() / 3600,
            'price_info': {
                'high': float(df['high'].max()),
                'low': float(df['low'].min()),
                'open': float(df['open'].iloc[0]),
                'close': float(df['close'].iloc[-1]),
                'change_pct': ((df['close'].iloc[-1] / df['open'].iloc[0]) - 1) * 100
            },
            'volume_info': {
                'total_volume': float(df['volume'].sum()),
                'total_quote_volume': float(df['quote_volume'].sum()),
                'avg_volume': float(df['volume'].mean()),
                'max_volume': float(df['volume'].max()),
                'total_trades': int(df['trades'].sum())
            },
            'trade_info': {
                'avg_buy_ratio': float(df['buy_ratio'].mean()),
                'avg_sell_ratio': float(df['sell_ratio'].mean()),
                'avg_trade_size': float(df['avg_trade_size'].mean()),
                'avg_vwap': float(df['vwap'].mean())
            }
        }


def test_dataloader():
    """데이터로더 테스트 함수"""
    print("🚀 BinanceDataLoader 테스트 시작...")
    
    loader = BinanceDataLoader()
    
    # 어제 데이터 테스트
    print("\n📊 어제 3분봉 데이터 테스트:")
    prev_day_df = loader.fetch_prev_day_3m('ETHUSDT')
    
    if prev_day_df is not None:
        info = loader.get_data_info(prev_day_df)
        print(f"✅ 어제 데이터: {info['count']}개 캔들")
        print(f"📈 가격 정보: ${info['price_info']['low']:.2f} ~ ${info['price_info']['high']:.2f}")
        print(f"💰 총 거래량: {info['volume_info']['total_volume']:.2f} ETH")
        print(f"📊 총 거래 횟수: {info['volume_info']['total_trades']:,}회")
    else:
        print("❌ 어제 데이터 로드 실패")
    
    # 최근 6시간 데이터 테스트
    print("\n⏰ 최근 6시간 데이터 테스트:")
    recent_df = loader.fetch_recent_3m('ETHUSDT', hours=6)
    
    if recent_df is not None:
        info = loader.get_data_info(recent_df)
        print(f"✅ 최근 데이터: {info['count']}개 캔들")
        print(f"📈 가격 변화: {info['price_info']['change_pct']:.2f}%")
        print(f"💰 평균 거래량: {info['volume_info']['avg_volume']:.2f} ETH")
    else:
        print("❌ 최근 데이터 로드 실패")
    
    print("\n🏁 테스트 완료!")


if __name__ == "__main__":
    test_dataloader()
