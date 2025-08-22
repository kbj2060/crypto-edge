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
    
    def fetch_3m_data(self, 
                     symbol: str = "ETHUSDT",
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None,
                     limit: int = 500) -> Optional[pd.DataFrame]:
        """
        3분봉 데이터 가져오기
        
        Args:
            symbol: 심볼 (기본값: ETHUSDT)
            start_time: 시작 시간 (UTC)
            end_time: 종료 시간 (UTC)
            limit: 최대 개수 (기본값: 500, 최대 1500)
        
        Returns:
            DataFrame 또는 None (실패 시)
        """
        try:
            # 파라미터 구성
            params = {
                'symbol': symbol.upper(),
                'interval': '3m',
                'limit': min(limit, 1500)  # 바이낸스 API 제한
            }
            
            # 시간 범위 지정
            if start_time:
                params['startTime'] = int(start_time.timestamp() * 1000)
            if end_time:
                params['endTime'] = int(end_time.timestamp() * 1000)
            
            print(f"📡 바이낸스 API 요청: {symbol} 3분봉 데이터")
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
            df = self._parse_klines_data(data)
            
            print(f"✅ 데이터 로드 성공: {len(df)}개 캔들")
            print(f"📊 기간: {df.index[0]} ~ {df.index[-1]}")
            print(f"💰 평균 거래량: {df['volume'].mean():.2f} ETH")
            
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
        
        return self.fetch_3m_data(symbol, start_time, end_time, limit=500)
    
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
        limit = min(candle_count, 1500)
        
        print(f"⏰ 최근 {hours}시간 데이터 요청")
        
        return self.fetch_3m_data(symbol, start_time, utc_now, limit=limit)
    
    def _parse_klines_data(self, data: List) -> pd.DataFrame:
        """
        바이낸스 klines 데이터를 DataFrame으로 변환
        
        바이낸스 API 응답 형식:
        [0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume,
         6: close_time, 7: quote_volume, 8: trades, 9: taker_buy_base, 10: taker_buy_quote, 11: ignore]
        """
        df_data = []
        
        for candle in data:
            candle_info = {
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5]),  # base asset volume (ETH)
                'quote_volume': float(candle[7]),  # USDT volume
                'trades': int(candle[8]),  # number of trades
                'taker_buy_base': float(candle[9]),  # taker buy base volume
                'taker_buy_quote': float(candle[10]),  # taker buy quote volume
                # 추가 계산 필드
                'avg_price': (float(candle[2]) + float(candle[3]) + float(candle[4])) / 3,  # HLC 평균
                'price_range': float(candle[2]) - float(candle[3]),  # 고가-저가
                'body_size': abs(float(candle[4]) - float(candle[1])),  # 몸통 크기
                'upper_wick': float(candle[2]) - max(float(candle[1]), float(candle[4])),  # 위꼬리
                'lower_wick': min(float(candle[1]), float(candle[4])) - float(candle[3])   # 아래꼬리
            }
            
            # 거래량 관련 계산
            if candle_info['volume'] > 0:
                candle_info['vwap'] = candle_info['quote_volume'] / candle_info['volume']  # 거래량가중평균가
                candle_info['avg_trade_size'] = candle_info['volume'] / candle_info['trades']  # 평균 거래 크기
            else:
                candle_info['vwap'] = candle_info['close']
                candle_info['avg_trade_size'] = 0
            
            # 매수/매도 비율 계산
            if candle_info['volume'] > 0:
                candle_info['buy_ratio'] = candle_info['taker_buy_base'] / candle_info['volume']
                candle_info['sell_ratio'] = 1 - candle_info['buy_ratio']
            else:
                candle_info['buy_ratio'] = 0.5
                candle_info['sell_ratio'] = 0.5
            
            df_data.append(candle_info)
        
        # DataFrame 생성 (close_time을 인덱스로 사용)
        df = pd.DataFrame(df_data)
        
        # close_time을 인덱스로 설정
        close_times = [datetime.fromtimestamp(candle[6] / 1000, tz=timezone.utc) for candle in data]
        df.index = pd.DatetimeIndex(close_times, name='close_time')
        
        return df
    
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
