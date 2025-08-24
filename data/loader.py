# # Data Loader Module
# # 데이터 로더 모듈

# import pandas as pd
# import numpy as np
# from typing import Literal
# from data.binance_client import fetch_klines
# from indicators.moving_averages import add_mas
# from indicators.bollinger import add_bollinger
# from indicators.macd import add_macd
# from indicators.stoch_rsi import add_stoch_rsi

# def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
#     """
#     Average True Range (ATR) 계산
    
#     Args:
#         df: OHLCV 데이터가 포함된 DataFrame
#         period: ATR 계산 기간
        
#     Returns:
#         pd.Series: ATR 값
#     """
#     high_low = df['high'] - df['low']
#     high_close = np.abs(df['high'] - df['close'].shift())
#     low_close = np.abs(df['low'] - df['close'].shift())
    
#     true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
#     atr = true_range.rolling(window=period).mean()
    
#     return atr

# def build_df(
#     symbol: str,
#     interval: str = "5m",
#     limit: int = 1500,
#     atr_len: int = 14,
#     market: Literal["spot", "futures"] = "futures",
#     price_source: Literal["last", "mark"] = "last",
#     ma_type: Literal["sma", "ema"] = "ema"
# ) -> pd.DataFrame:
#     """
#     완전한 기술적 분석 데이터프레임 구축
    
#     Args:
#         symbol: 거래 심볼 (예: BTCUSDT)
#         interval: 시간 간격 (예: 1m, 5m, 15m, 1h)
#         limit: 조회할 캔들 개수
#         atr_len: ATR 계산 기간
#         market: 거래 시장 (spot 또는 futures)
#         price_source: 가격 소스 (last 또는 mark)
#         ma_type: 이동평균 타입 (sma 또는 ema)
        
#     Returns:
#         pd.DataFrame: 모든 기술적 지표가 포함된 DataFrame
#     """
#     # 바이낸스에서 데이터 조회
#     df = fetch_klines(symbol, interval, limit, market, price_source)
    
#     if df.empty:
#         print(f"경고: {symbol} {interval} 데이터를 가져올 수 없습니다.")
#         return pd.DataFrame()
    
#     # 기본 기술적 지표 추가
#     df = add_mas(df, cfg_ma=(ma_type, 50), cfg_ma2=(ma_type, 200))
#     df = add_bollinger(df, length=20, mult=2.0)
#     df = add_macd(df, fast=12, slow=26, signal=9)
#     df = add_stoch_rsi(df, rsi_len=14, stoch_len=14, k=3, d=3)
    
#     # ATR 추가
#     df[f'ATR_{atr_len}'] = calculate_atr(df, atr_len)
    
#     # 추가 지표들
#     df['SMA_200'] = df['close'].rolling(200).mean()
    
#     # 거래량 관련 지표
#     df['volume_ma'] = df['volume'].rolling(20).mean()
#     df['volume_ratio'] = df['volume'] / df['volume_ma']
    
#     # VWAP (Volume Weighted Average Price)
#     df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    
#     # 가격 변화율
#     df['price_change'] = df['close'].pct_change()
#     df['price_change_5'] = df['close'].pct_change(5)
    
#     # 변동성 지표
#     df['volatility'] = df['close'].rolling(20).std()
#     df['volatility_ratio'] = df['volatility'] / df['close']
    
#     return df
