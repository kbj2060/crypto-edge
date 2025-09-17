# htf_trend_bot.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

@dataclass
class HTFConfig:
    ema_period: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    lookback_15m: int = 20
    atr_period: int = 14
    min_data_length: int = 300
    # 새로운 설정
    ltf_ema_period: int = 20  # 15분봉 트렌드용 EMA
    macd_threshold: float = 0.5  # MACD 강도 정규화 기준
    min_score: float = 0.2
    max_score: float = 0.9

class HTFTrend:
    """Higher Timeframe Trend Bot - 시간프레임 일치도 반영"""
    
    def __init__(self, config: HTFConfig = HTFConfig()):
        self.config = config
    
    @staticmethod
    def ensure_index(df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 인덱스를 DatetimeIndex로 변환"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
    
    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        """지수이동평균 계산"""
        return series.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def macd(series: pd.Series, fast=12, slow=26, signal=9):
        """MACD 지표 계산"""
        f = HTFTrend.ema(series, fast)
        s = HTFTrend.ema(series, slow)
        macd_line = f - s
        sig = HTFTrend.ema(macd_line, signal)
        hist = macd_line - sig
        return macd_line, sig, hist
    
    @staticmethod
    def atr(df: pd.DataFrame, period=14) -> pd.Series:
        """Average True Range 계산"""
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period, min_periods=1).mean()
    
    def get_trend_direction(self, close_series: pd.Series, ema_period: int) -> str:
        """트렌드 방향 판단"""
        if len(close_series) < ema_period + 3:
            return 'FLAT'
        
        ema = self.ema(close_series, ema_period)
        if len(ema) < 3:
            return 'FLAT'
        
        # 최근 3개 EMA 값의 기울기로 트렌드 판단
        slope = ema.iloc[-1] - ema.iloc[-3]
        current_price = close_series.iloc[-1]
        current_ema = ema.iloc[-1]
        
        # 가격이 EMA 위/아래에 있는지도 고려
        price_above_ema = current_price > current_ema
        
        if slope > 0 and price_above_ema:
            return 'UP'
        elif slope < 0 and not price_above_ema:
            return 'DOWN'
        else:
            return 'FLAT'
    
    def calculate_alignment_multiplier(self, htf_trend: str, ltf_trend: str) -> float:
        """시간프레임 간 방향 일치도 계산"""
        if htf_trend == ltf_trend and htf_trend != 'FLAT':
            # 강한 일치: 같은 방향
            return 1.3
        elif htf_trend != ltf_trend and 'FLAT' not in [htf_trend, ltf_trend]:
            # 강한 충돌: 반대 방향
            return 0.6
        elif 'FLAT' in [htf_trend, ltf_trend]:
            # 중간: 한쪽이 FLAT
            return 0.8
        else:
            # 기본값
            return 1.0
    
    def calculate_macd_strength(self, macd_hist: float) -> float:
        """MACD 히스토그램 강도 계산 (0-1 정규화)"""
        return min(0.8, abs(macd_hist) / self.config.macd_threshold)

    def on_kline_close_15m(
        self,
        df_15m: pd.DataFrame,
        df_1h: Optional[pd.DataFrame] = None,
        df_4h: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        트렌드 신호 생성 - 시간프레임 일치도 반영
        
        Returns:
            {'name','action'('LONG'/'SHORT'/'HOLD'),'score'(0..1),'entry','stop','context'}
        """
        df_15m = self.ensure_index(df_15m)
        if len(df_15m) < max(30, self.config.lookback_15m + 2):
            return {
                'name': 'HTF_TREND', 
                'action': 'HOLD', 
                'score': 0.0,
                'context': {'reason': 'insufficient_15m', 'n': len(df_15m)}
            }

        # 1. HTF 트렌드 분석 (1H 또는 4H)
        htf_df = df_1h if df_1h is not None else df_4h
        htf_trend = 'FLAT'
        htf_macd_hist = 0.0
        
        if htf_df is not None and len(htf_df) >= self.config.ema_period + 5:
            htf = self.ensure_index(htf_df)
            close_htf = htf['close'].astype(float)
            htf_trend = self.get_trend_direction(close_htf, self.config.ema_period)
            _, _, macd_hist_series = self.macd(close_htf, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal)
            htf_macd_hist = float(macd_hist_series.iloc[-1])
        else:
            # HTF 데이터가 없으면 15분봉으로 근사
            close_15 = df_15m['close'].astype(float)
            htf_trend = self.get_trend_direction(close_15, self.config.ema_period)
            _, _, macd_hist_series = self.macd(close_15, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal)
            htf_macd_hist = float(macd_hist_series.iloc[-1])

        # 2. LTF (15분) 트렌드 분석
        close_15m = df_15m['close'].astype(float)
        ltf_trend = self.get_trend_direction(close_15m, self.config.ltf_ema_period)
        
        # 15분봉 MACD
        _, _, ltf_macd_hist_series = self.macd(close_15m, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal)
        ltf_macd_hist = float(ltf_macd_hist_series.iloc[-1])

        # 3. 시간프레임 일치도 계산
        alignment_multiplier = self.calculate_alignment_multiplier(htf_trend, ltf_trend)
        
        # 4. MACD 강도 계산 (HTF와 LTF 평균)
        avg_macd_hist = (htf_macd_hist + ltf_macd_hist) / 2
        macd_strength = self.calculate_macd_strength(avg_macd_hist)
        
        # 5. 진입/청산 로직
        recent = df_15m.iloc[-(self.config.lookback_15m + 1):].copy()
        last_close = float(recent['close'].iloc[-1])
        ema15 = self.ema(recent['close'].astype(float), self.config.ltf_ema_period)
        last_ema15 = float(ema15.iloc[-1])
        atr_val = float(self.atr(df_15m, self.config.atr_period).iloc[-1])

        action = 'HOLD'
        entry = None
        stop = None
        base_score = 0.0

        # 매수 조건: HTF 상승 + LTF 상승 + 가격이 15분 EMA 위
        if htf_trend == 'UP' and last_close >= last_ema15:
            action = 'BUY'
            # 기본 점수 계산
            base_score = macd_strength
            entry = last_close
            stop = entry - 1.5 * atr_val
            
        # 매도 조건: HTF 하락 + LTF 하락 + 가격이 15분 EMA 아래
        elif htf_trend == 'DOWN' and last_close <= last_ema15:
            action = 'SELL'
            # 기본 점수 계산
            base_score = macd_strength
            entry = last_close
            stop = entry + 1.5 * atr_val

        # 6. 최종 점수 계산 (일치도 반영)
        final_score = base_score * alignment_multiplier
        final_score = max(self.config.min_score, min(self.config.max_score, final_score))

        return {
            'name': 'HTF_TREND',
            'action': action,
            'score': float(final_score),
            'entry': float(entry) if entry is not None else None,
            'stop': float(stop) if stop is not None else None,
            'context': {
                'htf_trend': htf_trend,
                'ltf_trend': ltf_trend,
                'alignment_multiplier': alignment_multiplier,
                'htf_macd_hist': htf_macd_hist,
                'ltf_macd_hist': ltf_macd_hist,
                'macd_strength': macd_strength,
                'base_score': base_score,
                'atr': atr_val
            }
        }