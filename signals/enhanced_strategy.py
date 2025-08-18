# Enhanced Trading Strategy
# 향상된 거래 전략 모듈

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from utils.config import RunConfig

@dataclass
class EnhancedConfig:
    """향상된 전략 설정"""
    # 기본 설정
    min_confirms_long: int = 3
    min_confirms_short: int = 3
    
    # 가중치 설정 (0.0 ~ 1.0)
    weight_trend: float = 0.25      # 추세 가중치
    weight_momentum: float = 0.25   # 모멘텀 가중치
    weight_volatility: float = 0.20 # 변동성 가중치
    weight_volume: float = 0.15     # 거래량 가중치
    weight_support_resistance: float = 0.15  # 지지/저항 가중치
    
    # 신호 강화 설정
    require_trend_alignment: bool = True  # 추세 정렬 필요
    require_volume_confirmation: bool = True  # 거래량 확인 필요
    use_divergence: bool = True  # 다이버전스 사용
    use_breakout_confirmation: bool = True  # 돌파 확인 사용
    
    # 리스크 관리
    max_risk_per_trade: float = 0.02  # 거래당 최대 리스크 (2%)
    min_risk_reward_ratio: float = 1.5  # 최소 리스크/보상 비율
    max_correlation_threshold: float = 0.8  # 최대 상관관계 임계값

def calculate_enhanced_signals(df: pd.DataFrame, cfg: EnhancedConfig) -> pd.DataFrame:
    """
    향상된 신호 계산
    """
    df = df.copy()
    
    # 1. 추세 신호 (가중치: 25%)
    df = _calculate_trend_signals(df)
    
    # 2. 모멘텀 신호 (가중치: 25%)
    df = _calculate_momentum_signals(df)
    
    # 3. 변동성 신호 (가중치: 20%)
    df = _calculate_volatility_signals(df)
    
    # 4. 거래량 신호 (가중치: 15%)
    df = _calculate_volume_signals(df)
    
    # 5. 지지/저항 신호 (가중치: 15%)
    df = _calculate_support_resistance_signals(df)
    
    # 6. 다이버전스 감지
    if cfg.use_divergence:
        df = _detect_divergences(df)
    
    # 7. 돌파 확인
    if cfg.use_breakout_confirmation:
        df = _confirm_breakouts(df)
    
    # 8. 복합 신호 생성
    df = _generate_composite_signal(df, cfg)
    
    return df

def _calculate_trend_signals(df: pd.DataFrame) -> pd.DataFrame:
    """추세 신호 계산"""
    # EMA 정렬 강도
    df['ema_alignment_strength'] = (df['EMA_50'] - df['EMA_200']) / df['EMA_200']
    
    # 추세 방향 (1: 상승, -1: 하락, 0: 중립)
    df['trend_direction'] = 0
    df.loc[df['ema_alignment_strength'] > 0.01, 'trend_direction'] = 1
    df.loc[df['ema_alignment_strength'] < -0.01, 'trend_direction'] = -1
    
    # 추세 강도 (0~1)
    df['trend_strength'] = abs(df['ema_alignment_strength'])
    
    # 가격과 EMA50의 관계
    df['price_vs_ema50'] = (df['close'] - df['EMA_50']) / df['EMA_50']
    
    return df

def _calculate_momentum_signals(df: pd.DataFrame) -> pd.DataFrame:
    """모멘텀 신호 계산"""
    # MACD 모멘텀
    df['macd_momentum'] = df['MACD'] - df['MACD_signal']
    df['macd_momentum_strength'] = abs(df['macd_momentum'])
    
    # StochRSI 모멘텀
    df['stoch_momentum'] = df['StochRSI_K'] - df['StochRSI_D']
    df['stoch_momentum_strength'] = abs(df['stoch_momentum'])
    
    # RSI 모멘텀 (RSI 계산)
    df['rsi'] = _calculate_rsi(df['close'], 14)
    df['rsi_momentum'] = df['rsi'] - 50  # 중립점 대비
    
    # 모멘텀 종합 점수
    df['momentum_score'] = (
        df['macd_momentum_strength'] * 0.4 +
        df['stoch_momentum_strength'] * 0.3 +
        abs(df['rsi_momentum']) * 0.3
    )
    
    return df

def _calculate_volatility_signals(df: pd.DataFrame) -> pd.DataFrame:
    """변동성 신호 계산"""
    # 볼린저 밴드 위치
    df['bb_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # 볼린저 밴드 압축/확장
    df['bb_squeeze'] = df['BB_width'] < df['BB_width'].rolling(20).mean()
    df['bb_expansion'] = df['BB_width'] > df['BB_width'].rolling(20).mean()
    
    # ATR 기반 변동성
    df['atr_ratio'] = df['ATR_14'] / df['close']
    df['volatility_regime'] = 'normal'
    df.loc[df['atr_ratio'] > df['atr_ratio'].rolling(20).quantile(0.8), 'volatility_regime'] = 'high'
    df.loc[df['atr_ratio'] < df['atr_ratio'].rolling(20).quantile(0.2), 'volatility_regime'] = 'low'
    
    return df

def _calculate_volume_signals(df: pd.DataFrame) -> pd.DataFrame:
    """거래량 신호 계산"""
    # 거래량 이동평균
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # 거래량 가중 가격 변화
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    
    # 거래량 모멘텀
    df['volume_momentum'] = df['volume'].pct_change()
    
    # 거래량 신호 강도
    df['volume_signal_strength'] = np.where(
        df['volume_ratio'] > 1.5, 1.0,
        np.where(df['volume_ratio'] > 1.2, 0.7,
        np.where(df['volume_ratio'] > 1.0, 0.5, 0.2))
    )
    
    return df

def _calculate_support_resistance_signals(df: pd.DataFrame) -> pd.DataFrame:
    """지지/저항 신호 계산"""
    # 스윙 고점/저점
    df['swing_high'] = df['high'].rolling(10, center=True).max()
    df['swing_low'] = df['low'].rolling(10, center=True).min()
    
    # 지지/저항 레벨 근접도
    df['resistance_proximity'] = (df['swing_high'] - df['close']) / df['close']
    df['support_proximity'] = (df['close'] - df['swing_low']) / df['close']
    
    # 지지/저항 강도
    df['support_strength'] = np.where(
        df['support_proximity'] < 0.01, 1.0,
        np.where(df['support_proximity'] < 0.02, 0.7,
        np.where(df['support_proximity'] < 0.05, 0.3, 0.0))
    )
    
    df['resistance_strength'] = np.where(
        df['resistance_proximity'] < 0.01, 1.0,
        np.where(df['resistance_proximity'] < 0.02, 0.7,
        np.where(df['resistance_proximity'] < 0.05, 0.3, 0.0))
    )
    
    return df

def _detect_divergences(df: pd.DataFrame) -> pd.DataFrame:
    """다이버전스 감지"""
    # 가격과 RSI 다이버전스
    df['price_rsi_divergence'] = 0
    
    # 가격 상승, RSI 하락 (베어리시 다이버전스)
    price_up = df['close'] > df['close'].shift(5)
    rsi_down = df['rsi'] < df['rsi'].shift(5)
    df.loc[price_up & rsi_down & (df['rsi'] > 70), 'price_rsi_divergence'] = -1
    
    # 가격 하락, RSI 상승 (불리시 다이버전스)
    price_down = df['close'] < df['close'].shift(5)
    rsi_up = df['rsi'] > df['rsi'].shift(5)
    df.loc[price_down & rsi_up & (df['rsi'] < 30), 'price_rsi_divergence'] = 1
    
    # 가격과 MACD 다이버전스
    df['price_macd_divergence'] = 0
    
    # 가격 상승, MACD 하락
    macd_down = df['MACD'] < df['MACD'].shift(5)
    df.loc[price_up & macd_down & (df['MACD'] < 0), 'price_macd_divergence'] = -1
    
    # 가격 하락, MACD 상승
    macd_up = df['MACD'] > df['MACD'].shift(5)
    df.loc[price_down & macd_up & (df['MACD'] > 0), 'price_macd_divergence'] = 1
    
    return df

def _confirm_breakouts(df: pd.DataFrame) -> pd.DataFrame:
    """돌파 확인"""
    # 볼린저 밴드 돌파
    df['bb_breakout'] = 0
    df.loc[df['close'] > df['BB_upper'], 'bb_breakout'] = 1
    df.loc[df['close'] < df['BB_lower'], 'bb_breakout'] = -1
    
    # EMA 돌파
    df['ema_breakout'] = 0
    df.loc[df['close'] > df['EMA_50'], 'ema_breakout'] = 1
    df.loc[df['close'] < df['EMA_50'], 'ema_breakout'] = -1
    
    # 돌파 강도 (거래량 확인)
    df['breakout_strength'] = df['bb_breakout'] * df['volume_signal_strength']
    
    return df

def _generate_composite_signal(df: pd.DataFrame, cfg: EnhancedConfig) -> pd.DataFrame:
    """복합 신호 생성"""
    # 각 신호의 가중 점수 계산
    df['trend_score'] = df['trend_direction'] * df['trend_strength'] * cfg.weight_trend
    df['momentum_score_weighted'] = np.sign(df['macd_momentum']) * df['momentum_score'] * cfg.weight_momentum
    df['volatility_score'] = np.where(df['bb_expansion'], 0.5, -0.5) * cfg.weight_volatility
    df['volume_score'] = df['volume_signal_strength'] * cfg.weight_volume
    df['sr_score'] = (df['support_strength'] - df['resistance_strength']) * cfg.weight_support_resistance
    
    # 다이버전스 보정
    divergence_correction = (df['price_rsi_divergence'] + df['price_macd_divergence']) * 0.1
    df['divergence_correction'] = divergence_correction
    
    # 돌파 보정
    breakout_correction = df['breakout_strength'] * 0.15
    df['breakout_correction'] = breakout_correction
    
    # 최종 복합 점수
    df['composite_score'] = (
        df['trend_score'] +
        df['momentum_score_weighted'] +
        df['volatility_score'] +
        df['volume_score'] +
        df['sr_score'] +
        df['divergence_correction'] +
        df['breakout_correction']
    )
    
    # 신호 생성
    df['enhanced_signal'] = 0
    df.loc[df['composite_score'] > 0.3, 'enhanced_signal'] = 1
    df.loc[df['composite_score'] < -0.3, 'enhanced_signal'] = -1
    
    # 신호 강도
    df['signal_strength'] = abs(df['composite_score'])
    
    return df

def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def make_enhanced_trade_plan(df: pd.DataFrame, vpvr_levels, cfg: EnhancedConfig) -> Dict[str, Any]:
    """
    향상된 거래 계획 생성
    """
    if len(df) < 20:
        return {"bias": "NEUTRAL", "reasons": ["데이터 부족"], "confidence": 0.0}
    
    # 향상된 신호 계산
    df_enhanced = calculate_enhanced_signals(df, cfg)
    
    # 최신 데이터
    latest = df_enhanced.iloc[-1]
    price = float(latest['close'])
    
    # 신호 분석
    signal = latest['enhanced_signal']
    strength = latest['signal_strength']
    
    # VPVR 레벨
    up, dn = _nearest_vpvr_barrier(price, vpvr_levels)
    
    # 리스크/보상 비율 계산
    atr = float(latest.get('ATR_14', price * 0.01))
    
    plan = {
        "bias": "NEUTRAL",
        "confidence": 0.0,
        "entry": None,
        "stop": None,
        "tp1": None,
        "tp2": None,
        "risk_reward_ratio": 0.0,
        "reasons": [],
        "vpvr_up": up,
        "vpvr_dn": dn
    }
    
    if signal == 1:  # LONG
        plan["bias"] = "LONG"
        plan["confidence"] = min(strength, 1.0)
        plan["entry"] = price
        
        # 손절 계산
        stop_candidates = [latest.get('EMA_50'), dn, price - 2 * atr]
        stop_candidates = [s for s in stop_candidates if s is not None and s < price]
        stop = max(stop_candidates) if stop_candidates else price * 0.98
        plan["stop"] = float(stop)
        
        # 익절 계산
        tp1 = min(latest.get('BB_upper', price * 1.02), price + 1.5 * atr)
        tp2 = up if up and up > price else price + 3 * atr
        plan["tp1"] = float(tp1)
        plan["tp2"] = float(tp2)
        
        # 리스크/보상 비율
        risk = price - stop
        reward = tp1 - price
        plan["risk_reward_ratio"] = reward / risk if risk > 0 else 0
        
        # 이유
        reasons = []
        if latest['trend_direction'] == 1:
            reasons.append("강한 상승 추세")
        if latest['momentum_score'] > 0.5:
            reasons.append("강한 모멘텀")
        if latest['volume_signal_strength'] > 0.7:
            reasons.append("거래량 확인")
        if latest['price_rsi_divergence'] == 1:
            reasons.append("RSI 불리시 다이버전스")
        plan["reasons"] = reasons
        
    elif signal == -1:  # SHORT
        plan["bias"] = "SHORT"
        plan["confidence"] = min(strength, 1.0)
        plan["entry"] = price
        
        # 손절 계산
        stop_candidates = [latest.get('EMA_50'), up, price + 2 * atr]
        stop_candidates = [s for s in stop_candidates if s is not None and s > price]
        stop = min(stop_candidates) if stop_candidates else price * 1.02
        plan["stop"] = float(stop)
        
        # 익절 계산
        tp1 = max(latest.get('BB_lower', price * 0.98), price - 1.5 * atr)
        tp2 = dn if dn and dn < price else price - 3 * atr
        plan["tp1"] = float(tp1)
        plan["tp2"] = float(tp2)
        
        # 리스크/보상 비율
        risk = stop - price
        reward = price - tp1
        plan["risk_reward_ratio"] = reward / risk if risk > 0 else 0
        
        # 이유
        reasons = []
        if latest['trend_direction'] == -1:
            reasons.append("강한 하락 추세")
        if latest['momentum_score'] > 0.5:
            reasons.append("강한 모멘텀")
        if latest['volume_signal_strength'] > 0.7:
            reasons.append("거래량 확인")
        if latest['price_rsi_divergence'] == -1:
            reasons.append("RSI 베어리시 다이버전스")
        plan["reasons"] = reasons
    
    return plan

def _nearest_vpvr_barrier(price: float, vpvr_levels) -> Tuple[Optional[float], Optional[float]]:
    """가장 가까운 VPVR 장벽 찾기"""
    if not vpvr_levels:
        return (None, None)
    uppers = sorted([lvl for (lvl, _) in vpvr_levels if lvl >= price])
    lowers = sorted([lvl for (lvl, _) in vpvr_levels if lvl <= price], reverse=True)
    up = uppers[0] if uppers else None
    dn = lowers[0] if lowers else None
    return up, dn
