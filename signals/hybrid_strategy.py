import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Literal
from dataclasses import dataclass
from utils.config import RunConfig
from indicators.vpvr import vpvr_key_levels
from datetime import datetime

@dataclass
class HybridConfig:
    """하이브리드 전략 설정"""
    # 기본 설정
    symbol: str = "ETHUSDT"
    interval_15m: str = "15m"
    interval_5m: str = "5m"
    limit_15m: int = 200
    limit_5m: int = 300
    
    # 15분봉 트렌드 가중치
    trend_weight: float = 0.4  # 0.6 → 0.5로 조정
    momentum_weight: float = 0.5  # 0.4 → 0.5로 조정
    
    # 5분봉 진입 가중치
    entry_weight: float = 0.6  # 0.7 → 0.6으로 조정
    confirmation_weight: float = 0.4  # 0.3 → 0.4로 조정
    
    # 신호 임계값 (더 여유롭게 조정)
    min_trend_strength: float = 0.4  # 0.6 → 0.4로 낮춤
    min_entry_strength: float = 0.3  # 0.5 → 0.3으로 낮춤
    min_hybrid_confidence: float = 0.20
    min_vpvr_headroom: float = 0.001  # 0.002 → 0.001로 낮춤
    
    # 리스크 관리
    atr_len: int = 14
    atr_stop_mult: float = 1.0
    atr_tp1_mult: float = 1.5
    atr_tp2_mult: float = 2.5
    
    # VPVR 설정
    vpvr_bins: int = 50
    vpvr_lookback: int = 200

def analyze_15m_trend(df: pd.DataFrame, vpvr_levels: List[Dict], config: HybridConfig) -> Dict[str, Any]:
    """15분봉 트렌드 분석 (더 엄격한 조건)"""
    if len(df) < 50:
        return {"trend": "NEUTRAL", "strength": 0.0, "reason": "데이터 부족"}
    
    current_price = df['close'].iloc[-1]
    ema_50 = df['EMA_50'].iloc[-1]  # ema_50 → EMA_50
    ema_200 = df['EMA_200'].iloc[-1]  # ema_200 → EMA_200
    
    # 1. EMA 트렌드 (더 엄격한 조건)
    ema_trend = 0.0
    if current_price > ema_50 > ema_200:
        ema_trend = 0.8  # 강한 상승
    elif current_price < ema_50 < ema_200:
        ema_trend = 0.8  # 강한 하락
    elif current_price > ema_50 and ema_50 < ema_200:
        ema_trend = 0.4  # 약한 상승
    elif current_price < ema_50 and ema_50 > ema_200:
        ema_trend = 0.4  # 약한 하락
    else:
        ema_trend = 0.0  # 횡보
    
    # 2. 볼린저 밴드 위치 (더 엄격한 조건)
    bb_trend = 0.0
    bb_upper = df['BB_upper'].iloc[-1]  # bb_upper → BB_upper
    bb_lower = df['BB_lower'].iloc[-1]  # bb_lower → BB_lower
    bb_middle = df['BB_basis'].iloc[-1]  # bb_middle → BB_basis
    
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
    
    if bb_position > 0.8:  # 상단 밴드 근처
        bb_trend = -0.6  # 과매수 신호 (SELL 유리)
    elif bb_position < 0.2:  # 하단 밴드 근처
        bb_trend = 0.6   # 과매도 신호 (BUY 유리)
    elif 0.4 <= bb_position <= 0.6:  # 중앙 밴드
        bb_trend = 0.0   # 중립
    else:
        bb_trend = 0.0   # 중립
    
    # 3. MACD 신호 (더 엄격한 조건)
    macd_trend = 0.0
    macd_line = df['MACD'].iloc[-1]  # macd → MACD
    macd_signal = df['MACD_signal'].iloc[-1]  # macd_signal → MACD_signal
    macd_hist = df['MACD_hist'].iloc[-1]  # macd_hist → MACD_hist
    
    if macd_line > macd_signal and macd_hist > 0:
        macd_trend = 0.7  # 상승 모멘텀
    elif macd_line < macd_signal and macd_hist < 0:
        macd_trend = 0.7  # 하락 모멘텀
    elif abs(macd_hist) < 0.1:  # MACD 히스토그램이 작음 (횡보)
        macd_trend = 0.0  # 중립
    else:
        macd_trend = 0.3  # 약한 신호
    
    # 4. StochRSI (더 엄격한 조건)
    stoch_trend = 0.0
    stoch_k = df['StochRSI_K'].iloc[-1]  # stoch_k → StochRSI_K
    stoch_d = df['StochRSI_D'].iloc[-1]  # stoch_d → StochRSI_D
    
    if stoch_k > 80 and stoch_d > 80:
        stoch_trend = -0.6  # 과매수 (SELL 유리)
    elif stoch_k < 20 and stoch_d < 20:
        stoch_trend = 0.6   # 과매도 (BUY 유리)
    elif 40 <= stoch_k <= 60 and 40 <= stoch_d <= 60:
        stoch_trend = 0.0   # 중립 (횡보)
    else:
        stoch_trend = 0.0   # 중립
    
    # 5. 가격 모멘텀 (새로 추가)
    momentum_trend = 0.0
    price_change_5 = df['price_change_5'].iloc[-1]
    price_change_1 = df['price_change'].iloc[-1]  # price_change_1 → price_change
    
    if price_change_5 > 0.5 and price_change_1 > 0.1:  # 강한 상승 모멘텀
        momentum_trend = 0.6
    elif price_change_5 < -0.5 and price_change_1 < -0.1:  # 강한 하락 모멘텀
        momentum_trend = -0.6
    elif abs(price_change_5) < 0.2:  # 횡보
        momentum_trend = 0.0
    else:
        momentum_trend = 0.0
    
    # 6. 종합 트렌드 계산 (가중치 조정)
    trend_score = (
        ema_trend * 0.25 +
        bb_trend * 0.20 +
        macd_trend * 0.20 +
        stoch_trend * 0.20 +
        momentum_trend * 0.15
    )
    
    # 7. 트렌드 결정 (더 완화된 조건)
    if trend_score > 0.2:  # 0.4 → 0.2로 완화
        trend = "BULLISH"
        strength = min(trend_score, 0.9)
    elif trend_score < -0.2:  # -0.4 → -0.2로 완화
        trend = "BEARISH"
        strength = min(abs(trend_score), 0.9)
    else:
        trend = "NEUTRAL"
        strength = 0.0
    
    # 8. 횡보장 감지 및 대응
    volatility = df['volatility'].iloc[-1]
    if volatility < 0.3:  # 낮은 변동성 (횡보장)
        if trend == "NEUTRAL":
            strength = 0.0  # 횡보장에서는 중립 유지
        else:
            strength *= 0.7  # 횡보장에서는 신호 강도 감소
    
    return {
        "trend": trend,
        "strength": strength,
        "reason": f"EMA:{ema_trend:.1f}, BB:{bb_trend:.1f}, MACD:{macd_trend:.1f}, Stoch:{stoch_trend:.1f}, Momentum:{momentum_trend:.1f}"
    }

def analyze_5m_entry(df: pd.DataFrame, trend_15m: Dict, config: HybridConfig) -> Dict[str, Any]:
    """5분봉 진입 타이밍 분석 (BUY/SELL 균형)"""
    if len(df) < 30:
        return {"action": "WAIT", "strength": 0.0, "reasons": ["5분봉 데이터 부족"]}
    
    current_price = df['close'].iloc[-1]
    ema_50 = df['EMA_50'].iloc[-1]  # ema_50 → EMA_50
    ema_200 = df['EMA_200'].iloc[-1]  # ema_200 → EMA_200
    
    reasons = []
    entry_score = 0.0
    max_score = 0.0
    
    # 1. EMA 크로스오버 (가중치: 3)
    max_score += 3
    if len(df) >= 2:
        prev_ema_50 = df['EMA_50'].iloc[-2]
        prev_ema_200 = df['EMA_200'].iloc[-2]
        
        # 골든 크로스 (상승 신호)
        if prev_ema_50 <= prev_ema_200 and ema_50 > ema_200:
            entry_score += 3
            reasons.append("5분봉: 골든 크로스 (강한 BUY)")
        # 데드 크로스 (하락 신호)
        elif prev_ema_50 >= prev_ema_200 and ema_50 < ema_200:
            entry_score += 0
            reasons.append("5분봉: 데드 크로스 (강한 SELL)")
        # 기존 트렌드 유지
        elif ema_50 > ema_200:
            entry_score += 2
            reasons.append("5분봉: 상승 트렌드 유지")
        else:
            entry_score += 1
            reasons.append("5분봉: 하락 트렌드 유지")
    
    # 2. 볼린저 밴드 진입 (가중치: 2.5)
    max_score += 2.5
    bb_upper = df['BB_upper'].iloc[-1]  # bb_upper → BB_upper
    bb_lower = df['BB_lower'].iloc[-1]  # bb_lower → BB_lower
    bb_middle = df['BB_basis'].iloc[-1]  # bb_middle → BB_basis
    
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
    
    if bb_position < 0.2:  # 하단 밴드 근처 (BUY 유리)
        entry_score += 2.5
        reasons.append("5분봉: BB 하단 근처 (BUY 유리)")
    elif bb_position > 0.8:  # 상단 밴드 근처 (SELL 유리)
        entry_score += 0
        reasons.append("5분봉: BB 상단 근처 (SELL 유리)")
    elif 0.3 <= bb_position <= 0.7:  # 중앙 밴드
        entry_score += 1.5
        reasons.append("5분봉: BB 중앙 구간 (중립)")
    else:
        entry_score += 1.0
        reasons.append("5분봉: BB 경계 구간")
    
    # 3. MACD 모멘텀 (가중치: 2)
    max_score += 2
    macd_line = df['MACD'].iloc[-1]  # macd → MACD
    macd_signal = df['MACD_signal'].iloc[-1]  # macd_signal → MACD_signal
    macd_hist = df['MACD_hist'].iloc[-1]  # macd_hist → MACD_hist
    
    if len(df) >= 2:
        prev_macd_hist = df['MACD_hist'].iloc[-2]
        
        if macd_line > macd_signal and macd_hist > prev_macd_hist:
            entry_score += 2
            reasons.append("5분봉: MACD 상승 모멘텀")
        elif macd_line < macd_signal and macd_hist < prev_macd_hist:
            entry_score += 0
            reasons.append("5분봉: MACD 하락 모멘텀")
        elif macd_line > macd_signal:
            entry_score += 1.5
            reasons.append("5분봉: MACD 상승 신호")
        elif macd_line < macd_signal:
            entry_score += 0.5
            reasons.append("5분봉: MACD 하락 신호")
        else:
            entry_score += 1.0
            reasons.append("5분봉: MACD 중립")
    
    # 4. StochRSI 과매수/과매도 (가중치: 2)
    max_score += 2
    stoch_k = df['StochRSI_K'].iloc[-1]  # stoch_k → StochRSI_K
    stoch_d = df['StochRSI_D'].iloc[-1]  # stoch_d → StochRSI_D
    
    if stoch_k < 20 and stoch_d < 20:
        entry_score += 2
        reasons.append("5분봉: StochRSI 과매도 (BUY 유리)")
    elif stoch_k > 80 and stoch_d > 80:
        entry_score += 0
        reasons.append("5분봉: StochRSI 과매수 (SELL 유리)")
    elif stoch_k < 30 and stoch_d < 30:
        entry_score += 1.5
        reasons.append("5분봉: StochRSI 약한 과매도")
    elif stoch_k > 70 and stoch_d > 70:
        entry_score += 0.5
        reasons.append("5분봉: StochRSI 약한 과매수")
    else:
        entry_score += 1.0
        reasons.append("5분봉: StochRSI 중립")
    
    # 5. 거래량 확인 (가중치: 1.5)
    max_score += 1.5
    volume_ratio = df.get('volume_ratio', 1.0).iloc[-1] if 'volume_ratio' in df.columns else 1.0
    
    if volume_ratio > 1.2:
        entry_score += 1.5
        reasons.append("5분봉: 거래량 증가")
    elif volume_ratio > 1.0:
        entry_score += 1.0
        reasons.append("5분봉: 거래량 보통")
    else:
        entry_score += 0.5
        reasons.append("5분봉: 거래량 감소")
    
    # 6. 가격 모멘텀 (가중치: 1)
    max_score += 1
    price_change_5 = df['price_change_5'].iloc[-1]
    price_change_1 = df['price_change'].iloc[-1]  # price_change_1 → price_change
    
    if price_change_5 > 0.3 and price_change_1 > 0.05:
        entry_score += 1
        reasons.append("5분봉: 상승 모멘텀")
    elif price_change_5 < -0.3 and price_change_1 < -0.05:
        entry_score += 0
        reasons.append("5분봉: 하락 모멘텀")
    elif abs(price_change_5) < 0.1:
        entry_score += 0.5
        reasons.append("5분봉: 가격 안정")
    else:
        entry_score += 0.5
        reasons.append("5분봉: 가격 변동")
    
    # 진입 강도 계산
    entry_strength = entry_score / max_score if max_score > 0 else 0.0
    
    # 진입 액션 결정 (BUY/SELL 균형) - 더 공격적
    if entry_strength > 0.4:  # 0.5 → 0.4로 더 완화
        action = "BUY"
    elif entry_strength < 0.4:  # 0.5 → 0.4로 더 완화 (SELL 신호 증가)
        action = "SELL"
    else:
        action = "WAIT"
    
    # 횡보장 감지 및 대응
    volatility = df['volatility'].iloc[-1]
    if volatility < 0.3:  # 낮은 변동성 (횡보장)
        if action == "WAIT":
            entry_strength *= 0.8  # 횡보장에서는 신호 강도 감소
        else:
            entry_strength *= 0.9  # 횡보장에서는 신호 강도 약간 감소

    return {
        "action": action,
        "strength": entry_strength,
        "reasons": reasons,
        "bb_position": bb_position
    }

def calculate_hybrid_confidence(trend_15m: Dict[str, Any], entry_5m: Dict[str, Any], cfg: HybridConfig) -> float:
    """
    하이브리드 신뢰도 계산
    """
    trend_strength = trend_15m["strength"]
    entry_strength = entry_5m["strength"]
    
    # 트렌드와 진입 방향 일치성 확인
    trend_direction = trend_15m["trend"]
    entry_signal = entry_5m["action"] # Changed from "signal" to "action"
    
    direction_match = 1.0
    if trend_direction == "BULLISH" and entry_signal in ["BUY", "WAIT"]: # Changed from "LONG" to "BUY"
        direction_match = 1.0
    elif trend_direction == "BEARISH" and entry_signal in ["SELL", "WAIT"]: # Changed from "SHORT" to "SELL"
        direction_match = 1.0
    elif trend_direction == "NEUTRAL":
        direction_match = 0.5
    else:
        direction_match = 0.2  # 방향 불일치
    
    # 가중 평균으로 최종 신뢰도 계산
    hybrid_confidence = (
        trend_strength * cfg.trend_weight +
        entry_strength * cfg.entry_weight
    ) * direction_match
    
    return min(1.0, max(0.0, hybrid_confidence))

def calculate_risk_reward(df_5m: pd.DataFrame, signal: str, cfg: HybridConfig) -> Tuple[float, float, float, float]:
    """
    리스크/보상 비율 계산
    """
    if df_5m.empty:
        return 0.0, 0.0, 0.0, 0.0
    
    current_price = df_5m.iloc[-1]["close"]
    atr = df_5m.iloc[-1].get(f"ATR_{cfg.atr_len}", current_price * 0.02)
    
    if signal == "BUY":
        stop_loss = current_price - (atr * cfg.atr_stop_mult)
        take_profit1 = current_price + (atr * cfg.atr_tp1_mult)
        take_profit2 = current_price + (atr * cfg.atr_tp2_mult)
    elif signal == "SELL":
        stop_loss = current_price + (atr * cfg.atr_stop_mult)
        take_profit1 = current_price - (atr * cfg.atr_tp1_mult)
        take_profit2 = current_price - (atr * cfg.atr_tp2_mult)
    else:
        return 0.0, 0.0, 0.0, 0.0
    
    # 리스크/보상 비율 계산 (첫 번째 익절 기준)
    risk = abs(current_price - stop_loss)
    reward = abs(take_profit1 - current_price)
    rr_ratio = reward / risk if risk > 0 else 0.0
    
    return stop_loss, take_profit1, take_profit2, rr_ratio

def make_hybrid_trade_plan(df_15m: pd.DataFrame, df_5m: pd.DataFrame, vpvr_levels: List[Dict], cfg: HybridConfig) -> Dict[str, Any]:
    """하이브리드 거래 계획 생성 (완전히 재설계)"""
    if df_15m.empty or df_5m.empty:
        return None
    
    # 15분봉 트렌드 분석
    trend_15m = analyze_15m_trend(df_15m, vpvr_levels, cfg)
    
    # 5분봉 진입 분석
    entry_5m = analyze_5m_entry(df_5m, trend_15m, cfg)
    
    # 하이브리드 신뢰도 계산
    trend_weight = cfg.trend_weight
    entry_weight = cfg.entry_weight
    
    confidence = (trend_15m["strength"] * trend_weight + 
                 entry_5m["strength"] * entry_weight)
    
    # 최종 신호 결정 (완전히 새로운 로직)
    final_signal = "NEUTRAL"
    
    # 1. 기본 신호 결정 (더 공격적)
    if entry_5m["action"] == "BUY" and entry_5m["strength"] >= 0.3:  # 0.4 → 0.3으로 더 완화
        final_signal = "BUY"
    elif entry_5m["action"] == "SELL" and entry_5m["strength"] >= 0.3:  # 0.4 → 0.3으로 더 완화
        final_signal = "SELL"
    
    # 2. 트렌드와 일치하는 경우 신호 강화
    if final_signal == "BUY" and trend_15m["trend"] == "BULLISH":
        confidence *= 1.2  # 신뢰도 20% 증가
    elif final_signal == "SELL" and trend_15m["trend"] == "BEARISH":
        confidence *= 1.2  # 신뢰도 20% 증가
    
    # 3. 트렌드와 반대인 경우 신호 약화 (하지만 완전히 차단하지 않음)
    if final_signal == "BUY" and trend_15m["trend"] == "BEARISH":
        confidence *= 0.8  # 신뢰도 20% 감소
    elif final_signal == "SELL" and trend_15m["trend"] == "BULLISH":
        confidence *= 0.8  # 신뢰도 20% 감소
    
    # 리스크/보상 계산 (NEUTRAL이 아닌 경우에만)
    current_price = df_5m['close'].iloc[-1]
    atr = df_5m.iloc[-1].get(f"ATR_{cfg.atr_len}", current_price * 0.02)
    
    if final_signal == "BUY":
        stop_loss = current_price - (atr * cfg.atr_stop_mult)
        take_profit1 = current_price + (atr * cfg.atr_tp1_mult)
        take_profit2 = current_price + (atr * cfg.atr_tp2_mult)
    elif final_signal == "SELL":
        stop_loss = current_price + (atr * cfg.atr_stop_mult)
        take_profit1 = current_price - (atr * cfg.atr_tp1_mult)
        take_profit2 = current_price - (atr * cfg.atr_tp2_mult)
    else:
        stop_loss = take_profit1 = take_profit2 = current_price
    
    # 리스크/보상 비율 계산
    if final_signal != "NEUTRAL":
        risk = abs(current_price - stop_loss)
        reward = abs(take_profit1 - current_price)
        risk_reward = reward / risk if risk > 0 else 0
    else:
        risk_reward = 0
    
    # VPVR 헤드룸 확인
    vpvr_headroom = 0
    if vpvr_levels:
        try:
            # vpvr_levels가 튜플인 경우와 딕셔너리인 경우를 모두 처리
            if isinstance(vpvr_levels[0], tuple):
                # 튜플 형태: (price, volume)
                prices = [level[0] for level in vpvr_levels if len(level) > 0]
            elif isinstance(vpvr_levels[0], dict):
                # 딕셔너리 형태: {'price': value, ...}
                prices = [level.get('price', 0) for level in vpvr_levels if level.get('price')]
            else:
                # 기타 형태
                prices = [level for level in vpvr_levels if isinstance(level, (int, float))]
            
            if prices:
                nearest_resistance = min([p for p in prices if p > current_price], default=current_price * 1.1)
                vpvr_headroom = (nearest_resistance - current_price) / current_price
        except Exception as e:
            print(f"VPVR 헤드룸 계산 중 오류: {e}")
            vpvr_headroom = 0
    
    return {
        "timestamp": datetime.now(),
        "signal_type": "HYBRID",  # 신호 타입 추가
        "final_signal": final_signal,
        "confidence": confidence,
        "risk_reward": risk_reward,
        "current_price": current_price,
        "stop_loss": stop_loss,
        "take_profit1": take_profit1,
        "take_profit2": take_profit2,
        "atr": atr,
        "trend_15m": trend_15m,
        "entry_5m": entry_5m,
        "vpvr_headroom": vpvr_headroom,
        "reasons": [
            f"15분봉: {trend_15m['reason']}",
            f"5분봉: {', '.join(entry_5m['reasons'])}"
        ]
    }
