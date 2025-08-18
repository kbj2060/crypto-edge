import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from utils.config import RunConfig

# ---------- 내부 유틸 ----------

def _bool(x) -> bool:
    try:
        return bool(x)
    except Exception:
        return False

def _nonan(*vals):
    return [v for v in vals if v is not None and pd.notna(v)]

def _slope_up(curr: float, prev: float) -> bool:
    return curr > prev

def _slope_down(curr: float, prev: float) -> bool:
    return curr < prev

def _nearest_vpvr_barrier(price: float, vpvr_levels) -> Tuple[Optional[float], Optional[float]]:
    if not vpvr_levels:
        return (None, None)
    uppers = sorted([lvl for (lvl, _) in vpvr_levels if lvl >= price])
    lowers = sorted([lvl for (lvl, _) in vpvr_levels if lvl <= price], reverse=True)
    up = uppers[0] if uppers else None
    dn = lowers[0] if lowers else None
    return up, dn

def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def _detect_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """다이버전스 감지"""
    df = df.copy()
    
    # RSI 계산
    df['rsi'] = _calculate_rsi(df['close'], 14)
    
    # 가격과 RSI 다이버전스
    df['price_rsi_divergence'] = 0
    
    # 베어리시 다이버전스 (가격 상승, RSI 하락)
    price_up = df['close'] > df['close'].shift(10)
    rsi_down = df['rsi'] < df['rsi'].shift(10)
    df.loc[price_up & rsi_down & (df['rsi'] > 70), 'price_rsi_divergence'] = -1
    
    # 불리시 다이버전스 (가격 하락, RSI 상승)
    price_down = df['close'] < df['close'].shift(10)
    rsi_up = df['rsi'] > df['rsi'].shift(10)
    df.loc[price_down & rsi_up & (df['rsi'] < 30), 'price_rsi_divergence'] = 1
    
    return df

def _calculate_volume_confirmation(df: pd.DataFrame) -> pd.DataFrame:
    """거래량 확인"""
    df = df.copy()
    
    # 거래량 이동평균
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # 거래량 신호 강도 (float 타입으로 명시적 변환)
    df['volume_signal'] = 0.0
    df.loc[df['volume_ratio'] > 1.5, 'volume_signal'] = 1.0
    df.loc[df['volume_ratio'] > 1.2, 'volume_signal'] = 0.7
    df.loc[df['volume_ratio'] > 1.0, 'volume_signal'] = 0.5
    
    return df

# ---------- 향상된 컨디션 체크 ----------

def _long_conditions(row: pd.Series, prev: pd.Series, cfg: RunConfig, reasons: List[str]) -> int:
    confirms = 0

    # 1) EMA: 정배열 + 가격이 EMA50 위 (가중치: 2)
    if row["EMA_50"] > row["EMA_200"] and row["close"] > row["EMA_50"]:
        confirms += 2
        reasons.append("EMA: 정배열 & 종가>EMA50")

    # 2) Bollinger: 종가가 basis 위 + (선택) 폭 확장 (가중치: 1)
    bb_ok = row["close"] > row["BB_basis"]
    if cfg.require_bb_expand:
        bb_ok = bb_ok and _slope_up(row["BB_width"], prev["BB_width"])
    if bb_ok:
        confirms += 1
        reasons.append("BB: basis 위(±폭 확장)")

    # 3) MACD: 골든 + (선택) 히스토그램 개선 (가중치: 2)
    macd_ok = row["MACD"] > row["MACD_signal"]
    if cfg.require_macd_slope:
        macd_ok = macd_ok and (row["MACD_hist"] >= prev["MACD_hist"])
    if macd_ok and row["MACD_hist"] > 0:
        confirms += 2
        reasons.append("MACD: 골든 & hist 양수(개선)")

    # 4) StochRSI: K>D (가중치: 1)
    stoch_ok = row["StochRSI_K"] > row["StochRSI_D"]
    cross_from_below = prev["StochRSI_K"] <= prev["StochRSI_D"]
    if stoch_ok and (cross_from_below or row["StochRSI_K"] < 60):
        confirms += 1
        reasons.append("StochRSI: K>D(골든/상승)")

    # 5) RSI 다이버전스 (가중치: 1)
    if hasattr(row, 'price_rsi_divergence') and row['price_rsi_divergence'] == 1:
        confirms += 1
        reasons.append("RSI 불리시 다이버전스")

    # 6) 거래량 확인 (가중치: 1)
    if hasattr(row, 'volume_signal') and row['volume_signal'] > 0.5:
        confirms += 1
        reasons.append("거래량 확인")

    return confirms

def _short_conditions(row: pd.Series, prev: pd.Series, cfg: RunConfig, reasons: List[str]) -> int:
    confirms = 0

    # 1) EMA: 역배열 + 가격이 EMA50 아래 (가중치: 2)
    if row["EMA_50"] < row["EMA_200"] and row["close"] < row["EMA_50"]:
        confirms += 2
        reasons.append("EMA: 역배열 & 종가<EMA50")

    # 2) Bollinger: 종가가 basis 아래 + (선택) 폭 확장 (가중치: 1)
    bb_ok = row["close"] < row["BB_basis"]
    if cfg.require_bb_expand:
        bb_ok = bb_ok and _slope_up(row["BB_width"], prev["BB_width"])
    if bb_ok:
        confirms += 1
        reasons.append("BB: basis 아래(±폭 확장)")

    # 3) MACD: 데드 + (선택) 히스토그램 악화 (가중치: 2)
    macd_ok = row["MACD"] < row["MACD_signal"]
    if cfg.require_macd_slope:
        macd_ok = macd_ok and (row["MACD_hist"] <= prev["MACD_hist"])
    if macd_ok and row["MACD_hist"] < 0:
        confirms += 2
        reasons.append("MACD: 데드 & hist 음수(악화)")

    # 4) StochRSI: K<D (가중치: 1)
    stoch_ok = row["StochRSI_K"] < row["StochRSI_D"]
    cross_from_above = prev["StochRSI_K"] >= prev["StochRSI_D"]
    if stoch_ok and (cross_from_above or row["StochRSI_K"] > 40):
        confirms += 1
        reasons.append("StochRSI: K<D(데드/하락)")

    # 5) RSI 다이버전스 (가중치: 1)
    if hasattr(row, 'price_rsi_divergence') and row['price_rsi_divergence'] == -1:
        confirms += 1
        reasons.append("RSI 베어리시 다이버전스")

    # 6) 거래량 확인 (가중치: 1)
    if hasattr(row, 'volume_signal') and row['volume_signal'] > 0.5:
        confirms += 1
        reasons.append("거래량 확인")

    return confirms

# ---------- 향상된 트레이드 플랜 ----------

def make_trade_plan(df_tail: pd.DataFrame, vpvr_levels, cfg: RunConfig) -> Dict[str, Any]:
    """
    향상된 거래 계획 생성
    df_tail: 최근 최소 20개 봉 포함 (마지막 = 현재 봉)
    """
    if len(df_tail) < 20:
        return {"bias": "NEUTRAL", "reasons": ["데이터 부족"], "confidence": 0.0, "entry": None, "stop": None, "tp1": None, "tp2": None}

    # 다이버전스 및 거래량 확인 추가
    df_enhanced = _detect_divergence(df_tail)
    df_enhanced = _calculate_volume_confirmation(df_enhanced)

    row = df_enhanced.iloc[-1]
    prev = df_enhanced.iloc[-2]

    price = float(row["close"])
    atr = float(row.get(f"ATR_{cfg.atr_len}", 0.0))
    up, dn = _nearest_vpvr_barrier(price, vpvr_levels)

    reasons_long: List[str] = []
    reasons_short: List[str] = []

    # 조건 카운트 (가중치 적용)
    long_cnt = _long_conditions(row, prev, cfg, reasons_long)
    short_cnt = _short_conditions(row, prev, cfg, reasons_short)

    # VPVR 여유(헤드룸) 체크
    long_headroom_ok = True
    if up is not None:
        long_headroom_ok = ((up - price) / price) >= cfg.min_vpvr_headroom
        if not long_headroom_ok:
            reasons_long.append(f"VPVR 상단 여유 부족({(up - price)/price:.3%})")

    short_headroom_ok = True
    if dn is not None:
        short_headroom_ok = ((price - dn) / price) >= cfg.min_vpvr_headroom
        if not short_headroom_ok:
            reasons_short.append(f"VPVR 하단 여유 부족({(price - dn)/price:.3%})")

    plan: Dict[str, Any] = {
        "bias": "NEUTRAL",
        "confidence": 0.0,
        "reasons": [],
        "entry": None,
        "stop": None,
        "tp1": None,
        "tp2": None,
        "risk_reward_ratio": 0.0,
        "vpvr_up": up,
        "vpvr_dn": dn,
    }

    # -------- LONG 확정 --------
    if long_cnt >= cfg.min_confirms_long and long_headroom_ok:
        plan["bias"] = "LONG"
        plan["confidence"] = min(long_cnt / 8.0, 1.0)  # 최대 8점 기준
        plan["entry"] = price

        # 향상된 손절 계산
        stop_candidates = _nonan(row.get("EMA_50"), dn)
        if stop_candidates:
            base_stop = min(stop_candidates)
        else:
            base_stop = price * 0.995  # fallback

        stop = base_stop - cfg.atr_stop_mult_long * atr if atr > 0 else base_stop * 0.995
        plan["stop"] = float(stop)

        # 향상된 익절 계산
        tp1_candidates = _nonan(row.get("BB_upper"))
        tp1_rule = price + cfg.atr_tp1_mult * atr if atr > 0 else price * 1.004
        tp1 = min([tp1_rule] + tp1_candidates) if tp1_candidates else tp1_rule
        plan["tp1"] = float(tp1)

        if up and up > price:
            plan["tp2"] = float(up)
        else:
            plan["tp2"] = float(price + cfg.atr_tp2_mult * atr) if atr > 0 else float(price * 1.008)

        # 리스크/보상 비율 계산
        risk = price - stop
        reward = tp1 - price
        plan["risk_reward_ratio"] = reward / risk if risk > 0 else 0

        plan["reasons"] = reasons_long
        return plan

    # -------- SHORT 확정 --------
    if short_cnt >= cfg.min_confirms_short and short_headroom_ok:
        plan["bias"] = "SHORT"
        plan["confidence"] = min(short_cnt / 8.0, 1.0)  # 최대 8점 기준
        plan["entry"] = price

        # 향상된 손절 계산
        stop_candidates = _nonan(row.get("EMA_50"), up)
        if stop_candidates:
            base_stop = max(stop_candidates)
        else:
            base_stop = price * 1.005  # fallback

        stop = base_stop + cfg.atr_stop_mult_short * atr if atr > 0 else base_stop * 1.005
        plan["stop"] = float(stop)

        # 향상된 익절 계산
        tp1_candidates = _nonan(row.get("BB_lower"))
        tp1_rule = price - cfg.atr_tp1_mult * atr if atr > 0 else price * 0.996
        tp1 = max([tp1_rule] + tp1_candidates) if tp1_candidates else tp1_rule
        plan["tp1"] = float(tp1)

        if dn and dn < price:
            plan["tp2"] = float(dn)
        else:
            plan["tp2"] = float(price - cfg.atr_tp2_mult * atr) if atr > 0 else float(price * 0.992)

        # 리스크/보상 비율 계산
        risk = stop - price
        reward = price - tp1
        plan["risk_reward_ratio"] = reward / risk if risk > 0 else 0

        plan["reasons"] = reasons_short
        return plan

    # -------- NEUTRAL --------
    plan["reasons"] = [f"롱 컨펌={long_cnt}"] + reasons_long + [f"숏 컨펌={short_cnt}"] + reasons_short
    return plan

# ---------- 기존 함수들 (호환성 유지) ----------

def long_signal(row: pd.Series) -> bool:
    conds = []
    # EMA: 정배열 + 가격이 EMA50 위
    conds.append(row.get("EMA_50") is not None and row.get("EMA_200") is not None and row["EMA_50"] > row["EMA_200"] and row["close"] > row["EMA_50"])
    # Bollinger: 종가가 중심선 위, 폭 확장(직전 대비)
    conds.append(row["close"] > row["BB_basis"])
    # MACD: 골든 + 히스토그램 양수
    conds.append(row["MACD"] > row["MACD_signal"] and row["MACD_hist"] > 0)
    # Stoch RSI: 20 이하에서 K가 D 상향돌파(완화: K > D and K<50)
    conds.append(row["StochRSI_K"] > row["StochRSI_D"] and row["StochRSI_K"] < 50)
    # 조건 중 3개 이상 충족
    return sum(bool(c) for c in conds) >= 3

def short_signal(row: pd.Series) -> bool:
    conds = []
    # EMA: 역배열 + 가격이 EMA50 아래
    conds.append(row.get("EMA_50") is not None and row.get("EMA_200") is not None and row["EMA_50"] < row["EMA_200"] and row["close"] < row["EMA_50"])
    # Bollinger: 종가가 중심선 아래
    conds.append(row["close"] < row["BB_basis"])
    # MACD: 데드 + 히스토그램 음수
    conds.append(row["MACD"] < row["MACD_signal"] and row["MACD_hist"] < 0)
    # Stoch RSI: 80 이상에서 K가 D 하향돌파(완화: K < D and K>50)
    conds.append(row["StochRSI_K"] < row["StochRSI_D"] and row["StochRSI_K"] > 50)
    return sum(bool(c) for c in conds) >= 3

def nearest_vpvr_barrier(price: float, vpvr_levels) -> Tuple[Optional[float], Optional[float]]:
    """
    vpvr_levels: [(level_price, volume), ...] 내림차순
    가장 가까운 위/아래 레벨 반환 (상단저항, 하단지지)
    """
    if not vpvr_levels:
        return (None, None)
    uppers = sorted([lvl for lvl,_ in vpvr_levels if lvl >= price])
    lowers = sorted([lvl for lvl,_ in vpvr_levels if lvl <= price], reverse=True)
    up = uppers[0] if uppers else None
    dn = lowers[0] if lowers else None
    return (up, dn)

def make_trade_plan_legacy(row: pd.Series, vpvr_levels) -> Dict[str, Any]:
    price = row["close"]
    up, dn = nearest_vpvr_barrier(price, vpvr_levels)

    plan = {"bias": None, "entry": None, "stop": None, "tp1": None, "tp2": None, "vpvr_up": up, "vpvr_dn": dn}
    if long_signal(row):
        plan["bias"] = "LONG"
        plan["entry"] = price  # 시장가 가정
        # 손절: EMA50 아래/VPVR 하단 살짝 아래
        stop = min([x for x in [row.get("EMA_50"), dn] if x is not None] + [price * 0.995])
        plan["stop"] = stop
        # 익절: 볼밴 상단, VPVR 상단
        tp1 = row.get("BB_upper", price * 1.004)
        tp2 = up if up and up > price else price * 1.008
        plan["tp1"] = tp1
        plan["tp2"] = tp2
    elif short_signal(row):
        plan["bias"] = "SHORT"
        plan["entry"] = price
        # 손절: EMA50 위/VPVR 상단 살짝 위
        stop = max([x for x in [row.get("EMA_50"), up] if x is not None] + [price * 1.005])
        plan["stop"] = stop
        # 익절: 볼밴 하단, VPVR 하단
        tp1 = row.get("BB_lower", price * 0.996)
        tp2 = dn if dn and dn < price else price * 0.992
        plan["tp1"] = tp1
        plan["tp2"] = tp2
    else:
        plan["bias"] = "NEUTRAL"
    return plan
