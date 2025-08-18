#!/usr/bin/env python3
"""
ETHUSDT 하이브리드 트레이딩 시스템
15분봉 트렌드 + 5분봉 진입 타이밍 결합
"""

import argparse
import os
from data.loader import build_df
from indicators.vpvr import vpvr_key_levels
from signals.hybrid_strategy import make_hybrid_trade_plan, HybridConfig

def parse_args():
    parser = argparse.ArgumentParser(description="ETHUSDT 하이브리드 트레이딩 시스템 (15분봉+5분봉)")
    parser.add_argument("--limit_15m", type=int, default=500, help="15분봉 캔들 개수")
    parser.add_argument("--limit_5m", type=int, default=500, help="5분봉 캔들 개수")
    parser.add_argument("--plot", action="store_true", help="플롯 저장")
    parser.add_argument("--plot_path", type=str, default="plots/ethusdt_hybrid.png", help="플롯 저장 경로")
    return parser.parse_args()

def _pp(label, p):
    """거래 계획 출력"""
    if not p:
        print(f"\n{label}: None")
        return
    
    print(f"\n{label}:")
    print(f"  bias : {p['bias']}")
    if 'confidence' in p:
        print(f"  confidence: {p['confidence']:.2f}")
    
    # None 값 처리
    entry = p.get('entry')
    stop = p.get('stop')
    tp1 = p.get('tp1')
    tp2 = p.get('tp2')
    
    print(f"  entry: {entry:.2f}" if entry is not None else "  entry: None")
    print(f"  stop : {stop:.2f}" if stop is not None else "  stop : None")
    print(f"  tp1  : {tp1:.2f}" if tp1 is not None else "  tp1  : None")
    print(f"  tp2  : {tp2:.2f}" if tp2 is not None else "  tp2  : None")
    
    if 'risk_reward_ratio' in p:
        rr = p['risk_reward_ratio']
        print(f"  R/R  : {rr:.2f}" if rr is not None else "  R/R  : None")

def main():
    args = parse_args()
    
    # ETHUSDT 고정 설정
    symbol = "ETHUSDT"
    
    print(f"\n[ETHUSDT 하이브리드 트레이더] {symbol}")
    print("=" * 50)
    
    # 하이브리드 설정 (더 여유롭게 조정)
    cfg = HybridConfig(
        symbol=symbol,
        interval_15m="15m",
        interval_5m="5m",
        limit_15m=args.limit_15m,
        limit_5m=args.limit_5m,
        trend_weight=0.6,
        entry_weight=0.7,
        min_hybrid_confidence=0.20,  # 0.4 → 0.20으로 낮춤
        atr_len=14,
        atr_stop_mult=1.0,
        atr_tp1_mult=1.5,
        atr_tp2_mult=2.5,
        vpvr_bins=50,
        vpvr_lookback=200,
        min_vpvr_headroom=0.002
    )
    
    print(f"📊 데이터 로딩 중...")
    print(f"  15분봉: {cfg.limit_15m}개 캔들")
    print(f"  5분봉: {cfg.limit_5m}개 캔들")
    
    # 15분봉 데이터 로드
    df_15m = build_df(
        symbol, 
        cfg.interval_15m, 
        cfg.limit_15m, 
        cfg.atr_len, 
        market="futures", 
        price_source="last", 
        ma_type="ema"
    )
    
    # 5분봉 데이터 로드
    df_5m = build_df(
        symbol, 
        cfg.interval_5m, 
        cfg.limit_5m, 
        cfg.atr_len, 
        market="futures", 
        price_source="last", 
        ma_type="ema"
    )
    
    if df_15m.empty or df_5m.empty:
        print("❌ 데이터 로딩 실패")
        return
    
    print(f"✅ 데이터 로딩 완료")
    print(f"  15분봉: {len(df_15m)}개 캔들 (최신: {df_15m.index[-1]})")
    print(f"  5분봉: {len(df_5m)}개 캔들 (최신: {df_5m.index[-1]})")
    
    # VPVR 레벨 계산 (15분봉 기준)
    vpvr_levels = vpvr_key_levels(
        df_15m, 
        bins=cfg.vpvr_bins, 
        lookback=min(cfg.vpvr_lookback, len(df_15m)), 
        topn=8
    )
    
    print(f"📈 VPVR 레벨 계산 완료 ({len(vpvr_levels)}개)")
    
    # 하이브리드 거래 계획 생성
    print(f"\n🔍 하이브리드 분석 중...")
    plan = make_hybrid_trade_plan(df_15m, df_5m, vpvr_levels, cfg)
    
    # 현재 가격 정보
    current_price_15m = df_15m.iloc[-1]['close']
    current_price_5m = df_5m.iloc[-1]['close']
    
    print(f"\n💰 현재 가격:")
    print(f"  15분봉: ${current_price_15m:.2f}")
    print(f"  5분봉: ${current_price_5m:.2f}")
    
    # 하이브리드 분석 결과 출력
    hybrid_info = plan.get("hybrid_info", {})
    trend_15m = hybrid_info.get("trend_15m", {})
    entry_5m = hybrid_info.get("entry_5m", {})
    
    print(f"\n📊 하이브리드 분석 결과:")
    print(f"  15분봉 트렌드: {trend_15m.get('trend', 'N/A')} (강도: {trend_15m.get('strength', 0):.2f})")
    print(f"  5분봉 진입: {entry_5m.get('signal', 'N/A')} (강도: {entry_5m.get('strength', 0):.2f})")
    print(f"  하이브리드 신뢰도: {plan.get('confidence', 0):.2f}")
    
    # 거래 계획 출력
    _pp("🎯 하이브리드 거래 계획", plan)
    
    # 이유 출력
    if plan.get("reasons"):
        print(f"\n📝 분석 이유:")
        for i, reason in enumerate(plan["reasons"], 1):
            print(f"  {i}. {reason}")
    
    # VPVR 레벨 정보
    if plan.get("vpvr_up") or plan.get("vpvr_dn"):
        print(f"\n🏗️ VPVR 레벨:")
        if plan.get("vpvr_up"):
            print(f"  저항선: ${plan['vpvr_up']:.2f}")
        if plan.get("vpvr_dn"):
            print(f"  지지선: ${plan['vpvr_dn']:.2f}")
    
    # 거래 권장사항
    bias = plan['bias']
    confidence = plan.get('confidence', 0)
    rr_ratio = plan.get('risk_reward_ratio', 0)
    
    print(f"\n💡 거래 권장사항:")
    if bias == 'LONG':
        if confidence >= 0.6 and rr_ratio >= 1.5:
            print(f"  🚀 강력한 LONG 신호! (신뢰도: {confidence:.2f}, R/R: {rr_ratio:.2f})")
        elif confidence >= 0.4:
            print(f"  📈 LONG 신호 (신뢰도: {confidence:.2f}, R/R: {rr_ratio:.2f})")
        else:
            print(f"  ⚠️ 약한 LONG 신호 (신뢰도: {confidence:.2f})")
    elif bias == 'SHORT':
        if confidence >= 0.6 and rr_ratio >= 1.5:
            print(f"  📉 강력한 SHORT 신호! (신뢰도: {confidence:.2f}, R/R: {rr_ratio:.2f})")
        elif confidence >= 0.4:
            print(f"  📉 SHORT 신호 (신뢰도: {confidence:.2f}, R/R: {rr_ratio:.2f})")
        else:
            print(f"  ⚠️ 약한 SHORT 신호 (신뢰도: {confidence:.2f})")
    else:
        print(f"  ⏸️ 명확한 신호 없음 (신뢰도: {confidence:.2f})")
        print(f"     - 15분봉 트렌드: {trend_15m.get('trend', 'N/A')}")
        print(f"     - 5분봉 진입: {entry_5m.get('signal', 'N/A')}")
    
    # 하이브리드 전략 설명
    print(f"\n🎯 하이브리드 전략 특징:")
    print(f"  • 15분봉: 큰 트렌드 방향 파악 (가중치: {cfg.trend_weight})")
    print(f"  • 5분봉: 정교한 진입 타이밍 (가중치: {cfg.entry_weight})")
    print(f"  • 최소 신뢰도: {cfg.min_hybrid_confidence}")
    print(f"  • 리스크 관리: ATR 기반 손절/익절")
    
    # 플롯 저장 (향후 구현)
    if args.plot:
        os.makedirs(os.path.dirname(args.plot_path) or ".", exist_ok=True)
        print(f"\n📊 플롯 기능은 향후 구현 예정")
        print(f"  저장 경로: {args.plot_path}")

if __name__ == "__main__":
    main()
