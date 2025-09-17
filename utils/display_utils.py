# display_utils.py
"""
트레이딩 관련 출력 유틸리티 함수들
- Decision interpretation 출력
- LLM judgment 출력
- 기타 트레이딩 관련 표시 함수들
"""

from typing import Dict, Any


def print_decision_interpretation(decision: dict) -> None:
    """
    decision: decide_trade_realtime(...) 반환값 (독립적 다중 포지션 구조)
    사람이 보기 쉽게 해석해서 출력합니다.
    """
    if not decision or not isinstance(decision, dict):
        print("⚠️ decision이 비어있거나 형식이 잘못되었습니다.")
        return

    # 새로운 구조에서 decisions와 conflicts 추출
    decisions = decision.get("decisions", {})
    conflicts = decision.get("conflicts", {})
    meta = decision.get("meta", {})

    # Header
    print("=" * 80)
    print(f"🕒 Multi-Category Decision @ {meta.get('timestamp_utc', 'unknown')}")
    print(f"📊 활성 포지션: {meta.get('active_positions', 0)}개 / {meta.get('total_categories', 0)}개 카테고리")
    print("=" * 80)

    # 각 카테고리별 결정 출력
    for category_name, category_decision in decisions.items():
        print(f"\n📈 {category_name} 카테고리")
        print("-" * 50)
        
        action = category_decision.get("action", "HOLD")
        net_score = category_decision.get("net_score", 0.0)
        reason = category_decision.get("reason", "")
        raw = category_decision.get("raw", {})
        sizing = category_decision.get("sizing", {})
        leverage = category_decision.get("leverage", 1)
        max_holding = category_decision.get("max_holding_minutes", 0)
        strategies_used = category_decision.get("strategies_used", [])
        timeframe = category_decision.get("meta", {}).get("timeframe", "unknown")

        # 액션에 따른 이모지
        action_emoji = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "🟡"}.get(action, "❓")
        
        print(f"{action_emoji} 액션: {action} | 점수: {net_score:.3f} | 레버리지: {leverage}x")
        print(f"⏱️ 보유기간: {max_holding}분 | 시간프레임: {timeframe}")
        print(f"💭 이유: {reason}")
        
        if strategies_used:
            print(f"🎯 사용 전략: {', '.join(strategies_used)}")
        
        # 포지션 크기 정보
        if action != "HOLD" and sizing:
            qty = sizing.get("qty")
            risk_usd = sizing.get("risk_usd", 0)
            entry = sizing.get("entry_used")
            stop = sizing.get("stop_used")
            
            if qty is not None:
                print(f"💰 포지션 크기: {qty:.4f} | 리스크: ${risk_usd:.2f}")
                if entry and stop:
                    print(f"📊 진입가: {entry:.4f} | 손절가: {stop:.4f}")

        # 시너지 엔진 분석 결과 출력
        if "synergy_meta" in category_decision.get("meta", {}):
            synergy_meta = category_decision["meta"]["synergy_meta"]
            
            if category_name == "SHORT_TERM":
                print("🧠 ShortTermSynergyEngine 분석:")
            elif category_name == "MEDIUM_TERM":
                print("🔍 MediumTermSynergyEngine 분석:")
            elif category_name == "LONG_TERM":
                print("📈 LongTermSynergyEngine 분석:")
            
            print(f"   🎯 신뢰도: {synergy_meta.get('confidence', 'UNKNOWN')}")
            print(f"   📊 시장 상황: {synergy_meta.get('market_context', 'UNKNOWN')}")
            print(f"   ⚖️ 매수 점수: {synergy_meta.get('buy_score', 0):.3f}")
            print(f"   ⚖️ 매도 점수: {synergy_meta.get('sell_score', 0):.3f}")
            print(f"   🔍 사용된 신호: {synergy_meta.get('signals_used', 0)}개")
            
            # 충돌 감지 결과
            detected_conflicts = synergy_meta.get('conflicts_detected', [])
            if detected_conflicts:
                print(f"   ⚠️ 충돌 감지: {', '.join(detected_conflicts)}")
            else:
                print(f"   ✅ 충돌 없음")
            
            # 보너스 적용 결과 (중기, 장기)
            bonuses_applied = synergy_meta.get('bonuses_applied', [])
            if bonuses_applied:
                print(f"   🎁 보너스 적용: {', '.join(bonuses_applied)}")
            
            # 장기 전략 특별 정보
            if category_name == "LONG_TERM":
                institutional_bias = synergy_meta.get('institutional_bias', 'NEUTRAL')
                macro_trend = synergy_meta.get('macro_trend_strength', 'WEAK')
                print(f"   🏛️ 기관 편향: {institutional_bias}")
                print(f"   🌍 거시 트렌드: {macro_trend}")
            
            # breakdown 정보가 있으면 출력
            if 'breakdown' in synergy_meta:
                breakdown = synergy_meta['breakdown']
                if breakdown.get('buy_signals'):
                    print("   🟢 매수 신호:")
                    for signal in breakdown['buy_signals']:
                        print(f"      - {signal['name']}: {signal['score']:.3f}")
                if breakdown.get('sell_signals'):
                    print("   🔴 매도 신호:")
                    for signal in breakdown['sell_signals']:
                        print(f"      - {signal['name']}: {signal['score']:.3f}")
        
        # 전략별 기여도 출력 (기존 로직)
        elif raw:
            print("📊 전략별 기여도:")
            contributions = []
            
            for name, info in raw.items():
                try:
                    act = (info.get("action") or "").upper()
                    score = float(info.get("score") or 0.0)
                    weight = float(info.get("weight") or 0.0)
                    sign = 1 if act == "BUY" else (-1 if act == "SELL" else 0)
                    contrib = sign * score * weight
                    contributions.append((name, contrib, act, score, weight))
                except Exception:
                    contributions.append((name, 0.0, info.get("action"), info.get("score"), info.get("weight")))
            
            # 기여도 순으로 정렬
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for (name, contrib, act, score, weight) in contributions:
                sign_sym = "+" if contrib > 0 else ("-" if contrib < 0 else " ")
                act_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(act, "⚪")
                print(f"  {act_emoji} {name:15s}  | score={score:.3f} weight={weight:.3f} | contrib={sign_sym}{abs(contrib):.4f}")

    # 충돌 정보 출력
    if conflicts.get("has_conflicts", False):
        print(f"\n⚠️ 포지션 충돌 감지!")
        print("-" * 50)
        print(f"🟢 LONG 카테고리: {', '.join(conflicts.get('long_categories', []))}")
        print(f"🔴 SHORT 카테고리: {', '.join(conflicts.get('short_categories', []))}")
        print("충돌 타입:")
        for conflict_type in conflicts.get("conflict_types", []):
            print(f"   - {conflict_type}")
        print("💡 권고: 반대 방향 포지션은 리스크 관리에 주의하세요.")
    else:
        print(f"\n✅ 포지션 충돌 없음")

    # 카테고리별 신호 요약 (한 줄로)
    print("\n📊 신호 요약:")
    signal_summary = []
    
    for category_name, category_decision in decisions.items():
        action = category_decision.get("action", "HOLD")
        net_score = category_decision.get("net_score", 0.0)
        
        # 액션에 따른 이모지
        action_emoji = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "🟡"}.get(action, "❓")
        
        # 카테고리별 약어
        category_short = {
            "SHORT_TERM": "단기",
            "MEDIUM_TERM": "중기", 
            "LONG_TERM": "장기",
            "SCALPING": "스캔핑"
        }.get(category_name, category_name)
        
        signal_summary.append(f"{category_short}  {action_emoji} ({net_score:.2f})")
    
    print("   " + " | ".join(signal_summary))

    print("=" * 80)
    print("")  # blank line for spacing




def print_trading_summary(signals: Dict[str, Any], decision: Dict[str, Any], judge: Dict[str, Any]) -> None:
    """
    트레이딩 요약 정보를 출력합니다. (독립적 다중 포지션 구조)
    """
    print("📊" + "="*60)
    print("📈 Multi-Category Trading Summary")
    print("📊" + "="*60)
    
    # 신호 개수
    signal_count = len(signals) if signals else 0
    print(f"🎯 활성 신호: {signal_count}개")
    
    # 새로운 구조에서 decisions 추출
    decisions = decision.get("decisions", {})
    conflicts = decision.get("conflicts", {})
    meta = decision.get("meta", {})
    
    # 카테고리별 요약
    print(f"📊 카테고리별 결정:")
    for category_name, category_decision in decisions.items():
        action = category_decision.get("action", "HOLD")
        net_score = category_decision.get("net_score", 0.0)
        leverage = category_decision.get("leverage", 1)
        strategies_count = len(category_decision.get("strategies_used", []))
        
        action_emoji = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "🟡"}.get(action, "❓")
        
        # 시너지 엔진 분석 정보
        if "synergy_meta" in category_decision.get("meta", {}):
            synergy_meta = category_decision["meta"]["synergy_meta"]
            confidence = synergy_meta.get('confidence', 'UNKNOWN')
            market_context = synergy_meta.get('market_context', 'UNKNOWN')
            
            if category_name == "SHORT_TERM":
                print(f"   {action_emoji} {category_name}: {action} (점수: {net_score:.3f}, 레버리지: {leverage}x, 전략: {strategies_count}개)")
                print(f"      🧠 단기 시너지: {confidence} 신뢰도, {market_context} 시장상황")
            elif category_name == "MEDIUM_TERM":
                bonuses = synergy_meta.get('bonuses_applied', [])
                bonus_info = f", 보너스: {len(bonuses)}개" if bonuses else ""
                print(f"   {action_emoji} {category_name}: {action} (점수: {net_score:.3f}, 레버리지: {leverage}x, 전략: {strategies_count}개)")
                print(f"      🔍 중기 시너지: {confidence} 신뢰도, {market_context} 시장상황{bonus_info}")
            elif category_name == "LONG_TERM":
                institutional_bias = synergy_meta.get('institutional_bias', 'NEUTRAL')
                macro_trend = synergy_meta.get('macro_trend_strength', 'WEAK')
                print(f"   {action_emoji} {category_name}: {action} (점수: {net_score:.3f}, 레버리지: {leverage}x, 전략: {strategies_count}개)")
                print(f"      📈 장기 시너지: {confidence} 신뢰도, {market_context} 시장상황")
                print(f"      🏛️ 기관편향: {institutional_bias}, 거시트렌드: {macro_trend}")
        else:
            print(f"   {action_emoji} {category_name}: {action} (점수: {net_score:.3f}, 레버리지: {leverage}x, 전략: {strategies_count}개)")
    
    # 활성 포지션 요약
    active_positions = meta.get("active_positions", 0)
    total_categories = meta.get("total_categories", 0)
    print(f"⚖️ 활성 포지션: {active_positions}개 / {total_categories}개 카테고리")
    
    # 충돌 정보
    if conflicts.get("has_conflicts", False):
        print(f"⚠️ 포지션 충돌: {len(conflicts.get('conflict_types', []))}개")
    else:
        print(f"✅ 포지션 충돌 없음")
    
    # LLM 판단 (기존 구조 유지)
    if judge:
        llm_decision = judge.get("decision", "HOLD")
        llm_confidence = judge.get("confidence", 0.0)
        print(f"🤖 LLM 판단: {llm_decision} (신뢰도: {llm_confidence:.2f})")
    
    print("📊" + "="*60)
    print("")  # blank line for spacing


def print_ai_final_decision(ai_decision: Dict[str, Any]) -> None:
    """
    강화학습 에이전트의 최종 거래 결정을 사람이 보기 좋게 출력합니다.
    expected keys:
      - timestamp, current_price, ai_confidence, signal_quality(dict)
      - position_change, target_leverage, target_holding_minutes
      - action, reason, quantity, stop_loss, take_profit
    """
    if not ai_decision or not isinstance(ai_decision, dict):
        print("⚠️ AI 결정이 비어있거나 형식이 잘못되었습니다.")
        return

    # 안전한 추출 및 형변환
    def _to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    ts = ai_decision.get("timestamp", "unknown")
    price = _to_float(ai_decision.get("current_price"))
    confidence = _to_float(ai_decision.get("ai_confidence"))
    signal_quality = ai_decision.get("signal_quality", {}) or {}
    action = ai_decision.get("action", "HOLD")
    reason = ai_decision.get("reason", "")
    pos_change = _to_float(ai_decision.get("position_change"))
    leverage = _to_float(ai_decision.get("target_leverage"), 1.0)
    holding_min = _to_float(ai_decision.get("target_holding_minutes"))
    qty = _to_float(ai_decision.get("quantity"))
    sl = ai_decision.get("stop_loss")
    tp = ai_decision.get("take_profit")

    # 이모지/라벨
    action_emoji = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "🟡"}.get(action, "❓")
    conf_level = (
        "🔥 매우 높음" if confidence >= 0.8 else
        ("📈 높음" if confidence >= 0.6 else ("📊 보통" if confidence >= 0.4 else "⚠️ 낮음"))
    )

    print("🤖" + "=" * 60)
    print("🧠 AI 최종 거래 결정")
    print("🤖" + "=" * 60)
    print(f"🕒 시각: {ts}")
    print(f"💵 현재가: {price:.4f}")
    print(f"{action_emoji} 액션: {action}")
    print(f"🎯 신뢰도: {confidence:.2f} ({conf_level})")
    if reason:
        print(f"💭 이유: {reason}")

    # 신호 품질 요약
    if isinstance(signal_quality, dict) and signal_quality:
        try:
            hc = int(signal_quality.get("high_confidence_signals", 0) or 0)
            total = int(signal_quality.get("total_signals", 0) or 0)
            agree = _to_float(signal_quality.get("agreement_score"))
            overall = _to_float(signal_quality.get("overall_score"))
            print("📊 신호 품질:")
            print(f"   - 높은 신뢰 신호: {hc}/{total}")
            print(f"   - 합의도: {agree:.2f} | 종합점수: {overall:.2f}")
        except Exception:
            pass

    # 포지션/리스크 요약
    print("⚙️ 실행 파라미터:")
    print(f"   - 포지션 변경: {pos_change:+.2f}")
    print(f"   - 레버리지: {leverage:.0f}x")
    print(f"   - 보유시간: {int(holding_min)}분")
    print(f"   - 수량: {qty:.4f}")
    if sl is not None or tp is not None:
        sl_str = f"{_to_float(sl):.4f}" if sl is not None else "-"
        tp_str = f"{_to_float(tp):.4f}" if tp is not None else "-"
        print(f"   - 손절/익절: {sl_str} / {tp_str}")

    print("🤖" + "=" * 60)
    print("")  # spacing