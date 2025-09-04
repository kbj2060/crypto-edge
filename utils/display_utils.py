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
    decision: decide_trade_realtime(...) 반환값
    사람이 보기 쉽게 해석해서 출력합니다.
    """
    if not decision or not isinstance(decision, dict):
        print("⚠️ decision이 비어있거나 형식이 잘못되었습니다.")
        return

    action = decision.get("action", "HOLD")
    net_score = decision.get("net_score", 0.0)
    reason = decision.get("reason", "")
    raw = decision.get("raw", {})
    sizing = decision.get("sizing", {})
    recommended_scale = decision.get("recommended_trade_scale", 0.0)
    oppositions = decision.get("oppositions", [])
    agree_counts = decision.get("agree_counts", {"BUY": 0, "SELL": 0})
    meta = decision.get("meta", {})

    # compute per-strategy signed contributions (if possible)
    contributions = []
    for name, info in (raw.items() if isinstance(raw, dict) else []):
        try:
            act = (info.get("action") or "").upper()
            score = float(info.get("score") or 0.0)
            conf = float(info.get("conf_factor") or 0.6)
            weight = float(info.get("weight") or 0.0)
            sign = 0
            if act == "BUY":
                sign = 1
            elif act == "SELL":
                sign = -1
            contrib = sign * score * conf * weight
            contributions.append((name, contrib, act, score, conf, weight))
        except Exception:
            # best-effort fallback
            contributions.append((name, 0.0, info.get("action"), info.get("score"), info.get("confidence"), info.get("weight")))

    # sort by absolute contribution descending
    contributions_sorted = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)

    # Header
    print("────────────────────────────────────────────────────────")
    print(f"🕒 Decision @ {meta.get('timestamp_utc', 'unknown')}")
    print(f"▶ 추천 액션: {action}    |   net_score={net_score:.3f}    |   recommended_scale={recommended_scale:.3f}")
    print(f"▶ 이유: {reason}")
    print("────────────────────────────────────────────────────────")

    # Top contributors
    if contributions_sorted:
        print("전략별 기여 (큰 순):")
        for (name, contrib, act, score, conf, weight) in contributions_sorted:
            # format contribution sign and percent-ish
            sign_sym = "+" if contrib > 0 else ("-" if contrib < 0 else " ")
            print(f" - {name:12s} | action={str(act):5s} | score={score:.3f} conf={conf:.2f} weight={weight:.2f} | contrib={sign_sym}{abs(contrib):.4f}")
    else:
        print("전략별 정보가 없습니다.")

    # human guidance
    print("────────────────────────────────────────────────────────")
    if action == "HOLD":
        # if hold, explain top reasons why
        reasons = []
        # net too small
        if abs(net_score) < 0.35:
            reasons.append("net_score가 작음 (잡음일 가능성)")
        if oppositions:
            reasons.append("상반되는 강한 신호 존재")
        if reasons:
            print("권고: HOLD (보류). 이유들:")
            for r in reasons:
                print(" -", r)
        else:
            print("권고: HOLD. 추가 확인 또는 더 강한 컨펌 대기.")
    else:
        # actionable suggestion
        print(f"권고: {action} — 실행 전 체크리스트:")
        # checklist items
        checklist = []
        # if any strong opposite exists -> warn
        if oppositions:
            checklist.append("상반되는 강한 신호 존재: 재확인 권장 (충돌 시 사이즈 축소)")
        # if recommended_scale small -> warn
        if recommended_scale < 0.35:
            checklist.append(f"권장 스케일이 작음 ({recommended_scale:.2f}) — 소량/스캘프 권장")
        # if confidence overall low (average conf factor small)
        avg_conf = 0.0
        if contributions_sorted:
            avg_conf = sum([c[4] for c in contributions_sorted]) / max(1.0, len(contributions_sorted))
        if avg_conf < 0.6:
            checklist.append("전반적 신뢰도 낮음(중·저) — 보수적 사이징 권장")
        # print checklist
        if checklist:
            for it in checklist:
                print(" -", it)
        else:
            print(" - 조건 양호: 설정한 사이즈로 진입 고려 가능")

    print("────────────────────────────────────────────────────────")
    print("")  # blank line for spacing


def print_llm_judgment(judge: dict) -> None:
    """
    LLM 판단 결과를 예쁘게 출력합니다.
    """
    if not judge or not isinstance(judge, dict):
        print("⚠️ LLM 판단 결과가 비어있거나 형식이 잘못되었습니다.")
        return

    decision = judge.get("decision", "HOLD")
    confidence = judge.get("confidence", 0.0)
    reason = judge.get("reason", "")

    # 결정에 따른 이모지 선택
    decision_emoji = {
        "BUY": "🟢",
        "SELL": "🔴", 
        "HOLD": "🟡"
    }.get(decision, "❓")

    # 신뢰도에 따른 색상/표시
    confidence_level = ""
    if confidence >= 0.8:
        confidence_level = "🔥 매우 높음"
    elif confidence >= 0.6:
        confidence_level = "📈 높음"
    elif confidence >= 0.4:
        confidence_level = "📊 보통"
    else:
        confidence_level = "⚠️ 낮음"

    print("🤖" + "="*60)
    print(f"🧠 LLM 최종 판단")
    print("🤖" + "="*60)
    print(f"{decision_emoji} 결정: {decision}")
    print(f"🎯 신뢰도: {confidence:.2f} ({confidence_level})")
    print(f"💭 이유: {reason}")
    print("🤖" + "="*60)
    print("")  # blank line for spacing


def print_trading_summary(signals: Dict[str, Any], decision: Dict[str, Any], judge: Dict[str, Any]) -> None:
    """
    트레이딩 요약 정보를 출력합니다.
    """
    print("📊" + "="*60)
    print("📈 트레이딩 요약")
    print("📊" + "="*60)
    
    # 신호 개수
    signal_count = len(signals) if signals else 0
    print(f"🎯 활성 신호: {signal_count}개")
    
    # 결정 정보
    action = decision.get("action", "HOLD")
    net_score = decision.get("net_score", 0.0)
    print(f"⚖️ 시스템 결정: {action} (net_score: {net_score:.3f})")
    
    # LLM 판단
    llm_decision = judge.get("decision", "HOLD")
    llm_confidence = judge.get("confidence", 0.0)
    print(f"🤖 LLM 판단: {llm_decision} (신뢰도: {llm_confidence:.2f})")
    
    # 최종 결정
    final_decision = llm_decision if llm_decision != "HOLD" else action
    print(f"✅ 최종 결정: {final_decision}")
    
    print("📊" + "="*60)
    print("")  # blank line for spacing
