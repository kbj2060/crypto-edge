from typing import Dict, Any
import requests
from datetime import datetime

# === 환경설정 ===
TELEGRAM_TOKEN = "8350844521:AAHpbD5_ScI1kp_m8UQXQGh42IpWsYQpFKk"
CHAT_ID = "8056624519"

def send_telegram_message(decision: Dict[str, Any]) -> None:
    """
    텔레그램으로 다중 포지션 메시지 전송 (독립적 다중 포지션 구조)
    """
    if not decision or not isinstance(decision, dict):
        print("⚠️ decision이 비어있거나 형식이 잘못되었습니다.")
        return

    # 새로운 구조에서 데이터 추출
    decisions = decision.get("decisions", {})
    conflicts = decision.get("conflicts", {})
    meta = decision.get("meta", {})
    
    # 활성 포지션만 필터링
    active_positions = {k: v for k, v in decisions.items() if v.get("action") != "HOLD"}
    
    # 메시지 구성
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    if not active_positions:
        # 모든 카테고리가 HOLD인 경우
        msg = f"🟡 *HOLD* - 모든 카테고리 대기 중\n"
        msg += f"🕒 {timestamp}\n"
        msg += f"📊 활성 포지션: 0개"
    else:
        # 활성 포지션이 있는 경우
        msg = f"📈 *Multi-Position Update*\n"
        msg += f"🕒 {timestamp}\n\n"
        
        # 각 활성 포지션별 요약
        for category_name, category_decision in active_positions.items():
            action = category_decision.get("action", "HOLD")
            net_score = category_decision.get("net_score", 0.0)
            leverage = category_decision.get("leverage", 1)
            max_holding = category_decision.get("max_holding_minutes", 0)
            strategies_count = len(category_decision.get("strategies_used", []))
            
            # 액션 이모지
            action_emoji = {"LONG": "🟢", "SHORT": "🔴"}.get(action, "❓")
            
            # 카테고리명 한글 변환
            category_kr = {
                "SHORT_TERM": "단기",
                "MEDIUM_TERM": "중기", 
                "LONG_TERM": "장기"
            }.get(category_name, category_name)
            
            msg += f"{action_emoji} *{category_kr}*: {action} "
            msg += f"(점수: {net_score:.2f}, {leverage}x, {max_holding}분, {strategies_count}개 전략)\n"
        
        # 포지션 크기 정보 (첫 번째 활성 포지션만)
        if active_positions:
            first_position = list(active_positions.values())[0]
            sizing = first_position.get("sizing", {})
            if sizing.get("qty") is not None:
                qty = sizing.get("qty", 0)
                risk_usd = sizing.get("risk_usd", 0)
                entry = sizing.get("entry_used")
                stop = sizing.get("stop_used")
                
                msg += f"\n💰 포지션: {qty:.4f} | 리스크: ${risk_usd:.1f}"
                if entry and stop:
                    msg += f"\n📊 진입: {entry:.2f} | 손절: {stop:.2f}"
        
        # 충돌 경고
        if conflicts.get("has_conflicts", False):
            conflict_count = len(conflicts.get("conflict_types", []))
            msg += f"\n\n⚠️ *충돌 감지*: {conflict_count}개"
    
    # 메시지 전송
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print("텔레그램 전송 실패:", response.text)
    except Exception as e:
        print("텔레그램 전송 에러:", e)

def send_telegram_agent_decision(agent_decision: Dict[str, Any]) -> None:
    """
    AI 에이전트의 거래 결정을 텔레그램으로 전송
    """
    if not agent_decision or not isinstance(agent_decision, dict):
        print("⚠️ agent_decision이 비어있거나 형식이 잘못되었습니다.")
        return

    # 안전한 추출 및 형변환
    def _to_float(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except Exception:
            return default

    # 데이터 추출
    ts = agent_decision.get("timestamp", "unknown")
    price = _to_float(agent_decision.get("current_price"))
    confidence = _to_float(agent_decision.get("ai_confidence"))
    signal_quality = agent_decision.get("signal_quality", {}) or {}
    action = agent_decision.get("action", "HOLD")
    reason = agent_decision.get("reason", "")
    pos_change = _to_float(agent_decision.get("position_change"))
    leverage = _to_float(agent_decision.get("target_leverage"), 1.0)
    holding_min = _to_float(agent_decision.get("target_holding_minutes"))
    qty = _to_float(agent_decision.get("quantity"))
    sl = agent_decision.get("stop_loss")
    tp = agent_decision.get("take_profit")

    # 이모지/라벨
    action_emoji = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "🟡"}.get(action, "❓")
    conf_level = (
        "🔥 매우 높음" if confidence >= 0.8 else
        ("📈 높음" if confidence >= 0.6 else ("📊 보통" if confidence >= 0.4 else "⚠️ 낮음"))
    )

    # 메시지 구성
    msg = f"🤖 *AI Trading Decision*\n"
    msg += f"🕒 {ts}\n\n"
    
    msg += f"{action_emoji} *{action}* (신뢰도: {confidence:.2f} - {conf_level})\n"
    msg += f"💵 현재가: {price:.4f}\n"
    
    if reason:
        msg += f"💭 이유: {reason}\n"
    
    # 신호 품질 정보
    if isinstance(signal_quality, dict) and signal_quality:
        try:
            hc = int(signal_quality.get("high_confidence_signals", 0) or 0)
            total = int(signal_quality.get("total_signals", 0) or 0)
            agree = _to_float(signal_quality.get("agreement_score"))
            overall = _to_float(signal_quality.get("overall_score"))
            msg += f"\n📊 신호 품질: {hc}/{total}개 고신뢰 | 합의도: {agree:.2f} | 종합: {overall:.2f}"
        except Exception:
            pass

    # 실행 파라미터
    msg += f"\n\n⚙️ *실행 파라미터:*\n"
    msg += f"• 포지션 변경: {pos_change:+.2f}\n"
    msg += f"• 레버리지: {leverage:.0f}x\n"
    msg += f"• 보유시간: {int(holding_min)}분\n"
    msg += f"• 수량: {qty:.4f}"
    
    if sl is not None or tp is not None:
        sl_str = f"{_to_float(sl):.4f}" if sl is not None else "-"
        tp_str = f"{_to_float(tp):.4f}" if tp is not None else "-"
        msg += f"\n• 손절/익절: {sl_str} / {tp_str}"

    # 메시지 전송
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print("텔레그램 AI 결정 전송 실패:", response.text)
    except Exception as e:
        print("텔레그램 AI 결정 전송 에러:", e)

def send_telegram_alert(message: str) -> None:
    """
    간단한 알림 메시지 전송
    """
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": f"🚨 *Alert*\n{message}",
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            print("텔레그램 알림 전송 실패:", response.text)
    except Exception as e:
        print("텔레그램 알림 전송 에러:", e)
