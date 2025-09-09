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
