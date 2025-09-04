from ast import Dict
import requests
from datetime import datetime

# === 환경설정 ===
TELEGRAM_TOKEN = "8350844521:AAHpbD5_ScI1kp_m8UQXQGh42IpWsYQpFKk"
CHAT_ID = "8056624519"

def send_telegram_message(action, net_score, decision, confidence):
    """
    텔레그램으로 메시지 전송
    """

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    msg = (
        f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"▶ Action: {action}\n"
        f"▶ Score: {net_score:.2f}\n"
        f"▶ Judge Decision: {decision}\n"
        f"▶ Confidence: {confidence:.2f}\n"
    )
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
