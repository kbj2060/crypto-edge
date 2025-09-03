from datetime import datetime, timedelta, timezone
from dateutil import tz
from playwright.sync_api import sync_playwright
import random, time

import pytz

from utils.time_manager import get_time_manager

# 1) 팝업 닫기 함수 (필요할 때마다 호출)
def close_popups(page, tries: int = 3, sleep_ms: int = 400):
    """
    다양한 팝업 닫기 버튼/오버레이를 주기적으로 시도해서 닫는다.
    - tries: 시도 횟수
    - sleep_ms: 각 시도 사이 대기 (약간 랜덤 지연 포함)
    """
    close_selectors = [
        "i.popupCloseIcon.largeBannerCloser",
        "button[aria-label='Close']",
        "button:has-text('Close')",
        "button:has-text('No thanks')",
        "button:has-text('Maybe later')",
        ".modal button.close",
        ".overlay .closeButton",
        "button[aria-label='Dismiss']",
        "div[class*='overlay'] button.close",
    ]
    remove_selectors = [
        "div[class*='overlay']",
        "div[class*='modal']",
        "div[id*='reg']",
        "div[class*='registration']",
        "div[class*='signup']",
        "div[class*='paywall']",
        "div[class*='login']",
        "#popupBox",
        "#PromoteSignUp",
    ]

    for _ in range(tries):
        # 1) 닫기 버튼들 먼저 클릭
        for sel in close_selectors:
            try:
                page.locator(sel).first.click(timeout=300)
            except:
                pass
        # 2) 그래도 남으면 컨테이너 자체 제거(하드킬)
        for sel in remove_selectors:
            try:
                els = page.locator(sel)
                count = els.count()
                if count:
                    page.evaluate(
                        """(selector) => {
                            document.querySelectorAll(selector).forEach(el => el.remove());
                        }""",
                        sel
                    )
            except:
                pass

        # 3) 짧은 랜덤 지연 후 재시도
        jitter = random.randint(-120, 120)
        page.wait_for_timeout(max(50, sleep_ms + jitter))

# 2) 일정 시간 동안 주기적으로 팝업 훑기
def poll_popups(page, duration_sec: float = 15.0, interval_ms: int = 700):
    """duration 동안 interval마다 close_popups 실행."""
    start = time.time()
    while time.time() - start < duration_sec:
        close_popups(page, tries=1, sleep_ms=interval_ms)
        # interval은 close_popups 내부에서 대기하므로 여기서 추가 대기는 생략

def parse_calendar_table(page, only_high: bool = True) -> list[dict]:
    """
    Investing.com #economicCalendarData 테이블을 파싱해 이벤트 리스트를 반환.
    - only_country: 해당 국가만 추리려면 국가명(예: "United States"), 전체 수집은 None
    - only_high: 중요도 High(bull3)만 추릴지 여부
    반환 예시 원소:
    {
        "name": "ISM Manufacturing PMI (Aug)",
        "country": "United States",
        "currency": "USD",
        "importance": "High",
        "time_seoul": datetime(..., tzinfo=SEOUL),
        "actual": "48.7",
        "forecast": "49.0",
        "previous": "48.0",
        "event_id": "173",
    }
    """
    events = []
    rows = page.locator("#economicCalendarData tbody tr.js-event-item")
    row_count = rows.count()

    for i in range(row_count):
        tr = rows.nth(i)

        # 1) 국가/통화
        country = ""
        currency = ""
        try:
            span = tr.locator("td.flagCur span").first
            country = (span.get_attribute("title") or "").strip()
            # 통화코드는 td.flagCur의 텍스트에서 추출(예: 'USD')
            currency = tr.locator("td.flagCur").inner_text(timeout=200).split()[-1].strip()
        except:
            pass

        # 2) 중요도
        importance = "Other"
        try:
            imp_key = tr.locator("td.textNum.sentiment").get_attribute("data-img_key") or ""
            if "bull3" in imp_key:
                importance = "High"
            elif "bull2" in imp_key:
                importance = "Medium"
            elif "bull1" in imp_key:
                importance = "Low"
        except:
            pass

        if only_high and importance != "High":
            continue

        # 3) 이벤트명
        try:
            name = tr.locator("td.event a").inner_text(timeout=300).strip()
        except:
            # 앵커가 없을 때를 대비
            try:
                name = tr.locator("td.event").inner_text(timeout=300).strip()
            except:
                continue
        if not name:
            continue

        # 4) 시간 파싱
        #    - 기본: td.time.js-time의 HH:MM
        #    - “17 min” 등 상대 표시는 data-event-datetime에서 가져와 서울로 해석
        try:
            t_text = tr.locator("td.time.js-time").inner_text(timeout=250).strip()
        except:
            t_text = ""

        if t_text and ":" in t_text:
            try:
                hh, mm = t_text.split(":")
                utc_tz = pytz.timezone("UTC")

                today = datetime.now(utc_tz).date()
                dt_utc = datetime(today.year, today.month, today.day, int(hh), int(mm), tzinfo=utc_tz) + timedelta(hours=4)
            except:
                dt_utc = None

        if 'min' in t_text:
            dt_utc = datetime.now(utc_tz) + timedelta(minutes=int(t_text.split(" ")[0])) + timedelta(hours=4)

        # 5) 수치(Actual/Forecast/Previous)
        def safe_text(sel, timeout=200):
            try:
                return tr.locator(sel).inner_text(timeout=timeout).strip().replace("\xa0", " ")
            except:
                return ""

        actual   = safe_text("td.act")
        forecast = safe_text("td.fore")
        previous = safe_text("td.prev")

        # 6) event id (있으면 추출)
        try:
            event_id = tr.get_attribute("event_attr_id") or ""
        except:
            event_id = ""

        events.append({
            "name": name,
            "country": country,
            "currency": currency,
            "importance": importance,
            "time": dt_utc,
            "actual": actual,
            "forecast": forecast,
            "previous": previous,
            "event_id": event_id,
        })

    # 시간순 정렬
    events.sort(key=lambda x: x["time"])
    return events

def fetch_us_high_events_today(headless: bool = True) -> list[dict]:
    """
    Investing.com 경제 캘린더에서 오늘(서울 기준) 미국 High 이벤트 크롤링.
    - 팝업은 폴링 방식으로 계속 훑어서 닫는다.
    """
    url = "https://www.investing.com/economic-calendar/"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--disable-blink-features=AutomationControlled"]
        )
        ctx = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"),
            locale="en-US",
            timezone_id="America/New_York",  # ✅ 뉴욕 시간
        )
        page = ctx.new_page()

        # 0) 진입 직후 1차 정리
        page.goto(url, wait_until="domcontentloaded")
        close_popups(page, tries=3, sleep_ms=400)

        # 1) 필터 여는 단계마다 팝업 폴링
        for sel in ['button:has-text("Filters")', '#filterStateAnchor']:
            try:
                page.locator(sel).click(timeout=4000)
                poll_popups(page, duration_sec=3.0, interval_ms=600)
                break
            except:
                pass

        # 2) 국가: United States만 체크
        # All Countries 해제
        for sel in [
            'label:has-text("All Countries") input[type="checkbox"]',
            'input[name="country[]"][value="0"]'  # UI 변경 대비 여유
        ]:
            try:
                page.locator(sel).uncheck(timeout=2000)
                poll_popups(page, duration_sec=1.5, interval_ms=500)
                break
            except:
                pass

        # 3) 중요도: High만
        for sel in [
            'label:has-text("All importance") input[type="checkbox"]',
            'input[name="importance[]"][value="0"]'
        ]:
            try:
                page.locator(sel).uncheck(timeout=2000)
                poll_popups(page, duration_sec=1.0, interval_ms=500)
                break
            except:
                pass
        for sel in [
            'label:has-text("High") input[type="checkbox"]',
            'input[name="importance[]"][value="3"]'
        ]:
            try:
                page.locator(sel).check(timeout=2500)
                poll_popups(page, duration_sec=1.0, interval_ms=500)
                break
            except:
                pass

        # 4) Apply/Show Results
        for sel in ['button:has-text("Apply")', 'button:has-text("Show Results")', '#ecSubmitButton']:
            try:
                page.locator(sel).click(timeout=3000)
                poll_popups(page, duration_sec=3.0, interval_ms=600)
                break
            except:
                pass

        # 테이블 로드 대기 + 중간중간 팝업 훑기
        poll_popups(page, duration_sec=2.5, interval_ms=600)

        # 5) 테이블 파싱
        events = parse_calendar_table(page, only_high=True)
        
        ctx.close()
        browser.close()
        events.sort(key=lambda x: x["time"])
        return events

if __name__ == "__main__":
    
    print(events)