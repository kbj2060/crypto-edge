
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import pandas as pd

from indicators.global_indicators import get_atr, get_vwap
from utils.time_manager import get_time_manager

@dataclass
class SessionORLiteCfg:
    """
    Lightweight Opening Range strategy config.
    - or_minutes: minutes to build OR (session open -> lock)
    - valid_minutes_after_open: only trade within this window after session open
    - body_ratio_min: min candle body/range ratio for a breakout candle
    - retest_atr: ATR multiplier buffer around OR edge to validate retest
    - retest_atr_mult_short: extra buffer multiplier for SHORT side retest
    - atr_stop_mult: base stop sizing (used with 0.5xATR for OR anchor stop)
    - tp_R1 / tp_R2: targets in multiples of R
    - vwap_filter_mode: 'off' | 'location' | 'slope'
        - location: long c>=vwap, short c<=vwap
        - slope:   uses vwap_prev; long if vwap>=vwap_prev else short
    - allow_wick_break: allow wick-based breakout in addition to body close
    - wick_needs_body_sign: if wick breakout used, body must agree with direction
    """
    or_minutes: int = 30
    valid_minutes_after_open: int = 120
    body_ratio_min: float = 0.20

    retest_atr: float = 0.40
    retest_atr_mult_short: float = 1.5  # SHORT only buffer multiplier

    atr_stop_mult: float = 1.0
    tp_R1: float = 1.2
    tp_R2: float = 2.0
    tick: float = 0.1

    vwap_filter_mode: str = "off"  # 'off' | 'location' | 'slope'
    allow_wick_break: bool = True
    wick_needs_body_sign: bool = True


class SessionORLite:
    """Simplified Opening Range breakout→retest strategy."""

    def __init__(self, cfg: SessionORLiteCfg = SessionORLiteCfg()):
        self.cfg = cfg
        self.session_open: Optional[datetime] = None
        self.or_locked: bool = False
        self.or_high: Optional[float] = None
        self.or_low: Optional[float] = None
        self.traded_long: bool = False
        self.traded_short: bool = False
        self.time_manager = get_time_manager()

        # Simple debug counters to diagnose side bias
        self.debug = {
            "break_long": 0, "break_short": 0,
            "retest_long_miss": 0, "retest_short_miss": 0,
            "vwap_long_block": 0, "vwap_short_block": 0
        }

    # ---- lifecycle ----
    def on_session_open(self) -> None:
        """Call at session open (tz-aware UTC)."""
        self.session_open = self.time_manager.get_current_time()
        self.or_locked = False
        self.or_high = None
        self.or_low = None
        self.traded_long = False
        self.traded_short = False
        # reset debug for a clean session view
        for k in self.debug:
            self.debug[k] = 0
            
        print(f"🚀 [SESSION_OR_LITE] 새 세션 시작: {self.session_open.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📊 OR 설정 초기화: 잠금={self.or_locked}, 고점={self.or_high}, 저점={self.or_low}")
        print(f"   📊 거래 상태 초기화: 롱={self.traded_long}, 숏={self.traded_short}")
        print(f"   📊 디버그 카운터 초기화 완료")

    def _in_valid_window(self, now: datetime) -> bool:
        if not self.session_open:
            return False
        
        valid_end = self.session_open + timedelta(minutes=self.cfg.valid_minutes_after_open)
        is_valid = now <= valid_end
        
        print(f"   📊 유효 시간대 확인: {now.strftime('%H:%M:%S')} <= {valid_end.strftime('%H:%M:%S')} - {'✅' if is_valid else '❌'}")
        print(f"      📏 세션 시작: {self.session_open.strftime('%H:%M:%S')}")
        print(f"      📏 유효 시간: {self.cfg.valid_minutes_after_open}분")
        
        return is_valid

    # ---- main hook (3m close) ----
    def on_kline_close_3m(
        self,
        df3: pd.DataFrame,
        vwap_prev: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate on 3m candle close.
        df3: pandas DataFrame with columns open, high, low, close (3m)
        vwap, vwap_std, atr: session-anchored preferred (floats)
        vwap_prev: previous value for slope filtering (optional)
        returns: signal dict or None
        """
        now = self.time_manager.get_current_time()
        
        print(f"🔍 [SESSION_OR_LITE] 3m 캔들 분석 시작: {now.strftime('%H:%M:%S')}")
        print(f"   📊 세션 시작: {self.session_open.strftime('%H:%M:%S') if self.session_open else self.time_manager.get_next_session_start().strftime('%H:%M:%S')}")
        print(f"   📊 다음 세션 시작 남은 시간 : {(self.time_manager.get_next_session_start() - now).total_seconds() // 60}분")
        print(f"   📊 OR 잠금 상태: {self.or_locked}")
        print(f"   📊 OR 범위: {self.or_high:.2f} ~ {self.or_low:.2f}" if self.or_high and self.or_low else "   📊 OR 범위: 설정되지 않음")
        
        if not self.session_open or not self._in_valid_window(now):
            print(f"   ⚠️ 세션 조건 불만족: session_open={bool(self.session_open)}, valid_window={self._in_valid_window(now)}")
            return None
        if df3 is None or len(df3) < 2:
            print(f"   ⚠️ 데이터 부족: df3={df3 is not None}, len={len(df3) if df3 is not None else 0}")
            return None

        vwap, vwap_std = get_vwap()
        atr = get_atr()
        
        last = df3.iloc[-1]
        prev = df3.iloc[-2]
        o = float(last["open"]); h = float(last["high"]); l = float(last["low"]); c = float(last["close"])
        ph = float(prev["high"]); pl = float(prev["low"])

        # 1) Build/lock OR (include the last candle if now == or_end)
        or_end = self.session_open + timedelta(minutes=self.cfg.or_minutes)
        print(f"   📊 OR 구축 중: {now.strftime('%H:%M:%S')} <= {or_end.strftime('%H:%M:%S')} (OR 종료)")
        
        if not self.or_locked:
            hi = h; lo = l
            print(f"   🔧 OR 구축: 현재 고점={hi:.2f}, 저점={lo:.2f}")
            
            # include last candle when now == or_end
            if now <= or_end:
                self.or_high = hi if self.or_high is None else max(self.or_high, hi)
                self.or_low  = lo if self.or_low  is None else min(self.or_low,  lo)
                print(f"   🔧 OR 업데이트: 고점={self.or_high:.2f}, 저점={self.or_low:.2f}")
                
                if now < or_end:
                    print(f"   ⏳ OR 구축 중: {now.strftime('%H:%M:%S')} < {or_end.strftime('%H:%M:%S')}")
                    return None
                else:
                    print(f"   🔒 OR 잠금 시점: {now.strftime('%H:%M:%S')} == {or_end.strftime('%H:%M:%S')}")
            
            # lock here (either == or_end or first call after or_end)
            self.or_locked = True
            print(f"   🔒 OR 잠금 완료: 고점={self.or_high:.2f}, 저점={self.or_low:.2f}")
        else:
            print(f"   🔒 OR 이미 잠김: 고점={self.or_high:.2f}, 저점={self.or_low:.2f}")

        # safety
        if self.or_high is None or self.or_low is None or self.or_high <= self.or_low:
            return None

        # 2) Breakout qualification (body or wick-based, configurable)
        rng = h - l
        if rng <= 0:
            return None
            
        body = abs(c - o)
        body_ok = (body / rng) >= self.cfg.body_ratio_min
        
        print(f"   📊 돌파 조건 분석:")
        print(f"      📏 캔들 범위: {rng:.2f}, 바디: {body:.2f}")
        print(f"      📏 바디 비율: {body/rng:.2f:.2f} (최소 {self.cfg.body_ratio_min:.2f}) - {'✅' if body_ok else '❌'}")

        # wick based breakout allowance
        wick_break_long  = (h >= self.or_high + self.cfg.tick)
        wick_break_short = (l <= self.or_low  - self.cfg.tick)

        wick_body_ok_long  = (c > o) if self.cfg.wick_needs_body_sign else True
        wick_body_ok_short = (c < o) if self.cfg.wick_needs_body_sign else True

        print(f"      🔥 롱 돌파 조건:")
        print(f"         📈 고점: {h:.2f} >= {self.or_high + self.cfg.tick:.2f} (ORH + tick) - {'✅' if wick_break_long else '❌'}")
        print(f"         📊 바디 방향: {c:.2f} > {o:.2f} - {'✅' if wick_body_ok_long else '❌'}")
        
        print(f"      🔥 숏 돌파 조건:")
        print(f"         📉 저점: {l:.2f} <= {self.or_low - self.cfg.tick:.2f} (ORL - tick) - {'✅' if wick_break_short else '❌'}")
        print(f"         📊 바디 방향: {c:.2f} < {o:.2f} - {'✅' if wick_body_ok_short else '❌'}")

        break_long_ok = (body_ok and (c >= self.or_high + self.cfg.tick)) or                         (self.cfg.allow_wick_break and wick_break_long and wick_body_ok_long)
        break_short_ok = (body_ok and (c <= self.or_low  - self.cfg.tick)) or                          (self.cfg.allow_wick_break and wick_break_short and wick_body_ok_short)

        print(f"      🎯 최종 돌파 결과:")
        print(f"         🟢 롱 돌파: {'✅' if break_long_ok else '❌'}")
        print(f"         🔴 숏 돌파: {'✅' if break_short_ok else '❌'}")

        if break_long_ok:
            self.debug["break_long"] += 1
        if break_short_ok:
            self.debug["break_short"] += 1

        # 3) Retest near the OR edge (allow previous candle to count)
        buf_long  = self.cfg.retest_atr * float(atr)
        buf_short = self.cfg.retest_atr * self.cfg.retest_atr_mult_short * float(atr)
        
        print(f"   📊 리테스트 조건 분석:")
        print(f"      📏 ATR: {atr:.2f}")
        print(f"      📏 롱 리테스트 버퍼: {buf_long:.2f} (ATR × {self.cfg.retest_atr})")
        print(f"      📏 숏 리테스트 버퍼: {buf_short:.2f} (ATR × {self.cfg.retest_atr} × {self.cfg.retest_atr_mult_short})")
        
        # use min low for long (deeper touch), max high for short (shallower touch)
        min_low = min(l, pl)
        max_high = max(h, ph)
        
        touched_long  = (min_low >= self.or_high - buf_long) and (min_low <= self.or_high + buf_long)
        touched_short = (max_high <= self.or_low + buf_short) and (max_high >= self.or_low - buf_short)

        print(f"      🔄 롱 리테스트:")
        print(f"         📉 최저점: {min_low:.2f}")
        print(f"         📊 OR 상단 ±버퍼: {self.or_high - buf_long:.2f} ~ {self.or_high + buf_long:.2f}")
        print(f"         ✅ 터치: {'✅' if touched_long else '❌'}")
        
        print(f"      🔄 숏 리테스트:")
        print(f"         📈 최고점: {max_high:.2f}")
        print(f"         📊 OR 하단 ±버퍼: {self.or_low - buf_short:.2f} ~ {self.or_low + buf_short:.2f}")
        print(f"         ✅ 터치: {'✅' if touched_short else '❌'}")

        if not touched_long:
            self.debug["retest_long_miss"] += 1
        if not touched_short:
            self.debug["retest_short_miss"] += 1

        # 4) VWAP filter
        vwap_ok_long = vwap_ok_short = True
        mode = (self.cfg.vwap_filter_mode or "off").lower()
        
        print(f"   📊 VWAP 필터 분석:")
        print(f"      📏 현재 VWAP: {vwap:.2f}")
        print(f"      📏 이전 VWAP: {vwap_prev:.2f}" if vwap_prev is not None else "      📏 이전 VWAP: None")
        print(f"      📏 필터 모드: {mode}")
        
        if mode == "location":
            vwap_ok_long  = c >= float(vwap)
            vwap_ok_short = c <= float(vwap)
            print(f"      📍 위치 기반 필터:")
            print(f"         🟢 롱: {c:.2f} >= {vwap:.2f} - {'✅' if vwap_ok_long else '❌'}")
            print(f"         🔴 숏: {c:.2f} <= {vwap:.2f} - {'✅' if vwap_ok_short else '❌'}")
        elif mode == "slope" and vwap_prev is not None:
            slope_up = float(vwap) >= float(vwap_prev)
            vwap_ok_long, vwap_ok_short = slope_up, (not slope_up)
            print(f"      📈 기울기 기반 필터:")
            print(f"         📊 VWAP 기울기: {vwap:.2f} {'↗️' if slope_up else '↘️'} {vwap_prev:.2f}")
            print(f"         🟢 롱: {'✅' if vwap_ok_long else '❌'} (기울기 {'상승' if slope_up else '하락'})")
            print(f"         🔴 숏: {'✅' if vwap_ok_short else '❌'} (기울기 {'하락' if slope_up else '상승'})")
        else:
            print(f"      ⚪ VWAP 필터 비활성화")

        if not vwap_ok_long:
            self.debug["vwap_long_block"] += 1
        if not vwap_ok_short:
            self.debug["vwap_short_block"] += 1

        # 5) Signals (one per side per session)
        sigs = []
        
        print(f"   🎯 신호 생성 분석:")
        print(f"      📊 거래 상태: 롱={self.traded_long}, 숏={self.traded_short}")

        if (not self.traded_long) and break_long_ok and touched_long and vwap_ok_long:
            print(f"      🟢 롱 신호 생성 조건 만족!")
            print(f"         ✅ 돌파: {break_long_ok}, 리테스트: {touched_long}, VWAP: {vwap_ok_long}")
            
            entry = h + self.cfg.tick
            stop  = min(l, self.or_high - 0.5*float(atr)) - self.cfg.tick
            R = entry - stop
            tp1, tp2 = entry + self.cfg.tp_R1 * R, entry + self.cfg.tp_R2 * R
            
            print(f"         📊 가격 계산:")
            print(f"            진입가: {entry:.2f} (고점 {h:.2f} + tick {self.cfg.tick})")
            print(f"            스탑가: {stop:.2f} (최소값: {l:.2f}, OR상단-0.5ATR: {self.or_high - 0.5*float(atr):.2f})")
            print(f"            리스크: {R:.2f}")
            print(f"            TP1: {tp1:.2f} (진입가 + {self.cfg.tp_R1}R)")
            print(f"            TP2: {tp2:.2f} (진입가 + {self.cfg.tp_R2}R)")
            
            self.traded_long = True
            sigs.append({
                "stage": "ENTRY", "action": "BUY", "entry": float(entry), "stop": float(stop),
                "targets": [float(tp1), float(tp2)],
                "context": {
                    "mode": "SESSION_OR_LITE", "or_high": float(self.or_high),
                    "atr": float(atr), "vwap": float(vwap), "vwap_std": float(vwap_std),
                    "touched_buf": float(buf_long), "body_ok": body_ok, "wick_break": wick_break_long
                }
            })
        else:
            print(f"      ❌ 롱 신호 생성 조건 불만족:")
            print(f"         돌파: {break_long_ok}, 리테스트: {touched_long}, VWAP: {vwap_ok_long}, 이미거래: {self.traded_long}")

        if (not self.traded_short) and break_short_ok and touched_short and vwap_ok_short:
            print(f"      🔴 숏 신호 생성 조건 만족!")
            print(f"         ✅ 돌파: {break_short_ok}, 리테스트: {touched_short}, VWAP: {vwap_ok_short}")
            
            entry = l - self.cfg.tick
            stop  = max(h, self.or_low + 0.5*float(atr)) + self.cfg.tick
            R = stop - entry
            tp1, tp2 = entry - self.cfg.tp_R1 * R, entry - self.cfg.tp_R2 * R
            
            print(f"         📊 가격 계산:")
            print(f"            진입가: {entry:.2f} (저점 {l:.2f} - tick {self.cfg.tick})")
            print(f"            스탑가: {stop:.2f} (최대값: {h:.2f}, OR하단+0.5ATR: {self.or_low + 0.5*float(atr):.2f})")
            print(f"            리스크: {R:.2f}")
            print(f"            TP1: {tp1:.2f} (진입가 - {self.cfg.tp_R1}R)")
            print(f"            TP2: {tp2:.2f} (진입가 - {self.cfg.tp_R2}R)")
            
            self.traded_short = True
            sigs.append({
                "stage": "ENTRY", "action": "SELL", "entry": float(entry), "stop": float(stop),
                "targets": [float(tp1), float(tp2)],
                "context": {
                    "mode": "SESSION_OR_LITE", "or_low": float(self.or_low),
                    "atr": float(atr), "vwap": float(vwap), "vwap_std": float(vwap_std),
                    "touched_buf": float(buf_short), "body_ok": body_ok, "wick_break": wick_break_short
                }
            })
        else:
            print(f"      ❌ 숏 신호 생성 조건 불만족:")
            print(f"         돌파: {break_short_ok}, 리테스트: {touched_short}, VWAP: {vwap_ok_short}, 이미거래: {self.traded_short}")

        # 최종 결과 출력
        if sigs:
            print(f"   🎉 신호 생성 완료: {len(sigs)}개")
            for i, sig in enumerate(sigs):
                print(f"      📊 신호 {i+1}: {sig['action']} @ {sig['entry']:.2f}, 스탑: {sig['stop']:.2f}")
                print(f"         목표: TP1={sig['targets'][0]:.2f}, TP2={sig['targets'][1]:.2f}")
        else:
            print(f"   ⚠️ 신호 생성 없음")

        return sigs[0] if sigs else None
