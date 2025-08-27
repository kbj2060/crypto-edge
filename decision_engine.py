
# ollama_decision_engine.py
# Copyright
# Description: Parse human-readable trading logs into features and query an Ollama model (DeepSeek-R1) to get a structured decision.
# Requirements: pip install ollama

from __future__ import annotations

import re
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ollama import chat


class DecisionEngine:
    """
    Turn your strategy logs into features and ask an Ollama LLM (DeepSeek-R1) for a
    structured decision (BUY/SELL/HOLD + entry/stop/targets).

    Example:
        engine = DecisionEngine(model='deepseek-r1:8b')
        feats, decision = engine.decide_from_log(log_text, pair="ETHUSDT", tf="3m")
        print(feats)
        print(decision)
    """

    DEFAULT_SCHEMA: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
            "entry": {"type": "number"},
            "stop": {"type": "number"},
            "targets": {"type": "array", "items": {"type": "number"}},
            "size_pct": {"type": "number"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "playbook": {"type": "string"},
            "risk": {
                "type": "object",
                "properties": {
                    "rr": {"type": "number"},
                    "atr_mult": {"type": "number"},
                    "stop_distance": {"type": "number"},
                },
            },
            "timing": {
                "type": "object",
                "properties": {
                    "urgency": {"type": "string", "enum": ["now", "wait"]},
                    "ttl_seconds": {"type": "number"},
                },
            },
            "reasons": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["action", "entry", "stop", "targets", "confidence", "reasons"],
    }

    DEFAULT_SYSTEM_PROMPT: str = (
        "너는 코인 단타(스캘핑·단타 혼합) 의사결정 엔진이다.\n"
        "출력은 반드시 JSON 한 줄만 반환한다. 설명/사고과정/문장 금지.\n\n"
        "## 입력(사용자 메시지 또는 tool 결과)에는 다음 피처가 일부/전체 포함된다:\n"
        "- pair, tf, now_ts\n"
        "- price, spread, price_precision\n"
        "- vwap, vwap_z, vwap_std\n"
        "- atr, atrp (atr/price)\n"
        "- session (ASIA/EU/NY), session_mins, time_to_next_open\n"
        "- trend (up/down/side), structure (HH/HL/LH/LL), momentum (macd, rsi, stoch 등 선택적)\n"
        "- support[], resistance[]\n"
        "- lpi (Liquidation Pressure Index: 0~∞), liq_spike_long, liq_spike_short (Z or σ 단위)\n"
        "- ob_imbalance (0~1 매수우위), bid_density, ask_density (상대 밀도), volume_1m/5m\n"
        "- funding, oi_change, basis (선물-현물 괴리, 선택적)\n\n"
        "## 플레이북·결정 규칙(간결):\n"
        "1) 기본: R≥1.5 미만이면 HOLD. stop은 최소 max(tick_size, 0.5*ATR), 권장 0.8~1.2*ATR.\n"
        "2) VWAP 페이드(A):\n"
        "   - BUY: vwap_z ≤ -1.8 AND liq_spike_long ≥ 2.0 AND (support에 근접 ≤0.3*ATR) → 반등 노림.\n"
        "   - SELL: vwap_z ≥ +1.8 AND liq_spike_short ≥ 2.0 AND (resistance 근접) → 되돌림.\n"
        "3) 브레이크아웃(B):\n"
        "   - BUY: price > 첫 저항 돌파 AND ob_imbalance ≥ 0.6 AND momentum↑.\n"
        "   - SELL: price < 첫 지지 이탈 AND ob_imbalance ≤ 0.4 AND momentum↓.\n"
        "4) 세션 컨텍스트:\n"
        "   - NY 오픈±45분 변동성↑: 목표를 2분할(보수/공격)로 권장.\n"
        "   - 세션 경계 5분 전엔 신규 진입 보수적(confidence↓).\n"
        "5) 리스크 가드레일:\n"
        "   - 최대 손실(스탑 거리) ≤ 1.3*ATR, RR<1.5면 HOLD.\n"
        "   - 스프레드 > 0.25*ATR이면 HOLD.\n"
        "   - 가격/스탑/타깃은 price_precision에 맞춰 반올림.\n"
        "6) 불확실:\n"
        "   - 주요 피처 결핍(예: vwap/atr 미존재) 또는 신호 충돌(페이드 vs 브레이크아웃) → HOLD, reasons에 결핍/충돌 사유 명시.\n\n"
        "## 출력 스키마(반드시 준수)\n"
        "{\n"
        '  "action": "BUY" | "SELL" | "HOLD",\n'
        '  "entry": number,\n'
        '  "stop": number,\n'
        '  "targets": [number, ...],\n'
        '  "size_pct": number,\n'
        '  "confidence": number,\n'
        '  "playbook": "A|B|C",\n'
        '  "risk": { "rr": number, "atr_mult": number, "stop_distance": number },\n'
        '  "timing": { "urgency": "now|wait", "ttl_seconds": number },\n'
        '  "reasons": [ "간결 근거 3~5개" ]\n'
        "}\n\n"
        "절대 지키기:\n"
        "- JSON 이외 콘텐츠 출력 금지.\n"
        "- 생각 과정 노출 금지. 내부적으로만 추론하고 최종 JSON만.\n"
    )

    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        system_prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        price_tick_default: float = 0.01,
        default_pair: str = "ETHUSDT",
        default_tf: str = "3m",
        default_session_label: str = "NY",
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.schema = schema or self.DEFAULT_SCHEMA
        self.price_tick_default = price_tick_default
        self.default_pair = default_pair
        self.default_tf = default_tf
        self.default_session_label = default_session_label

    # ----------------------
    # Helpers
    # ----------------------
    @staticmethod
    def _num(s: str) -> float:
        """Convert '$1,234.56' → 1234.56 safely."""
        return float(re.sub(r"[,$]", "", s))

    @staticmethod
    def _last_float(pattern: str, text: str) -> Optional[float]:
        m = re.findall(pattern, text, flags=re.M)
        return DecisionEngine._num(m[-1]) if m else None

    @staticmethod
    def _last_group(pattern: str, text: str) -> Optional[Tuple[str, ...]]:
        m = list(re.finditer(pattern, text, flags=re.M))
        return m[-1].groups() if m else None

    def _round_to_tick(self, x: float, tick: float) -> float:
        if tick <= 0:
            tick = self.price_tick_default
        return round(x / tick) * tick

    def _pick_session(self, text: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """
        Extract session start time and current UTC time from the log,
        compute minutes since session start, and return (session_label, minutes, now_ts).
        """
        g1 = self._last_group(r"세션 시작:\s*([0-9]{2}:[0-9]{2}:[0-9]{2})", text)
        # Accept HH:MM or HH:MM:SS after the date
        g2 = self._last_group(r"현재 시간:\s*([0-9\-]{4,}\s+[0-9:]{5,8})\s*UTC", text)
        if not (g1 and g2):
            return None, None, None
        sess_start = g1[0]  # e.g., "07:00:00"
        now_utc = g2[0]     # e.g., "2025-08-27 11:27:00" or "2025-08-27 11:27"

        # Normalize missing seconds to :00
        if len(now_utc.split()[-1]) == 5:  # HH:MM only
            now_utc = now_utc + ":00"

        try:
            today = now_utc.split()[0]
            t0 = datetime.fromisoformat(f"{today} {sess_start}")
            t1 = datetime.fromisoformat(now_utc)
            mins = int((t1 - t0).total_seconds() // 60)
            return self.default_session_label, mins, int(t1.timestamp())
        except Exception:
            return None, None, None

    # ----------------------
    # Parsing
    # ----------------------
    def parse_log_to_features(
        self,
        text: Dict[str, Any],
        pair: Optional[str] = None,
        tf: Optional[str] = None,
        price_precision: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Parse the provided log text into a features dict that the model understands.
        Unknown / missing features are omitted.
        """
        feat: Dict[str, Any] = {
            "pair": pair or self.default_pair,
            "tf": tf or self.default_tf,
            "price_precision": price_precision if price_precision is not None else self.price_tick_default,
        }
        text = str(text)
        # [FADE] 지표: VWAP=$..., VWAP_STD=$..., ATR=$...
        g = self._last_group(r"지표:\s*VWAP=\$?([\d,\.]+),\s*VWAP_STD=\$?([\d,\.]+),\s*ATR=\$?([\d,\.]+)", text)
        if g:
            feat["vwap"] = self._num(g[0])
            feat["vwap_std"] = self._num(g[1])
            feat["atr"] = self._num(g[2])

        # [FADE] 가격: 이전=..., 현재=..., 고가=..., 저가=...
        g = self._last_group(r"가격:\s*이전=([\d\.]+),\s*현재=([\d\.]+),\s*고가=([\d\.]+),\s*저가=([\d\.]+)", text)
        if g:
            feat["price_prev"] = float(g[0])
            feat["price"] = float(g[1])
            feat["high"] = float(g[2])
            feat["low"] = float(g[3])

        # ATR_1M (optional)
        v = self._last_float(r"ATR_1M=\$?([\d,\.]+)", text)
        if v is not None:
            feat["atr_1m"] = v

        # VPVR: POC/HVN/LVN
        gg = self._last_group(r"GLOBAL VPVR\]\s*POC:\s*\$?([\d,\.]+),\s*HVN:\s*([\d,\.]+),\s*LVN:\s*([\d,\.]+)", text)
        if gg:
            feat["poc"] = self._num(gg[0])
            feat["hvn"] = self._num(gg[1])
            feat["lvn"] = self._num(gg[2])
            # Add resistance from hvn/lvn, keep existing if provided elsewhere
            feat.setdefault("resistance", [feat["hvn"], feat["lvn"]])

        # Gold pocket LONG - $a~$b → support
        gg = self._last_group(r"골든 포켓 구간:\s*LONG\s*-\s*\$?([\d,\.]+)~\$?([\d,\.]+)", text)
        if gg:
            feat["support"] = [self._num(gg[0]), self._num(gg[1])]

        # FADE signal → liq_spike_(long/short), lpi
        gg_all = list(re.finditer(r"FADE\]\s*신호 생성:\s*(BUY|SELL)\s*\|\s*Z=([\d\.]+)\s*\|\s*LPI=([\d\.]+)", text))
        if gg_all:
            side, z, lpi = gg_all[-1].groups()
            feat["lpi"] = float(lpi)
            if side == "BUY":
                feat["liq_spike_long"] = float(z)
            else:
                feat["liq_spike_short"] = float(z)

        # Session info
        session, session_mins, now_ts = self._pick_session(text)
        if session:
            feat["session"] = session
        if session_mins is not None:
            feat["session_mins"] = session_mins
        if now_ts:
            feat["now_ts"] = now_ts

        # Derived
        if "atr" in feat and "price" in feat and feat["price"]:
            feat["atrp"] = feat["atr"] / feat["price"]

        # vwap_z estimate if missing (prefer your exact engine output if available)
        if "vwap" in feat and "vwap_std" in feat and "price" in feat and "vwap_z" not in feat:
            try:
                if feat["vwap_std"]:
                    feat["vwap_z"] = (feat["price"] - feat["vwap"]) / feat["vwap_std"]
            except ZeroDivisionError:
                pass

        return feat

    # ----------------------
    # Model call
    # ----------------------
    def ask_model(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send features to the Ollama model and enforce JSON schema output.
        """
        try:
            print(f"🤖 [DecisionEngine] Ollama 모델 호출 시작: {self.model}")
            print(f"   📊 입력 피처: {json.dumps(features, ensure_ascii=False, indent=2)}")
            
            resp = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": json.dumps(features, ensure_ascii=False)},
                ],
                options={
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": self.schema,
                        "strict": True,
                    }
                },
            )
            
            print(f"   📥 모델 응답: {resp.message.content}")
            
            # 응답 내용 검증
            if not resp.message.content or resp.message.content.strip() == "":
                print("❌ [DecisionEngine] 모델 응답이 비어있음")
                return self._get_default_decision("HOLD", "모델 응답 없음")
            
            # JSON 파싱 시도
            try:
                decision = json.loads(resp.message.content)
                print(f"   ✅ JSON 파싱 성공: {decision}")
                return decision
            except json.JSONDecodeError as e:
                print(f"❌ [DecisionEngine] JSON 파싱 오류: {e}")
                print(f"   📝 원본 응답: '{resp.message.content}'")
                return self._get_default_decision("HOLD", f"JSON 파싱 오류: {e}")
                
        except Exception as e:
            print(f"❌ [DecisionEngine] 모델 호출 오류: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_decision("HOLD", f"모델 호출 오류: {e}")
    
    def _get_default_decision(self, action: str, reason: str) -> Dict[str, Any]:
        """기본 의사결정 반환 (오류 발생 시)"""
        return {
            "action": action,
            "entry": 0.0,
            "stop": 0.0,
            "targets": [0.0],
            "size_pct": 0.0,
            "confidence": 0.0,
            "playbook": "ERROR",
            "risk": {"rr": 0.0, "atr_mult": 0.0, "stop_distance": 0.0},
            "timing": {"urgency": "wait", "ttl_seconds": 300},
            "reasons": [reason]
        }

    # ----------------------
    # End-to-end
    # ----------------------
    def decide_from_log(
        self,
        log_text: Dict[str, Any],
        pair: Optional[str] = None,
        tf: Optional[str] = None,
        price_precision: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Parse log → build features → ask model → post-process (rounding & RR check).
        Returns (features, decision).
        """
        feats = self.parse_log_to_features(
            log_text, pair=pair, tf=tf, price_precision=price_precision
        )
        decision = self.ask_model(feats)

        # Round to tick
        tick = feats.get("price_precision", self.price_tick_default)
        for k in ("entry", "stop"):
            if k in decision and isinstance(decision[k], (int, float)):
                decision[k] = float(self._round_to_tick(decision[k], tick))

        if "targets" in decision and isinstance(decision["targets"], list):
            decision["targets"] = [
                float(self._round_to_tick(x, tick)) for x in decision["targets"]
            ]

        # Simple RR check (optional)
        try:
            entry = decision.get("entry")
            stop = decision.get("stop")
            tgs = decision.get("targets", [])
            if isinstance(entry, (int, float)) and isinstance(stop, (int, float)) and tgs:
                best = max(tgs) if decision.get("action") == "BUY" else min(tgs)
                rr = abs((best - entry) / (entry - stop)) if entry != stop else 0.0
                decision.setdefault("risk", {}).update({"rr_checked": round(rr, 2)})
                if rr < 1.5:
                    decision["action"] = "HOLD"
                    reasons = decision.get("reasons") or []
                    decision["reasons"] = list(dict.fromkeys(reasons + ["RR<1.5 → HOLD 전환"]))
        except Exception:
            pass

        return feats, decision
