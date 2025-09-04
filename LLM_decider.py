# LLM_decider.py  — Ollama (deepseek-r1:8b) • rules-in-init • streaming • replay/feedback/adaptive context
#                    + Symbol-wise position engine (LONG/SHORT/HOLD) with auto-PnL from signal.candle_data.close
import os, json, time, asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timezone
import requests
# =========================
# Replay buffer & Calibrator
# =========================

@dataclass
class DecisionRecord:
    timestamp_utc: str            # ISO UTC
    symbol: str
    decision: str                 # LONG/SHORT/HOLD
    confidence: float
    reason: str
    features: Dict[str, Any]      # thin features from signal
    outcome_pnl: Optional[float] = None  # set later via feedback()

class ReplayBuffer:
    """Append-only JSONL logger for decisions/outcomes."""
    def __init__(self, path: str = "decisions_log.jsonl", maxlen: int = 200_000):
        self.path = path
        self.maxlen = maxlen
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def append(self, rec: DecisionRecord):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def last_n(self, n: int = 3000) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out = deque(maxlen=n)
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except:
                    pass
        return list(out)

class Calibrator:
    """
    Build small 'learned context' from recent outcomes:
      - decision_bias: LONG/SHORT bias from winrate
      - strategy_reliability: EMA winrate per strategy name
    This context is injected into the LLM prompt (adaptive without fine-tuning).
    """
    def __init__(self, ema_alpha: float = 0.12):
        self.ema_alpha = float(ema_alpha)

    def build_context(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        side_stats = self._agg_side_perf(logs)  # {"LONG": {...}, "SHORT": {...}}
        long_perf  = side_stats.get("LONG", {"winrate": 0.5, "avg_pnl": 0.0})
        short_perf = side_stats.get("SHORT", {"winrate": 0.5, "avg_pnl": 0.0})

        decision_bias = {
            "LONG":  self._score_to_bias(long_perf),
            "SHORT": self._score_to_bias(short_perf),
        }
        strat_rel = self._strategy_reliability(logs)
        return {
            "decision_bias": decision_bias,           # -0.2..+0.2
            "strategy_reliability": strat_rel,        # 0..1
            "stats_window": len(logs),
        }

    def _score_to_bias(self, perf: Dict[str, float]) -> float:
        try:
            wr = float(perf.get("winrate", 0.5))
        except Exception:
            wr = 0.5
        bias = (wr - 0.5) * 0.8            # [-0.2, +0.2] 범위 매핑
        return max(-0.2, min(0.2, bias))


    def _agg_side_perf(self, logs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        by_side = {"LONG": [], "SHORT": []}
        for r in logs:
            dec = str(r.get("decision") or "").upper()
            pnl = r.get("outcome_pnl")
            if dec in by_side and isinstance(pnl, (int, float)):
                by_side[dec].append(float(pnl))

        def perf(arr):
            if not arr:
                return {"winrate": 0.5, "avg_pnl": 0.0}
            wins = sum(1 for x in arr if x > 0)
            return {"winrate": wins / len(arr), "avg_pnl": sum(arr) / len(arr)}

        return {"LONG": perf(by_side["LONG"]), "SHORT": perf(by_side["SHORT"])}

    def _strategy_reliability(self, logs: List[Dict[str, Any]]) -> Dict[str, float]:
        stats = defaultdict(lambda: {"ema": 0.5, "n": 0})
        for r in logs:
            pnl = r.get("outcome_pnl")
            if pnl is None:
                continue
            pnl = float(pnl)
            feats = r.get("features") or {}
            strats = feats.get("strategies") or {}

            win = 1.0 if pnl > 0 else 0.0

            for name in strats.items():
                key = (str(name) if name is not None else "UNKNOWN").upper()
                prev = stats[key]["ema"]
                stats[key]["ema"] = (1 - self.ema_alpha) * prev + self.ema_alpha * win
                stats[key]["n"] += 1

        return {k: round(v["ema"], 3) for k, v in stats.items()}

# =========================
# Position state (per symbol)
# =========================

@dataclass
class PositionState:
    side: Optional[str] = None         # "LONG" / "SHORT" / None
    entry_price: Optional[float] = None
    entry_time_utc: Optional[str] = None

# =========================
# LLM Decider (Ollama only)
# =========================

class LLMDecider:
    """
    - Ollama 전용 LLM 래퍼 (deepseek-r1:8b)
    - __init__에서 rules_text를 시스템 프롬프트로 1회 주입 (이후 decide()/decide_async()에는 signal만)
    - 스트리밍(JSON 강제) → 안전 파싱
    - ReplayBuffer + Calibrator 로 'learned context' 자동 주입
    - feedback()으로 손익 기록 → 다음 호출부터 자동 보정
    - **심볼별 포지션 엔진**: LONG/SHORT/HOLD, signal.candle_data.close를 체결가로 사용해
      반대 신호 시 자동 청산 & 손익 기록 → 반대 포지션으로 전환
    """

    def __init__(
        self,
        model: str = "deepseek-r1:14b",
        rules_text: str = "",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_retries: int = 2,
        use_json_format: bool = True,  # Ollama 'format': 'json'
        decisions_log_path: str = "decisions_log.jsonl",
        ema_alpha: float = 0.12,
    ):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.use_json_format = bool(use_json_format)

        # --- System prompt (Rules-in-init)
        self._json_schema_hint = (
            '{"decision":"LONG|SHORT|HOLD","confidence":0.0,"reason":"<=120 chars"}'
        )
        default_rules = """
            === Creative, signal-first rules (exploratory mode) ===

            목적
            - 모델에게 더 많은 자유를 주어, 단순 룰 기반을 넘는 창의적·경험적 판단을 탐색하게 함.
            - 단, 모든 판단은 오직 호출 시 전달된 signal (및 learned_context) 범위 내에서만 이루어져야 함.
            - 외부 웹/데이터/기억을 참조하거나 사실을 발명하지 말 것.

            입력
            - Signal JSON (required): net_score, recommended_trade_scale, raw/strategies (전략별 action/score/conf_factor/weight/entry/stop),
            candle_data {open,high,low,close,volume,quote_volume}, 기타 메타.
            - Learned context (optional): decision_bias, strategy_reliability, past_outcomes 등.

            자유도(핵심)
            - **주요 원칙**: signal-first — 우선 signal을 사용하되, 필요하면 candle_data·strategies에서 파생된 추가 특성(모멘텀, 페이스, 패턴, 슬로프, wick-ratio, volume-spike 등)을 스스로 계산해 판단할 것.
            - 모델은 내부적으로 새로운 유의미한 피처(derived_features)를 만들고 이를 근거로 사고해도 된다. 다만 그 근거는 출력의 `inference_basis`에 간단히 기술해야 함.
            - 외부 사실(뉴스 텍스트, 실시간 사건 등)은 절대 사용 금지. 숫자/가격은 반드시 입력값만 사용.

            결정 규범(공격적·탐색적)
            - 결정은 LONG/SHORT/HOLD 중 하나.
            - **더 공격적**: 작은 신호라도 (a) 여러 전략이 같은 방향으로 연쇄 동의하거나, (b) 파생 피처(예: 강한 모멘텀+볼륨)와 결합되면 LONG/SHORT를 선택할 수 있음.
            - 단, 불확실성(입력 결손, 강한 상반 신호, candle_data 없음) 시에는 HOLD를 선택하라.
            - 권장 스케일은 LLM의 판단으로 계산하되, 반드시 0..1 범위를 지켜라(캡 0.5 권장). 위험 이벤트(learned_context의 'market_event' 등)가 감지되면 스케일을 보수적으로 줄여라.

            투명성(출력 보조)
            - 출력 JSON 에 `inference_basis` (선택적 배열)를 포함시켜라 — 예: ["VOL_SPIKE_3M score=1.0", "close>open strong_momentum"].
            - `reason`은 120자 이내의 간결한 핵심 근거(전략 요약 혹은 파생 피처 언급).

            출력 포맷(절대 준수)
            - 반드시 단 하나의 JSON 객체만 반환. **추가 텍스트/주석 금지.**
            - 기본 필드(필수):
            {
                "decision": "LONG" | "SHORT" | "HOLD",
                "confidence": float,                # 0.0 ~ 1.0
                "recommended_trade_scale": float,   # 0..1 (없으면 0)
                "reason": "짧고 핵심(<=120 chars)"
            }
            - 선택 필드(권장):
            - "inference_basis": ["...","..."]  # 모델이 창의적으로 사용한 근거(최대 6개)
            - "derived_features": {"momentum": 0.12, "volume_spike": true}  # 내부 계산 결과(선택)
            - 예외: JSON이 파싱 불가하면 반드시 빈결과 대신 {"decision":"HOLD","confidence":0.0,"reason":"invalid_json"} 를 반환.

            안전·운영 메모
            - 모형이 창의적으로 판단할수록 **실험 로그**를 남겨라(입력, 출력, inference_basis). 모델의 판단 패턴을 모니터링하면서 규칙을 튜닝할 것.
            - 이 모드는 탐색적(exploratory)이다. 실매매에 바로 투입하기 전 충분한 백테스트/페이퍼트레이드 권장.
            """

        rules_text = (rules_text.strip() or default_rules)
        self._system_prompt = (
            "You are a crypto short-term trading arbiter for a trader who holds longer than a scalper "
            "(minutes to hours). Follow these rules strictly:\n" + rules_text +
            "\nReturn ONLY one JSON object exactly like: " + self._json_schema_hint
        )

        # --- Learning utilities
        self._replay = ReplayBuffer(path=os.getenv("DECISIONS_LOG", decisions_log_path))
        self._calibrator = Calibrator(ema_alpha=ema_alpha)
        self._learned_context: Dict[str, Any] = {}

        # --- Position book (per symbol)
        self._positions: Dict[str, PositionState] = {}

    # --------------------
    # Public API (sync)
    # --------------------
    def decide(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """동기 판단(내부적으로 async 실행). 자동 포지션 엔진/피드백 포함."""
        return asyncio.run(self.decide_async(signal))

    # --------------------
    # Public API (async)
    # --------------------
    async def decide_async(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        비동기 판단:
            1) rules(system) + signal(user) + learned_context 전달 → LLM 응답 파싱
            2) signal.candle_data.close를 체결가로 사용하여, 심볼별 포지션 자동 갱신
                - 반대 신호 → 기존 포지션 청산 & PnL 기록(feedback) → 반대 포지션으로 전환
                - HOLD or 동일 방향 → 상태 유지
            3) 결정 로그 기록
        반환: {"decision","confidence","reason","position_events":[(action,pnl)],"position_state":{...}}
        """
        serializable_signal = self._make_serializable(signal)
        self._update_learned_context()
        context_json = json.dumps(self._learned_context, ensure_ascii=False, separators=(',',':'))

        user_prompt = (
            "Signal JSON:\n" +
            json.dumps(serializable_signal, ensure_ascii=False, separators=(',',':')) +
            "\nLearned context (from past outcomes):\n" + context_json +
            "\nRespond with ONLY one JSON object like: " + self._json_schema_hint
        )

        raw = await self._call_ollama_async(self._system_prompt, user_prompt)
        out = self._parse_response_safe(raw)

        # ---- Log decision immediately (without outcome)
        symbol = str(signal.get("symbol",""))
        rec = DecisionRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            decision=out["decision"],
            confidence=float(out["confidence"]),
            reason=str(out["reason"]),
            features=self._thin_features(serializable_signal),
        )
        self._replay.append(rec)

        # ---- Auto position engine using candle_data.close
        exec_price = self._extract_exec_price(serializable_signal)
        pos_events = self._auto_position_update(symbol, out["decision"], exec_price)

        # ---- Return enriched output
        pos_state = asdict(self._positions.get(symbol, PositionState()))
        enriched = {**out, "position_events": pos_events, "position_state": pos_state}
        return enriched

    # --------------------
    # Runtime rule update
    # --------------------
    def set_rules(self, rules_text: str):
        rules_text = (rules_text or "").strip()
        if not rules_text:
            return
        self._system_prompt = (
            "You are a crypto short-term trading arbiter for a trader who holds longer than a scalper "
            "(minutes to hours). Follow these rules strictly:\n" + rules_text +
            "\nReturn ONLY one JSON object exactly like: " + self._json_schema_hint
        )

    # --------------------
    # Feedback API (manual)
    # --------------------
    def feedback(self, symbol: str, closed_pnl: float, since_utc_iso: Optional[str] = None):
        """
        포지션 종료 시 실제 PnL을 가장 최근 미평가 레코드(outcome_pnl=None)에 채워넣음.
        since_utc_iso가 주어지면 그 이후의 기록만 대상으로 함.
        """
        path = self._replay.path
        if not os.path.exists(path): 
            return

        items: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    items.append(json.loads(line))
                except:
                    continue

        since_dt = None
        if since_utc_iso:
            try:
                since_dt = datetime.fromisoformat(since_utc_iso.replace("Z","+00:00"))
            except:
                since_dt = None

        for i in range(len(items)-1, -1, -1):
            it = items[i]
            if it.get("symbol") != symbol:
                continue
            if since_dt:
                t = it.get("timestamp_utc")
                if t:
                    try:
                        tdt = datetime.fromisoformat(t.replace("Z","+00:00"))
                        if tdt < since_dt:
                            break
                    except:
                        pass
            if it.get("outcome_pnl") is None:
                it["outcome_pnl"] = float(closed_pnl)
                items[i] = it
                break

        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # --------------------
    # Internal: position engine
    # --------------------
    def _auto_position_update(self, symbol: str, decision: str, price: Optional[float]) -> List[List[Any]]:
        """
        LONG/SHORT/HOLD 전용 포지션 관리.
        - price는 signal.candle_data.close (체결가로 사용)
        - 반대 신호 시: 기존 포지션 청산 → feedback에 PnL 기록 → 반대 포지션 신규 진입
        반환 예: [["LONG→OPEN",0.0], ["LONG→CLOSE",12.3], ["SHORT→OPEN",0.0]]
        """
        events: List[List[Any]] = []
        if price is None or not isinstance(price, (int, float)):
            return events  # 가격 없으면 포지션 작업 생략

        st = self._positions.get(symbol, PositionState())

        act = (decision or "").upper()
        if act not in ("LONG","SHORT","HOLD"):
            return events

        # LONG → SHORT 전환
        if st.side == "LONG" and act == "SHORT":
            pnl = float(price) - float(st.entry_price)
            self._log_feedback(symbol, pnl)
            events.append(["LONG→CLOSE", pnl])
            st.side = "SHORT"
            st.entry_price = float(price)
            st.entry_time_utc = datetime.now(timezone.utc).isoformat()
            events.append(["SHORT→OPEN", 0.0])

        # SHORT → LONG 전환
        elif st.side == "SHORT" and act == "LONG":
            pnl = float(st.entry_price) - float(price)
            self._log_feedback(symbol, pnl)
            events.append(["SHORT→CLOSE", pnl])
            st.side = "LONG"
            st.entry_price = float(price)
            st.entry_time_utc = datetime.now(timezone.utc).isoformat()
            events.append(["LONG→OPEN", 0.0])

        # 포지션 없음 → 신규 진입
        elif st.side is None:
            if act == "LONG":
                st.side = "LONG"
                st.entry_price = float(price)
                st.entry_time_utc = datetime.now(timezone.utc).isoformat()
                events.append(["LONG→OPEN", 0.0])
            elif act == "SHORT":
                st.side = "SHORT"
                st.entry_price = float(price)
                st.entry_time_utc = datetime.now(timezone.utc).isoformat()
                events.append(["SHORT→OPEN", 0.0])

        # HOLD or 동일 방향 → 아무 것도 안 함

        self._positions[symbol] = st
        return events

    def _log_feedback(self, symbol: str, pnl: float):
        """포지션 청산 시 ReplayBuffer에 outcome_pnl 기록"""
        self.feedback(symbol=symbol, closed_pnl=pnl)

    # --------------------
    # Internal helpers
    # --------------------
    def _thin_features(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        # 필요한 핵심만 축약 저장 → 프롬프트 길이 관리 & 사후 분석에 충분
        return {
            "symbol": signal.get("symbol"),
            "net_score": signal.get("net_score") or signal.get("net") or 0.0,
            "recommended_trade_scale": signal.get("recommended_trade_scale") or 0.0,
            "strategies": signal.get("raw") or signal.get("strategies") or [],
            "candle_close": self._extract_exec_price(signal),  # 기록용
        }

    def _update_learned_context(self):
        logs = self._replay.last_n(3000)
        self._learned_context = self._calibrator.build_context(logs)

    def _extract_exec_price(self, signal_like: Dict[str, Any]) -> Optional[float]:
        """signal['candle_data']['close']를 체결가로 사용. 없으면 None"""
        cd = (signal_like or {}).get("candle_data") or {}
        px = cd.get("close")
        try:
            return float(px) if px is not None else None
        except Exception:
            return None

    def _make_serializable(self, obj: Any) -> Any:
        from datetime import datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(x) for x in obj]
        return obj

    # --------------------
    # Ollama calls (async)
    # --------------------
    async def _call_ollama_async(self, system: str, user: str) -> str:
        """비동기 Ollama 호출 with retry logic (streaming)"""
        last_err = None
        for _ in range(self.max_retries + 1):
            try:
                return await self._ollama_stream_async(system, user)
            except Exception as e:
                last_err = e
                await asyncio.sleep(0.5)
        return json.dumps({"decision":"HOLD","confidence":0.0,"reason":f"llm_error:{last_err}"})

    async def _ollama_stream_async(self, system: str, user: str) -> str:
        """비동기 Ollama 스트리밍 호출 (requests를 asyncio.to_thread로 래핑)"""
        def _make_request():
            url = f"{self.api_base}/api/generate"
            payload = {
                "model": self.model,
                "prompt": user,
                "system": system,
                "options": {"temperature": self.temperature},
                "stream": True,
            }
            if self.use_json_format:
                payload["format"] = "json"

            chunks: List[str] = []
            with requests.post(url, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        txt = data.get("response", "")
                        if txt:
                            chunks.append(txt)
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            return "".join(chunks)
        
        return await asyncio.to_thread(_make_request)

    # --------------------
    # Safe parser
    # --------------------
    def _parse_response_safe(self, raw_text: str) -> Dict[str, Any]:
        try:
            s = raw_text.find("{"); e = raw_text.rfind("}")
            if s == -1 or e == -1 or e <= s:
                raise ValueError("no_json")
            obj = json.loads(raw_text[s:e+1])
        except Exception:
            return {"decision":"HOLD","confidence":0.0,"reason":"invalid_json"}

        # 모델이 가끔 BUY/SELL을 낼 경우 대비해 매핑
        dec = str(obj.get("decision","HOLD")).upper().strip()
        dec = {"BUY":"LONG","SELL":"SHORT"}.get(dec, dec)
        if dec not in ("LONG","SHORT","HOLD"):
            dec = "HOLD"

        try:
            conf = float(obj.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        reason = str(obj.get("reason",""))[:120]
        return {"decision": dec, "confidence": conf, "reason": reason}
