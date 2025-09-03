# LLM_decider.py  (Ollama only, rules-in-init, streaming supported)
import os, json, time
from typing import Any, Dict, Optional, List
import requests

class LLMDecider:
    """
    - Ollama 전용 LLM 심사 래퍼
    - __init__에서 rules_text를 시스템 프롬프트로 주입 (이후 decide()에선 signal만)
    - stream=True일 때 청크를 모아 최종 JSON 파싱
    - JSON 강제: Ollama 'format': 'json' 사용(가능한 모델에서)
    """

    def __init__(
        self,
        model: str = "deepseek-r1:8b",
        rules_text: str = "",
        api_base: str = "http://localhost:11434",
        temperature: float = 0.0,
        timeout_s: int = 25,
        max_retries: int = 2,
        stream: bool = True,
        use_json_format: bool = True,   # Ollama 'format': 'json' 사용 여부
    ):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.temperature = float(temperature)
        self.timeout_s = int(timeout_s)
        self.max_retries = int(max_retries)
        self.stream = bool(stream)
        self.use_json_format = bool(use_json_format)

        # 시스템 규칙 프롬프트 (초기 1회 주입)
        self._json_schema_hint = (
            '{"decision":"BUY|SELL|HOLD","confidence":0.0,"reason":"<=120 chars"}'
        )
        default_rules = (
            "- You are not a high-frequency scalper. Prefer fewer, higher-conviction trades (minutes→hours).\n"
            "- Approve BUY/SELL only if:\n"
            "  1) Confluence: ≥2 same-direction signals among {SESSION, VWAP, VPVR, EMA_TREND_15M}.\n"
            "  2) Trend gate: If EMA_TREND_15M exists, avoid going against it unless net_score ≥ 0.65.\n"
            "  3) Quality: net_score ≥ 0.35 and recommended_trade_scale ≥ 0.08; otherwise HOLD.\n"
            "  4) Prefer R:R ≥ 1.5 if entry/stop/tp exist; else conservative.\n"
            "  5) If strong opposite (score≥0.60) from VWAP/SESSION/VPVR, prefer HOLD unless net advantage ≥ 0.15.\n"
            "  6) Near high-impact events/extreme vol spikes, require ≥3 confirmations.\n"
            "- Discount microstructure-only signals (ORDERFLOW_CVD, VOL_SPIKE_3M) unless confirming higher-timeframe/context.\n"
            "- If info is incomplete or confidence is LOW, choose HOLD.\n"
        )
        rules_text = rules_text.strip() or default_rules
        self._system_prompt = (
            "You are a crypto short-term trading arbiter for a trader who holds longer than a scalper (minutes to hours). "
            "Follow these rules strictly:\n" + rules_text +
            "\nReturn ONLY one JSON object exactly like: " + self._json_schema_hint
        )

    # --- 외부 API ---
    def decide(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        signal만 넘기면 됨. (rules는 __init__에서 이미 세팅됨)
        반환: {"decision": "BUY|SELL|HOLD", "confidence": float, "reason": str}
        실패 시 보수적으로 HOLD.
        """
        serializable_signal = self._make_serializable(signal)
        user_prompt = (
            "Signal JSON:\n" +
            json.dumps(serializable_signal, ensure_ascii=False, separators=(',',':')) +
            "\nRespond with ONLY one JSON object like: " + self._json_schema_hint
        )
        raw = self._call_ollama(self._system_prompt, user_prompt)
        return self._parse_response_safe(raw)

    def set_rules(self, rules_text: str):
        """운영 중 규칙 갱신이 필요할 때 호출."""
        rules_text = (rules_text or "").strip()
        if not rules_text:
            return
        self._system_prompt = (
            "You are a crypto short-term trading arbiter for a trader who holds longer than a scalper (minutes to hours). "
            "Follow these rules strictly:\n" + rules_text +
            "\nReturn ONLY one JSON object exactly like: " + self._json_schema_hint
        )

    # --- 내부: 직렬화 도우미 ---
    def _make_serializable(self, obj: Any) -> Any:
        from datetime import datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(x) for x in obj]
        return obj

    # --- 내부: Ollama 호출 ---
    def _call_ollama(self, system: str, user: str) -> str:
        last_err = None
        for _ in range(self.max_retries + 1):
            try:
                if self.stream:
                    return self._ollama_stream(system, user)
                else:
                    return self._ollama_once(system, user)
            except Exception as e:
                last_err = e
                time.sleep(0.5)
        return json.dumps({"decision":"HOLD","confidence":0.0,"reason":f"llm_error:{last_err}"})

    def _ollama_stream(self, system: str, user: str) -> str:
        url = f"{self.api_base}/api/generate"
        payload = {
            "model": self.model,
            "prompt": user,
            "system": system,
            "options": {"temperature": self.temperature},
            "stream": True,
        }
        # JSON 강제 가능 모델이면 'format': 'json'
        if self.use_json_format:
            payload["format"] = "json"

        chunks: List[str] = []
        with requests.post(url, json=payload, stream=True, timeout=self.timeout_s) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                txt = data.get("response", "")
                if txt:
                    chunks.append(txt)
                if data.get("done", False):
                    break
        return "".join(chunks)

    def _ollama_once(self, system: str, user: str) -> str:
        url = f"{self.api_base}/api/generate"
        payload = {
            "model": self.model,
            "prompt": user,
            "system": system,
            "options": {"temperature": self.temperature},
            "stream": False,
        }
        if self.use_json_format:
            payload["format"] = "json"
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")

    # --- 내부: 안전 파서 ---
    def _parse_response_safe(self, raw_text: str) -> Dict[str, Any]:
        try:
            s = raw_text.find("{"); e = raw_text.rfind("}")
            if s == -1 or e == -1 or e <= s:
                raise ValueError("no_json")
            obj = json.loads(raw_text[s:e+1])
        except Exception:
            return {"decision":"HOLD","confidence":0.0,"reason":"invalid_json"}

        decision = str(obj.get("decision","HOLD")).upper().strip()
        if decision not in ("BUY","SELL","HOLD"):
            decision = "HOLD"

        try:
            conf = float(obj.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        reason = str(obj.get("reason",""))[:120]
        return {"decision": decision, "confidence": conf, "reason": reason}
