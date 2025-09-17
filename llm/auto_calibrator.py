import os, json
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any

# --- AutoCalibrator: EMA-based lightweight calibrator persisted to disk ---
class AutoCalibrator:
    def __init__(self, path="llm_calibration_state.json", ema_alpha=0.06, min_samples=30):
        self.path = path
        self.ema_alpha = float(ema_alpha)
        self.min_samples = int(min_samples)
        self.strategy_perf = defaultdict(lambda: {"ema": 0.5, "n": 0})
        self.decision_bias = {"long_minus_short": 0.0, "n": 0}
        self.total = 0
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k,v in data.get("strategy_perf", {}).items():
                        self.strategy_perf[k] = v
                    self.decision_bias.update(data.get("decision_bias", {}))
                    self.total = int(data.get("total", 0))
            except Exception:
                # ignore load errors, start fresh
                pass

    def _save(self):
        data = {
            "strategy_perf": dict(self.strategy_perf),
            "decision_bias": self.decision_bias,
            "total": self.total
        }
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=lambda o: o)
        except Exception:
            pass

    def update_from_feedback(self, sanitized_signal: dict, llm_decision: dict, closed_pnl: float):
        try:
            win = 1.0 if float(closed_pnl) > 0 else 0.0
        except Exception:
            return
        strategies = sanitized_signal.get("features", {}).get("strategies", {})
        for name, info in strategies.items():
            key = str(name).upper()
            prev = float(self.strategy_perf.get(key, {}).get("ema", 0.5))
            new = (1 - self.ema_alpha) * prev + self.ema_alpha * win
            self.strategy_perf[key] = {"ema": new, "n": self.strategy_perf.get(key, {}).get("n", 0) + 1}
        dec = str(llm_decision.get("decision", "HOLD")).upper()
        val = 0.0
        if dec == "LONG":
            val = 1.0
        elif dec == "SHORT":
            val = -1.0
        prev_bias = float(self.decision_bias.get("long_minus_short", 0.0))
        self.decision_bias["long_minus_short"] = (1 - self.ema_alpha) * prev_bias + self.ema_alpha * (val * win)
        self.decision_bias["n"] = self.decision_bias.get("n", 0) + 1
        self.total += 1
        self._save()

    def get_strategy_reliability(self):
        return {k: round(v["ema"], 3) for k, v in self.strategy_perf.items()}

    def get_decision_bias(self):
        return max(-1.0, min(1.0, float(self.decision_bias.get("long_minus_short", 0.0))))

    def is_ready(self):
        return self.total >= self.min_samples

# --- sanitize helper (based on user's provided sample) ---
def sanitize_signal_for_llm(signal: Dict[str, Any]) -> Dict[str, Any]:
    import copy
    s = copy.deepcopy(signal)
    s.pop("action", None)
    raw = s.get("raw") or {}
    clean_strats = {}
    top_scores = []
    agree_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for name, info in (raw.items() if isinstance(raw, dict) else []):
        if not isinstance(info, dict):
            continue
        key_name = str(name).upper()
        action = info.get("action")
        if isinstance(action, str):
            a = action.strip().upper()
            if a in ("BUY", "LONG"):
                agree_counts["BUY"] += 1
            elif a in ("SELL", "SHORT"):
                agree_counts["SELL"] += 1
            else:
                agree_counts["HOLD"] += 1
        try:
            score = float(info.get("score", 0.0))
        except Exception:
            score = 0.0
        try:
            weight = float(info.get("weight", 0.0))
        except Exception:
            weight = 0.0
        conf_factor = info.get("conf_factor", None)
        try:
            if conf_factor is not None:
                conf_factor = float(conf_factor)
        except Exception:
            conf_factor = None
        info_copy = {}
        for k, v in info.items():
            if k == "action":
                continue
            if isinstance(v, datetime):
                info_copy[k] = v.isoformat()
            else:
                info_copy[k] = v
        info_copy["score"] = score
        info_copy["weight"] = weight
        if conf_factor is not None:
            info_copy["conf_factor"] = conf_factor
        clean_strats[key_name] = info_copy
        top_scores.append({"name": key_name, "score": score, "weight": weight})
    top_scores.sort(key=lambda x: x["score"], reverse=True)
    candle = s.get("candle_data") or {}
    candle_features = {}
    try:
        o = float(candle.get("open"))
        h = float(candle.get("high"))
        l = float(candle.get("low"))
        c = float(candle.get("close"))
        rng = max(h - l, 1e-9)
        body = c - o
        body_pct = body / rng if rng else 0.0
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        wick_total = upper_wick + lower_wick
        wick_ratio = wick_total / rng if rng else 0.0
        close_pos = (c - l) / rng if rng else 0.0
        candle_features = {
            "open": o, "high": h, "low": l, "close": c,
            "range": rng,
            "body": body,
            "body_pct_of_range": body_pct,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "wick_ratio": wick_ratio,
            "close_pos": close_pos,
        }
        candle_features["momentum"] = body
    except Exception:
        pass
    sizing = s.get("sizing", {})
    meta = s.get("meta", {})
    summary = {
        "net_score": float(s.get("net_score", 0.0) or 0.0),
        "recommended_trade_scale": float(s.get("recommended_trade_scale", 0.0) or 0.0),
        "agree_counts": agree_counts,
        "top_scores": top_scores[:5],
        "used_weight_sum": float(meta.get("used_weight_sum", 0.0) or 0.0),
        "sizing": sizing,
    }
    features = {
        "strategies": clean_strats,
        "summary": summary,
        "candle_features": candle_features,
        "meta": meta,
    }
    sanitized = {
        "features": features,
        "net_score": summary["net_score"],
        "recommended_trade_scale": summary["recommended_trade_scale"],
        "candle_data": candle,
        "original_meta": meta,
    }
    if "reason" in s:
        sanitized["prior_reason"] = s.get("reason")
    return sanitized

# --- runtime monkeypatch helper ---
def _attach_calibrator_and_methods(module):
    """
    Attach calibrator and helper methods to module.LLMDecider at runtime.
    This wrapper will:
        - add self.calibrator, self._last_sanitized_signal, self._last_llm_output
        - wrap decide_async to call sanitize + apply calibration to LLM output
        - add feedback() method to update calibrator from closed pnl
    """
    if not hasattr(module, "LLMDecider"):
        # nothing to do
        return

    L = module.LLMDecider

    # only attach once
    if getattr(L, "_cal_patch_applied", False):
        return
    L._cal_patch_applied = True

    # store original decide_async and __init__ if present
    orig_decide = getattr(L, "decide_async", None)
    orig_init = getattr(L, "__init__", None)

    # new __init__ wrapper
    def __init_wrapper(self, *args, **kwargs):
        # call original constructor
        if orig_init is not None:
            orig_init(self, *args, **kwargs)
        # attach calibrator and last-caches
        self.calib_path = getattr(self, "calib_path", os.getenv("LLM_CALIB_PATH", "llm_calibration_state.json"))
        self.calibrator = AutoCalibrator(path=self.calib_path,
                                            ema_alpha=getattr(self, "ema_alpha", 0.06),
                                            min_samples=getattr(self, "min_samples", 30))
        self._last_sanitized_signal = None
        self._last_llm_output = None

    L.__init__ = __init_wrapper

    # add _apply_calibration method
    def _apply_calibration(self, llm_out: Dict[str, Any], sanitized_signal: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(llm_out, dict):
            return llm_out
        if not hasattr(self, "calibrator") or self.calibrator is None:
            return llm_out
        bias = self.calibrator.get_decision_bias()
        dec = str(llm_out.get("decision", "")).upper()
        if self.calibrator.is_ready():
            conf = float(llm_out.get("confidence", 0.0))
            if dec in ("LONG", "BUY"):
                conf = min(1.0, conf + max(0.0, bias * 0.05))
            elif dec in ("SHORT", "SELL"):
                conf = min(1.0, conf + max(0.0, -bias * 0.05))
            llm_out["confidence"] = round(conf, 3)
            if abs(bias) > 0.6:
                if (bias > 0 and dec in ("SHORT","SELL") and conf < 0.6) or \
                   (bias < 0 and dec in ("LONG","BUY") and conf < 0.6):
                    llm_out["decision"] = "HOLD"
                    llm_out["reason"] = "bias_veto"
        # cache last sanitized signal + llm out for later feedback
        try:
            self._last_sanitized_signal = sanitized_signal
            self._last_llm_output = llm_out
        except Exception:
            pass
        return llm_out

    L._apply_calibration = _apply_calibration
    
    # add feedback method: wrap existing feedback if present to preserve original behavior
    orig_feedback = getattr(L, "feedback", None)

        
    def feedback_wrapper(self, *args, **kwargs):
        """Compatibility wrapper.
        Accepts both original signature: feedback(symbol, closed_pnl, since_utc_iso=None)
        and the calibrator-oriented form: feedback(sanitized_signal=..., llm_out=..., closed_pnl=...).
        Always calls the original feedback (if present) with the original args/kwargs,
        then updates AutoCalibrator using cached last sanitized/llm_out and the detected closed_pnl.
        """
        # 1) Detect closed_pnl from kwargs or positional
        closed_pnl = None
        if 'closed_pnl' in kwargs:
            closed_pnl = kwargs.get('closed_pnl')
        else:
            # positional: original form -> (symbol, closed_pnl, [since])
            if len(args) >= 2:
                closed_pnl = args[1]
            elif len(args) == 1 and isinstance(args[0], (int,float)):
                closed_pnl = args[0]
        # 2) Extract optional sanitized/llm_out if provided explicitly
        sanitized_signal = kwargs.get('sanitized_signal', None)
        llm_out = kwargs.get('llm_out', None)

        # 3) Call original feedback (preserve exact behavior)
        called = False
        if orig_feedback is not None:
            try:
                orig_feedback(self, *args, **kwargs)
                called = True
            except Exception:
                # do not fail; proceed to calibrator update
                called = False

        # 4) Update calibrator (use cached last values if explicit ones not provided)
        try:
            sig = sanitized_signal or getattr(self, '_last_sanitized_signal', None)
            out = llm_out or getattr(self, '_last_llm_output', None)
            if sig is not None and out is not None and closed_pnl is not None and hasattr(self, 'calibrator'):
                self.calibrator.update_from_feedback(sig, out, float(closed_pnl))
        except Exception:
            pass

        return called

    L.feedback = feedback_wrapper

    # wrap decide_async to sanitize input and apply calibration to the result
    if orig_decide is not None:
        import inspect
        if inspect.iscoroutinefunction(orig_decide):
            async def decide_async_wrapper(self, signal, *args, **kwargs):
                sanitized = sanitize_signal_for_llm(signal)
                # call original decide_async
                result = await orig_decide(self, signal, *args, **kwargs)
                try:
                    # if result is not dict, leave as-is
                    if isinstance(result, dict):
                        result = self._apply_calibration(result, sanitized)
                except Exception:
                    # swallow and return original result on patch error
                    pass
                return result
            L.decide_async = decide_async_wrapper
        else:
            # sync original decide_async (unlikely) - wrap accordingly
            def decide_sync_wrapper(self, signal, *args, **kwargs):
                sanitized = sanitize_signal_for_llm(signal)
                result = orig_decide(self, signal, *args, **kwargs)
                try:
                    if isinstance(result, dict):
                        result = self._apply_calibration(result, sanitized)
                except Exception:
                    pass
                return result
            L.decide_async = decide_sync_wrapper

# try to attach when this patch is imported/executed
try:
    import LLM_decider as _mod  # if module imported by name
    _attach_calibrator_and_methods(_mod)
except Exception:
    # if imported with another name or not imported yet, attachment will be done by caller
    pass