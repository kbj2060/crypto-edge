"""
fade_liquidation_resolver.py

Resolve conflicts between Fade (reversion/fade) and Liquidation (sweep/momentum) signals.
This module exposes `resolve_fade_liquidation_conflict(analysis, vpvr=None, vwap=None, atr=None, params=None)`
which returns a dict containing a recommended decision and metadata for orchestration.

It is intentionally defensive about field names and supports common keys observed in user's pipeline:
- strategy keys such as "liquidation_strategies_lite", "fade_reentry_strategy", "fade_strategy", "liquidation", "fade"
- per-signal fields: score, confidence, action, context (may include z, z_score, lpi, total_volume, total_amount, last_update, last_vpvr_update)
- global vpvr/vwap/atr can be provided or read from analysis dict if present.

Example usage:
    from fade_liquidation_resolver import resolve_fade_liquidation_conflict
    res = resolve_fade_liquidation_conflict(analysis_dict,
                                            vpvr=analysis_dict.get('vpvr'),
                                            vwap=analysis_dict.get('vwap', {}).get('vwap'),
                                            atr=analysis_dict.get('atr', {}).get('atr'))
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import math


def _conf_bonus(conf: Optional[str]) -> float:
    if not conf:
        return 0.0
    c = str(conf).upper()
    if c == "HIGH":
        return 0.18
    if c == "MEDIUM":
        return 0.08
    return 0.0


def _age_bonus(last_update_iso: Optional[str]) -> float:
    if not last_update_iso:
        return 0.0
    try:
        # normalize iso strings that may lack timezone
        iso = str(last_update_iso)
        if iso.endswith("Z"):
            dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(iso)
        secs = (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()
        if secs < 60:
            return 0.10
        if secs < 180:
            return 0.06
        if secs < 600:
            return 0.02
        return 0.0
    except Exception:
        return 0.0


def _extra_context_bonus(sig: Dict[str, Any]) -> float:
    b = 0.0
    ctx = sig.get("context") or {}
    # z-score variants
    z = ctx.get("z") or ctx.get("z_score") or ctx.get("Z") or ctx.get("zScore")
    try:
        if z is not None:
            zf = float(z)
            if abs(zf) >= 3.0:
                b += 0.20
            elif abs(zf) >= 2.0:
                b += 0.12
            elif abs(zf) >= 1.5:
                b += 0.06
    except Exception:
        pass
    # lpi (liquidation pressure index) proximity
    lpi = ctx.get("lpi")
    try:
        if lpi is not None:
            lf = float(lpi)
            b += min(0.08, abs(lf) * 0.06)
    except Exception:
        pass
    # notional/volume heuristics
    tot = ctx.get("total_amount") or ctx.get("total_volume") or ctx.get("total_notional") or ctx.get("total_amount_usd")
    try:
        if tot is not None:
            tf = float(tot)
            if tf > 5e6:
                b += 0.08
            elif tf > 5e5:
                b += 0.04
    except Exception:
        pass
    return min(0.45, b)  # cap


def _gather_candidate_signals(analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidate_keys = [
        "liquidation_strategies_lite", "fade_reentry_strategy",
        "fade_strategy", "liquidation", "fade", "fade_reentry"
    ]
    sigs = []
    for k in candidate_keys:
        s = analysis.get(k)
        if s and isinstance(s, dict) and s.get("action"):
            copy = dict(s)
            copy["_source_key"] = k
            sigs.append(copy)
    # Additionally, if multiple signals are nested in a list under a key, try to pick them
    # e.g., analysis.get('liquidation_signals') might be a list
    for k, v in analysis.items():
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and item.get("action") and item.get("score") is not None:
                    item_copy = dict(item)
                    item_copy["_source_key"] = k
                    sigs.append(item_copy)
    return sigs


def resolve_fade_liquidation_conflict(analysis: Dict[str, Any],
                                      vpvr: Optional[Dict[str, Any]] = None,
                                      vwap: Optional[float] = None,
                                      atr: Optional[float] = None,
                                      params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Resolve conflict between fade vs liquidation signals.
    Returns a dict:
      { decision: 'BUY'|'SELL'|'NO_TRADE'|'SMALL_BUY'|'SMALL_SELL'|'HEDGE',
        reason: str,
        chosen_signal: dict or None,
        size_mult: float,
        meta: { buy_score, sell_score, entries... } }
    """
    cfg = {
        "min_diff_to_take": 0.15,
        "small_trade_mult": 0.45,
        "glue_threshold": 0.08,
        "prefer_momentum_atr": 1.5,  # domain-specific; if atr > this favor momentum
    }
    if params:
        cfg.update(params)

    sigs = _gather_candidate_signals(analysis)
    if not sigs:
        return {"decision": "NO_TRADE", "reason": "no fade/liquidation signals", "chosen_signal": None, "size_mult": 0.0, "meta": {}}

    entries = []
    for s in sigs:
        base = 0.0
        try:
            base = float(s.get("score") or s.get("strength") or 0.0)
        except Exception:
            base = 0.0
        conf = s.get("confidence") or s.get("conf") or ""
        b_conf = _conf_bonus(conf)
        # find last_update in multiple possible places
        last_update = s.get("last_update") or s.get("updated_at") or s.get("timestamp") or \
                      (s.get("context") or {}).get("last_vpvr_update") or (s.get("context") or {}).get("last_update")
        b_age = _age_bonus(last_update)
        b_extra = _extra_context_bonus(s)
        total = base + b_conf + b_age + b_extra
        total = float(total)
        entries.append({
            "sig": s,
            "base": base,
            "conf_bonus": b_conf,
            "age_bonus": b_age,
            "extra_bonus": b_extra,
            "total_score": min(1.5, total)
        })

    buy_score = sum(e["total_score"] for e in entries if str(e["sig"].get("action", "")).upper() == "BUY")
    sell_score = sum(e["total_score"] for e in entries if str(e["sig"].get("action", "")).upper() == "SELL")

    top_buy = max((e for e in entries if str(e["sig"].get("action","")).upper()=="BUY"), key=lambda x: x["total_score"], default=None)
    top_sell = max((e for e in entries if str(e["sig"].get("action","")).upper()=="SELL"), key=lambda x: x["total_score"], default=None)

    diff = buy_score - sell_score

    # volatility bias: favor momentum if ATR large, favor fade if ATR small
    vol_bias = 0.0
    try:
        if atr is not None:
            if atr > cfg["prefer_momentum_atr"]:
                vol_bias = -0.05
            elif atr < cfg["prefer_momentum_atr"] * 0.5:
                vol_bias = 0.05
    except Exception:
        vol_bias = 0.0
    diff += vol_bias

    meta = {
        "entries": entries,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "diff": diff,
        "vol_bias": vol_bias,
        "atr": atr,
        "vwap": vwap,
        "vpvr": vpvr
    }

    # decisive selection
    if diff >= cfg["min_diff_to_take"]:
        chosen = top_buy["sig"] if top_buy else None
        return {"decision": "BUY", "reason": f"buy aggregated stronger; buy={buy_score:.2f} sell={sell_score:.2f}", "chosen_signal": chosen, "size_mult": 1.0, "meta": meta}
    if diff <= -cfg["min_diff_to_take"]:
        chosen = top_sell["sig"] if top_sell else None
        return {"decision": "SELL", "reason": f"sell aggregated stronger; buy={buy_score:.2f} sell={sell_score:.2f}", "chosen_signal": chosen, "size_mult": 1.0, "meta": meta}

    # very close contest
    if abs(diff) < cfg["glue_threshold"]:
        return {"decision": "NO_TRADE", "reason": "signals nearly balanced; avoid whipsaw", "chosen_signal": None, "size_mult": 0.0, "meta": meta}

    # intermediate: small trade
    if diff > 0:
        chosen = top_buy["sig"] if top_buy else None
        return {"decision": "SMALL_BUY", "reason": "buy slightly stronger; small size recommended", "chosen_signal": chosen, "size_mult": cfg["small_trade_mult"], "meta": meta}
    else:
        chosen = top_sell["sig"] if top_sell else None
        return {"decision": "SMALL_SELL", "reason": "sell slightly stronger; small size recommended", "chosen_signal": chosen, "size_mult": cfg["small_trade_mult"], "meta": meta}


# simple CLI test helper (not executed on import)
if __name__ == "__main__":
    # example quick test
    sample_analysis = {
        "liquidation_strategies_lite": {"action": "SELL", "score": 0.6, "confidence": "MEDIUM", "context": {"z": 2.2, "total_amount": 300000}},
        "fade_reentry_strategy": {"action": "BUY", "score": 0.45, "confidence": "LOW", "context": {"lpi": -0.3}},
        "vpvr": {"poc": 4603.44}
    }
    print(resolve_fade_liquidation_conflict(sample_analysis, vpvr=sample_analysis.get("vpvr"), atr=0.8))
