
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from indicators.global_indicators import get_vwap, get_atr
from utils.time_manager import get_time_manager

@dataclass
class VWAPPinballCfg:
    entry_sigma_steps: Tuple[float, ...] = (0.5, 1.0, 1.5)
    max_entries: int = 3
    tick: float = 0.05
    atr_stop_mult: float = 0.7
    tp_vwap_bonus_sigma: float = 0.3
    require_bounce_confirmation: bool = False
    bounce_lookback_bars: int = 2
    min_body_ratio: float = 0.20
    w_distance: float = 0.40
    w_bounce: float = 0.40
    w_volume: float = 0.10

class VWAPPinballStrategy:
    def __init__(self, cfg: VWAPPinballCfg = VWAPPinballCfg()):
        self.cfg = cfg
        self.tm = get_time_manager()

    def _conf_bucket(self, v: float) -> str:
        if v >= 0.75: return "HIGH"
        if v >= 0.50: return "MEDIUM"
        return "LOW"

    def _score_lin(self, x: float, lo: float, hi: float) -> float:
        if hi == lo: return 0.0
        return float(max(0.0, min(1.0, (x - lo) / (hi - lo))))

    def _bounce_quality(self, recent: pd.DataFrame, direction: str) -> float:
        lb = max(1, min(len(recent), self.cfg.bounce_lookback_bars))
        seg = recent.tail(lb)
        qual = 0.0
        count = 0
        for _, row in seg.iterrows():
            o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
            rng = max(1e-9, h - l)
            body = abs(c - o)
            body_ratio = body / rng if rng > 0 else 0.0
            lower_pct = (c - l) / rng if rng > 0 else 0.0
            upper_pct = (h - c) / rng if rng > 0 else 1.0
            if direction == "BUY":
                cond = (c > o) and (body_ratio >= self.cfg.min_body_ratio) and (lower_pct >= 0.35)
            else:
                cond = (c < o) and (body_ratio >= self.cfg.min_body_ratio) and (upper_pct >= 0.35)
            if cond:
                score = 0.4 * self._score_lin(body_ratio, self.cfg.min_body_ratio, 1.0) + 0.6 * 0.5
                qual += score
                count += 1
        if count == 0:
            return 0.0
        return float(max(0.0, min(1.0, qual / count)))

    def _volume_strength(self, recent: pd.DataFrame) -> float:
        if 'quote_volume' in recent.columns:
            v = recent['quote_volume'].astype(float)
        elif 'volume' in recent.columns and 'close' in recent.columns:
            v = (recent['volume'] * recent['close']).astype(float)
        else:
            return 0.0
        if len(v) < 5:
            return 0.0
        ma = v.rolling(20, min_periods=1).mean().iloc[-1]
        last = float(v.iloc[-1])
        if ma <= 0:
            return 0.0
        ratio = last / ma
        if ratio <= 1.0:
            return 0.0
        if ratio >= 2.0:
            return 1.0
        return float((ratio - 1.0) / 1.0)

    def on_kline_close_3m(self, df3: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if df3 is None or len(df3) < 3:
            return None

        try:
            vwap_val, vwap_std = get_vwap()
            vwap_std = float(vwap_std or 0.0)
        except Exception:
            vwap_val = None
            vwap_std = float(np.std(df3['close'].values)) * 0.02

        if vwap_val is None:
            return None
        vwap_val = float(vwap_val)
        vwap_std = float(vwap_std) if vwap_std and vwap_std > 0 else max(0.01, float(np.std(df3['close'].values)) * 0.02)

        atr = get_atr() or float(np.std(df3['close'].pct_change().fillna(0).values) * df3['close'].iloc[-1] or 1.0)
        atr = float(atr or 1.0)

        last = df3.iloc[-1]
        prev = df3.iloc[-2]
        last_o, last_h, last_l, last_c = float(last['open']), float(last['high']), float(last['low']), float(last['close'])

        cand_signals = []
        for idx, sigma in enumerate(self.cfg.entry_sigma_steps[:self.cfg.max_entries]):
            threshold_buy = vwap_val - sigma * vwap_std
            threshold_sell = vwap_val + sigma * vwap_std

            if last_l <= threshold_buy:
                bounce_q = self._bounce_quality(df3.iloc[-(self.cfg.bounce_lookback_bars+1):], "BUY") if self.cfg.require_bounce_confirmation else 0.5
                vol_q = self._volume_strength(df3)
                if bounce_q > 0 or sigma <= 1.0 or not self.cfg.require_bounce_confirmation:
                    entry = last_h + self.cfg.tick
                    stop = last_l - self.cfg.atr_stop_mult * atr
                    tp1 = vwap_val
                    tp2 = vwap_val + self.cfg.tp_vwap_bonus_sigma * vwap_std
                    cand_signals.append(("BUY", sigma, entry, stop, tp1, tp2, bounce_q, vol_q))

            if last_h >= threshold_sell:
                bounce_q = self._bounce_quality(df3.iloc[-(self.cfg.bounce_lookback_bars+1):], "SELL") if self.cfg.require_bounce_confirmation else 0.5
                vol_q = self._volume_strength(df3)
                if bounce_q > 0 or sigma <= 1.0 or not self.cfg.require_bounce_confirmation:
                    entry = last_l - self.cfg.tick
                    stop = last_h + self.cfg.atr_stop_mult * atr
                    tp1 = vwap_val
                    tp2 = vwap_val - self.cfg.tp_vwap_bonus_sigma * vwap_std
                    cand_signals.append(("SELL", sigma, entry, stop, tp1, tp2, bounce_q, vol_q))

        if not cand_signals:
            return None

        scored = []
        for (direction, sigma, entry, stop, tp1, tp2, bounce_q, vol_q) in cand_signals:
            dist_score = self._score_lin(max(0.0, (3.0 - sigma)), 0.0, 3.0)
            bounce_score = float(bounce_q)
            vol_score = float(vol_q)
            score = (self.cfg.w_distance * dist_score) + (self.cfg.w_bounce * bounce_score) + (self.cfg.w_volume * vol_score)
            score = max(0.0, min(1.0, score))
            scored.append({
                "direction": direction,
                "sigma": sigma,
                "entry": float(entry),
                "stop": float(stop),
                "targets": [float(tp1), float(tp2)],
                "dist_score": dist_score,
                "bounce_score": bounce_score,
                "vol_score": vol_score,
                "score": score
            })

        best = sorted(scored, key=lambda x: x["score"], reverse=True)[0]
        confidence = self._conf_bucket(best["score"])
        reasons: List[str] = [
            f"sigma={best['sigma']:.2f} (dist_score={best['dist_score']:.2f})",
            f"bounce_score={best['bounce_score']:.2f}",
            f"vol_score={best['vol_score']:.2f}"
        ]

        result = {
            "stage": "ENTRY",
            "action": best["direction"],
            "entry": float(best["entry"]),
            "stop": float(best["stop"]),
            "targets": best["targets"],
            "context": {
                "mode": "VWAP_PINBALL",
                "vwap": float(vwap_val),
                "vwap_std": float(vwap_std),
                "atr": float(atr),
                "sigma": float(best["sigma"])
            },
            "score": float(best["score"]),
            "confidence": confidence,
            "reasons": reasons
        }

        return result
