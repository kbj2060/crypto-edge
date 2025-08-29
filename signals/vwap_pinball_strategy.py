from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from indicators.global_indicators import get_vwap, get_atr
from utils.time_manager import get_time_manager

@dataclass
class VWAPPinballCfg:
    entry_sigma_steps: Tuple[float, ...] = (0.15, 0.3, 0.6, 1.0)   # 기존보다 촘촘
    max_entries: int = 3
    tick: float = 0.02
    atr_stop_mult: float = 0.6
    tp_vwap_bonus_sigma: float = 0.2
    require_bounce_confirmation: bool = True
    bounce_lookback_bars: int = 2
    min_body_ratio: float = 0.14
    w_distance: float = 0.40
    w_bounce: float = 0.35
    w_volume: float = 0.15
    min_vol_req: float = 0.03
    min_bounce_score: float = 0.12
    debug: bool = False

    # 추가된 파라미터
    score_threshold: float = 0.40      # 이 값 미만이면 신호 무시 (기본 0.40)
    momentum_weight: float = 0.20      # 모멘텀 가중치
    slope_weight: float = 0.20         # price-vs-vwap 가중치
    # distance/bounce/volume 가중치의 합과 momentum+ slope 합이 1.0 이되도록 권장
    # (기본 w_distance+w_bounce+w_volume = 0.90, momentum+slope=0.40 -> 총 1.3 이므로 내부에서 정규화 처리)

class VWAPPinballStrategy:
    """
    VWAP Pinball with balanced scoring and low-score suppression.
    - Automatically skips wick-only triggers unless bounce/vol/close-reentry criteria pass.
    - Adds momentum & price-vs-vwap bias and a score threshold to avoid low-score outputs.
    """
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
                cond = (c > o) and (body_ratio >= self.cfg.min_body_ratio) and (lower_pct >= 0.30)
            else:
                cond = (c < o) and (body_ratio >= self.cfg.min_body_ratio) and (upper_pct >= 0.30)
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

    def _momentum_score(self, prev_c: float, last_c: float, vwap_std: float) -> float:
        # normalize price change by vwap_std; map to (0,1) with tanh
        denom = max(1e-9, vwap_std)
        raw = (last_c - prev_c) / denom
        return float((np.tanh(raw) + 1.0) / 2.0)

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
        prev_c = float(prev['close'])

        if self.cfg.debug:
            print(f"[VWAP_PINBALL DEBUG] vwap={vwap_val:.3f} vwap_std={vwap_std:.3f} atr={atr:.3f} "
                  f"last_h={last_h:.3f} last_l={last_l:.3f} last_c={last_c:.3f} prev_c={prev_c:.3f}")

        cand_signals = []
        reentry_margin = max(0.01, 0.10 * vwap_std)

        # compute momentum once
        mom_score_global = self._momentum_score(prev_c, last_c, vwap_std)

        for idx, sigma in enumerate(self.cfg.entry_sigma_steps[:self.cfg.max_entries]):
            threshold_buy = vwap_val - sigma * vwap_std
            threshold_sell = vwap_val + sigma * vwap_std

            vol_q = self._volume_strength(df3)
            bounce_q_buy = self._bounce_quality(df3.iloc[-(self.cfg.bounce_lookback_bars+1):], "BUY") if self.cfg.require_bounce_confirmation else 0.5
            bounce_q_sell = self._bounce_quality(df3.iloc[-(self.cfg.bounce_lookback_bars+1):], "SELL") if self.cfg.require_bounce_confirmation else 0.5

            # ---------- BUY detection ----------
            buy_trigger = False; buy_reason = None
            if last_l <= threshold_buy:
                buy_trigger = True; buy_reason = "low_touched"
            elif prev_c <= threshold_buy and last_c >= threshold_buy:
                buy_trigger = True; buy_reason = "close_reentry"
            elif last_c > prev_c and last_c < vwap_val:
                buy_trigger = True; buy_reason = "momentum_reentry"

            if buy_trigger:
                wick_only = (buy_reason == "low_touched") and not (prev_c <= threshold_buy and last_c >= threshold_buy)
                if wick_only:
                    allow_wick = (bounce_q_buy >= self.cfg.min_bounce_score) or (vol_q >= self.cfg.min_vol_req) or (last_c >= (threshold_buy + reentry_margin))
                    if not allow_wick:
                        if self.cfg.debug:
                            print(f"[VWAP_PINBALL DEBUG] BUY skipped (wick-only): bounce_q={bounce_q_buy:.3f} vol_q={vol_q:.3f} last_c={last_c:.3f} th={threshold_buy:.3f}")
                        # skip
                    else:
                        entry = last_h + self.cfg.tick
                        stop = last_l - self.cfg.atr_stop_mult * atr
                        tp1 = vwap_val
                        tp2 = vwap_val + self.cfg.tp_vwap_bonus_sigma * vwap_std
                        cand_signals.append(("BUY", sigma, entry, stop, tp1, tp2, bounce_q_buy, vol_q, buy_reason))
                else:
                    entry = last_h + self.cfg.tick
                    stop = last_l - self.cfg.atr_stop_mult * atr
                    tp1 = vwap_val
                    tp2 = vwap_val + self.cfg.tp_vwap_bonus_sigma * vwap_std
                    cand_signals.append(("BUY", sigma, entry, stop, tp1, tp2, bounce_q_buy, vol_q, buy_reason))

            # ---------- SELL detection ----------
            sell_trigger = False; sell_reason = None
            if last_h >= threshold_sell:
                sell_trigger = True; sell_reason = "high_touched"
            elif prev_c >= threshold_sell and last_c <= threshold_sell:
                sell_trigger = True; sell_reason = "close_rejection"
            elif last_c < prev_c and last_c > vwap_val:
                sell_trigger = True; sell_reason = "momentum_reentry"

            if sell_trigger:
                wick_only = (sell_reason == "high_touched") and not (prev_c >= threshold_sell and last_c <= threshold_sell)
                if wick_only:
                    allow_wick = (bounce_q_sell >= self.cfg.min_bounce_score) or (vol_q >= self.cfg.min_vol_req) or (last_c <= (threshold_sell - reentry_margin))
                    if not allow_wick:
                        if self.cfg.debug:
                            print(f"[VWAP_PINBALL DEBUG] SELL skipped (wick-only): bounce_q={bounce_q_sell:.3f} vol_q={vol_q:.3f} last_c={last_c:.3f} th={threshold_sell:.3f}")
                        # skip
                    else:
                        entry = last_l - self.cfg.tick
                        stop = last_h + self.cfg.atr_stop_mult * atr
                        tp1 = vwap_val
                        tp2 = vwap_val - self.cfg.tp_vwap_bonus_sigma * vwap_std
                        cand_signals.append(("SELL", sigma, entry, stop, tp1, tp2, bounce_q_sell, vol_q, sell_reason))
                else:
                    entry = last_l - self.cfg.tick
                    stop = last_h + self.cfg.atr_stop_mult * atr
                    tp1 = vwap_val
                    tp2 = vwap_val - self.cfg.tp_vwap_bonus_sigma * vwap_std
                    cand_signals.append(("SELL", sigma, entry, stop, tp1, tp2, bounce_q_sell, vol_q, sell_reason))

        if not cand_signals:
            if self.cfg.debug:
                print("[VWAP_PINBALL DEBUG] no candidates for any sigma steps")
            return None

        # scoring & include momentum and price-vs-vwap bias
        scored = []
        # normalize weights so total contribution = 1
        base_w_sum = (self.cfg.w_distance + self.cfg.w_bounce + self.cfg.w_volume)
        extra_w_sum = (self.cfg.momentum_weight + self.cfg.slope_weight)
        # if overall >1, scale down proportionally
        scale = 1.0 / max(1.0, base_w_sum + extra_w_sum)

        for (direction, sigma, entry, stop, tp1, tp2, bounce_q, vol_q, reason) in cand_signals:
            dist_score = self._score_lin(max(0.0, (3.0 - sigma)), 0.0, 3.0)
            bounce_score = float(bounce_q)
            vol_score = float(vol_q)
            base_score = (self.cfg.w_distance * dist_score + self.cfg.w_bounce * bounce_score + self.cfg.w_volume * vol_score)

            # momentum: high positive favors BUY
            mom = mom_score_global
            # price vs vwap bias (price_score in [0,1], >0.5 means price above vwap)
            price_vs_vwap_raw = (last_c - vwap_val) / max(1e-9, vwap_std)
            price_score = float((np.tanh(price_vs_vwap_raw) + 1.0) / 2.0)

            # direction-specific combination
            if direction == "BUY":
                dir_extra = self.cfg.momentum_weight * mom + self.cfg.slope_weight * (1.0 - price_score)
            else:
                dir_extra = self.cfg.momentum_weight * (1.0 - mom) + self.cfg.slope_weight * price_score

            total_score = (base_score * (1.0 - extra_w_sum) + dir_extra) * scale
            # clamp
            total_score = max(0.0, min(1.0, float(total_score)))

            scored.append({
                "direction": direction,
                "sigma": sigma,
                "entry": float(entry),
                "stop": float(stop),
                "targets": [float(tp1), float(tp2)],
                "dist_score": dist_score,
                "bounce_score": bounce_score,
                "vol_score": vol_score,
                "momentum": float(mom),
                "price_score": float(price_score),
                "score": total_score,
                "reason": reason
            })

        # filter by threshold
        scored_filtered = [s for s in scored if s["score"] >= float(self.cfg.score_threshold)]
        if not scored_filtered:
            if self.cfg.debug:
                # show best candidate even if below threshold for debugging
                best_tmp = sorted(scored, key=lambda x: x["score"], reverse=True)[0]
                print(f"[VWAP_PINBALL DEBUG] best below threshold score={best_tmp['score']:.3f} reason={best_tmp['reason']} "
                      f"mom={best_tmp['momentum']:.3f} price_score={best_tmp['price_score']:.3f}")
            return None

        best = sorted(scored_filtered, key=lambda x: x["score"], reverse=True)[0]
        confidence = self._conf_bucket(best["score"])
        reasons = [
            f"sigma={best['sigma']:.2f} (dist_score={best['dist_score']:.2f})",
            f"bounce_score={best['bounce_score']:.2f}",
            f"vol_score={best['vol_score']:.2f}",
            f"momentum={best['momentum']:.2f}",
            f"price_bias={best['price_score']:.2f}",
            f"trigger={best.get('reason')}"
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

        if self.cfg.debug:
            print(f"[VWAP_PINBALL DEBUG] chosen {result['action']} score={result['score']:.3f} reason={best.get('reason')}")

        return result
