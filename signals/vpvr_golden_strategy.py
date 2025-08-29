# vpvr_golden_strategy.py
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from indicators.global_indicators import get_atr, get_vpvr

class LVNGoldenPocket:
    @dataclass
    class VPVRConfig:
        lookback_bars: int = 200       # 반응 빠르게: lookback 줄임
        bin_size: Optional[float] = None
        max_bins: int = 80
        use_quote_volume: bool = True
        min_price_tick: float = 0.01

    @dataclass
    class LVNSettings:
        low_percentile: float = 0.40   # 더 많은 LVN 후보 허용
        local_min: bool = True
        merge_neighbors: bool = True
        merge_ticks: int = 2

    @dataclass
    class GoldenPocketCfg:
        swing_lookback: int = 60       # 짧게
        dryup_lookback: int = 8
        dryup_window: int = 3
        dryup_frac: float = 0.6      # dry-up 판단 완화 (낮출수록 더 쉽게 dryup)
        dryup_k: int = 1
        tolerance_atr_mult: float = 0.6
        confirm_body_ratio: float = 0.06
        atr_len: int = 14
        tick: float = 0.01
        lvn_max_atr: float = 6.0
        confirm_mode: str = "wick_or_break"
        zone_widen_atr: float = 0.6

        # ----- 새로 추가된 설정(민감도 조절용) -----
        prox_tol_mult: float = 2.0
        prox_zone_frac: float = 0.75
        lvn_tol_mult: float = 1.5
        min_body_allow: float = 0.04
        vol_multiplier_relax: float = 0.6
        soft_accept_min_vol_ratio: float = 0.8
        enable_soft_accept: bool = True
        soft_accept_zone_frac: float = 0.5
        require_k_min: int = 1
        allow_wick_only: bool = True

    @dataclass
    class TargetsStopsCfg:
        stop_atr_mult: float = 0.6
        tp_R1: float = 1.0
        tp_R2: float = 1.6

    def __init__(self, vpvr: Optional['LVNGoldenPocket.VPVRConfig'] = None,
                 lvn: Optional['LVNGoldenPocket.LVNSettings'] = None,
                 gp: Optional['LVNGoldenPocket.GoldenPocketCfg'] = None,
                 risk: Optional['LVNGoldenPocket.TargetsStopsCfg'] = None):
        self.vpvr = vpvr or LVNGoldenPocket.VPVRConfig()
        self.lvn = lvn or LVNGoldenPocket.LVNSettings()
        self.gp = gp or LVNGoldenPocket.GoldenPocketCfg()
        self.risk = risk or LVNGoldenPocket.TargetsStopsCfg()

    def _atr(self, df: pd.DataFrame, n: int) -> pd.Series:
        h, l, c = df['high'], df['low'], df['close']
        prev_c = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return tr.rolling(n, min_periods=1).mean()

    def _compute_vpvr(self, df: pd.DataFrame) -> Dict[str, Any]:
        cfg = self.vpvr
        df_ = df.tail(cfg.lookback_bars).copy()
        if cfg.use_quote_volume and 'quote_volume' in df_.columns:
            vol = df_['quote_volume'].to_numpy(dtype=float)
        else:
            vol = (df_['volume'] * df_['close']).to_numpy(dtype=float)
        lows  = df_['low'].to_numpy(dtype=float)
        highs = df_['high'].to_numpy(dtype=float)
        pmin = float(np.nanmin(lows))
        pmax = float(np.nanmax(highs))
        if pmax <= pmin:
            pmax = pmin + cfg.min_price_tick
        if cfg.bin_size is None:
            span = pmax - pmin
            bins = max(10, min(cfg.max_bins, int(span / max(cfg.min_price_tick, span / cfg.max_bins)) if span>0 else cfg.max_bins))
            bin_size = max(cfg.min_price_tick, span / bins) if span>0 else cfg.min_price_tick
        else:
            bin_size = max(cfg.min_price_tick, float(cfg.bin_size))
        nbins = int(np.ceil((pmax - pmin) / bin_size)) + 1
        hist  = np.zeros(nbins, dtype=float)
        for lo, hi, v in zip(lows, highs, vol):
            lo_idx = int(np.floor((max(lo, pmin) - pmin) / bin_size))
            hi_idx = int(np.floor((min(hi, pmax) - pmin) / bin_size))
            if hi_idx < lo_idx:
                lo_idx, hi_idx = hi_idx, lo_idx
            width = max(1, hi_idx - lo_idx + 1)
            hist[lo_idx:hi_idx+1] += v / width
        centers = pmin + (np.arange(nbins) + 0.5) * bin_size
        poc_idx = int(np.argmax(hist))
        return {
            "centers": centers,
            "volumes": hist,
            "bin_size": float(bin_size),
            "poc_index": int(poc_idx),
            "poc_price": float(centers[poc_idx]),
            "pmin": float(pmin),
            "pmax": float(pmax),
        }

    def _find_lvn_nodes(self, vpvr: Dict[str, Any]) -> List[Tuple[int, float, float]]:
        settings = self.lvn
        vols = vpvr["volumes"]; centers = vpvr["centers"]
        n = len(vols)
        if n < 3: return []
        thresh = np.quantile(vols[~np.isnan(vols)], settings.low_percentile)
        idxs = []
        for i in range(1, n-1):
            if vols[i] <= thresh:
                if not settings.local_min or (vols[i] < vols[i-1] and vols[i] < vols[i+1]):
                    idxs.append(i)
        if settings.merge_neighbors and idxs:
            merged = [idxs[0]]
            for i in idxs[1:]:
                if i - merged[-1] <= settings.merge_ticks:
                    merged[-1] = i if vols[i] < vols[merged[-1]] else merged[-1]
                else:
                    merged.append(i)
            idxs = merged
        return [(int(i), float(centers[i]), float(vols[i])) for i in idxs]

    def _nearest_lvn_to_price(self, lvns: List[Tuple[int, float, float]], price: float) -> Optional[Tuple[int, float, float]]:
        if not lvns:
            print('lvns 없음')
            return None
        i = int(np.argmin([abs(p - price) for _, p, _ in lvns]))
        return lvns[i]

    def _detect_last_swing(self, df: pd.DataFrame, lookback: int) -> Optional[Tuple[int, int]]:
        if len(df) < lookback + 1:
            lookback = len(df) - 1
        if lookback < 6:
            print('lookback 6 이하')
            return None
        seg = df.tail(lookback)
        idx_low  = int(seg['low' ].idxmin())
        idx_high = int(seg['high'].idxmax())
        if idx_high > idx_low:
            return idx_low, idx_high
        else:
            return idx_high, idx_low

    def _golden_pocket_zone(self, df: pd.DataFrame, swing: Tuple[int, int]) -> Tuple[float, float, str]:
        i0, i1 = swing
        lo = float(df.loc[i0, 'low'])
        hi = float(df.loc[i1, 'high'])
        if i1 > i0 and hi > lo:
            rng = hi - lo
            gp_low  = hi - 0.65  * rng
            gp_high = hi - 0.618 * rng
            return gp_low, gp_high, "long"
        else:
            hi2 = float(df.loc[i0, 'high'])
            lo2 = float(df.loc[i1, 'low'])
            rng = hi2 - lo2
            gp_low  = lo2 + 0.618 * rng
            gp_high = lo2 + 0.65  * rng
            return gp_low, gp_high, "short"

    def _volume_dryup(self, df: pd.DataFrame, sma_len: int, window: int, dry_frac: float, dry_k: int = 1) -> bool:
        """
        관대화된 dry-up 판정:
        - recent window봉 중 sma_len 으로 계산한 SMA 대비 dry_frac 이하인 봉이
          최소 required 개수 이상이면 dry-up으로 간주.
        - required는 window의 20% 또는 dry_k 중 큰 값으로 설정(작은 window에서도 통과 쉬움)
        """
        if 'quote_volume' in df.columns:
            v_series = df['quote_volume'].astype(float)
        else:
            v_series = (df['volume'] * df['close']).astype(float)
        sma = v_series.rolling(sma_len, min_periods=1).mean()
        recent = df.tail(window)
        idx = recent.index
        try:
            cond_series = (v_series.loc[idx] <= (dry_frac * sma.loc[idx])).fillna(False).astype(bool)
        except Exception:
            cond_series = (v_series.tail(window) <= (dry_frac * sma.tail(window))).fillna(False).astype(bool)
        conds = cond_series.tolist()
        sat = sum(1 for c in conds if c)
        min_frac = 0.2
        required = max(int(dry_k), max(1, int(round(window * min_frac))))
        return sat >= required

    def _rejection_confirm(self,
                       df: pd.DataFrame,
                       zone_low: float,
                       zone_high: float,
                       direction: str,
                       lvn_price: Optional[float],
                       tol: float,
                       body_ratio_min: float,
                       tick: float,
                       lookback: int = 3,
                       require_k: int = 1,
                       vol_multiplier: Optional[float] = None,
                       allow_proximity: bool = True,
                       allow_wick_only: bool = True
                       ) -> bool:
        """
        설정값 기반 완화된 rejection 확인.
        """
        gp = getattr(self, "gp", None)
        prox_tol_mult = getattr(gp, "prox_tol_mult", 2.0)
        prox_zone_frac = getattr(gp, "prox_zone_frac", 0.75)
        lvn_tol_mult = getattr(gp, "lvn_tol_mult", 1.5)
        min_body_allow_cfg = getattr(gp, "min_body_allow", None)
        vol_relax = getattr(gp, "vol_multiplier_relax", 0.6)
        soft_min_vol_ratio = getattr(gp, "soft_accept_min_vol_ratio", 0.8)
        enable_soft_accept = getattr(gp, "enable_soft_accept", True)
        soft_zone_frac = getattr(gp, "soft_accept_zone_frac", 0.5)
        require_k_min_cfg = getattr(gp, "require_k_min", 1)
        allow_wick_only_cfg = getattr(gp, "allow_wick_only", True)

        n = len(df)
        if n == 0:
            return False

        lookback = min(max(1, int(lookback)), n)
        require_k = max(int(require_k), int(require_k_min_cfg), 1)

        if vol_multiplier is not None:
            if 'quote_volume' in df.columns:
                vol_series = df['quote_volume'].astype(float)
            elif 'volume' in df.columns:
                vol_series = (df['volume'] * df['close']).astype(float)
            else:
                vol_series = None
            vol_sma = vol_series.rolling(max(1, lookback), min_periods=1).mean() if vol_series is not None else None
        else:
            vol_series = vol_sma = None

        seg = df.iloc[-lookback:]
        sat = 0
        zone_mid = 0.5 * (zone_low + zone_high)
        zone_width = max(1e-9, (zone_high - zone_low))

        prox_tol = max(tol * float(prox_tol_mult), zone_width * float(prox_zone_frac))
        min_body_allow = float(min_body_allow_cfg) if min_body_allow_cfg is not None else max(0.04, body_ratio_min * 0.4)

        debug_rows = []

        for idx, row in seg.iterrows():
            o = float(row.get('open', row.get('o', 0.0)))
            h = float(row.get('high', row.get('h', 0.0)))
            l = float(row.get('low', row.get('l', 0.0)))
            c = float(row.get('close', row.get('c', 0.0)))
            rng = max(1e-9, h - l)
            body = abs(c - o) / rng if rng > 0 else 0.0

            bar_overlaps = (h >= zone_low - tol) and (l <= zone_high + tol)
            mid = 0.5 * (h + l)
            near_zone_prox = abs(mid - zone_mid) <= prox_tol
            in_zone = bar_overlaps or (allow_proximity and near_zone_prox)

            near_lvn = True
            if lvn_price is not None:
                mid_bar = 0.5 * (l + h)
                near_lvn = (abs(lvn_price - mid_bar) <= (tol * float(lvn_tol_mult))) or (l - tol*lvn_tol_mult <= lvn_price <= h + tol*lvn_tol_mult)

            body_ok = body >= min_body_allow

            if direction == 'long':
                wick_through = (l <= zone_high + tol) or (lvn_price is not None and l <= lvn_price + tol*lvn_tol_mult)
                close_back = (c > o and c >= zone_high - tick) or (c >= zone_high - (1.0 * tick))
                wick_or_close = wick_through or close_back or (abs(mid - zone_mid) <= prox_tol)
            else:
                wick_through = (h >= zone_low - tol) or (lvn_price is not None and h >= lvn_price - tol*lvn_tol_mult)
                close_back = (c < o and c <= zone_low + tick) or (c <= zone_low + (1.0 * tick))
                wick_or_close = wick_through or close_back or (abs(mid - zone_mid) <= prox_tol)

            ok = False

            if in_zone and near_lvn and body_ok and wick_or_close:
                ok = True
            else:
                if (allow_wick_only and allow_wick_only_cfg) and near_lvn and wick_or_close:
                    vol_ok = True
                    if vol_series is not None and vol_sma is not None and vol_multiplier is not None:
                        try:
                            sma_v = float(vol_sma.loc[idx])
                            if sma_v > 0:
                                vol_req = max(1.0, (vol_multiplier * float(vol_relax)) * sma_v)
                                vol_ok = float(vol_series.loc[idx]) >= vol_req
                            else:
                                vol_ok = True
                        except Exception:
                            vol_ok = True
                    if (body >= max(0.03, body_ratio_min * 0.35)) and vol_ok:
                        ok = True

            debug_rows.append({
                "idx": idx, "o": o, "h": h, "l": l, "c": c, "body": round(body, 3),
                "zone_low": zone_low, "zone_high": zone_high, "tol": tol,
                "in_zone": bool(in_zone), "near_lvn": bool(near_lvn),
                "wick_or_close": bool(wick_or_close), "body_ok": bool(body_ok), "ok": bool(ok)
            })

            if ok:
                sat += 1

        if sat >= require_k:
            result = True
        elif sat >= 1 and enable_soft_accept:
            soft_ok = False
            if vol_series is not None and vol_sma is not None:
                last_idx = seg.index[-1]
                try:
                    sma_v = float(vol_sma.loc[last_idx])
                    last_v = float(vol_series.loc[last_idx])
                    if sma_v > 0 and last_v >= soft_min_vol_ratio * sma_v:
                        soft_ok = True
                except Exception:
                    soft_ok = True
            if not soft_ok:
                last_mid = 0.5 * (seg.iloc[-1]['high'] + seg.iloc[-1]['low'])
                if abs(last_mid - zone_mid) <= max(prox_tol, zone_width * float(soft_zone_frac)):
                    soft_ok = True
            result = soft_ok
        else:
            result = False

        if getattr(self, "debug", False):
            print(f"[REJECT_CONFIRM DEBUG] lookback={lookback} sat={sat} req={require_k} result={result}")
            for r in debug_rows:
                print(r)

        return bool(result)

    def evaluate(self, df: pd.DataFrame, now_utc: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        if df is None:
            print('df 없음')
            return None
        need = max(self.vpvr.lookback_bars, self.gp.swing_lookback) + 5
        if len(df) < need:
            print('df 길이 부족')
            return None
        df = df.copy()
        df.index = pd.Index(range(len(df)))
        atr_last = get_atr()
        if atr_last is None:
            atr_last = float(self._atr(df, self.gp.atr_len).iloc[-1])
        tol = self.gp.tolerance_atr_mult * atr_last
        try:
            poc_global, hvn_global, lvn_global = get_vpvr()
        except Exception:
            poc_global, hvn_global, lvn_global = (None, None, None)
        vp = None
        lvns = []
        used_global = False
        if poc_global is not None:
            used_global = True
            poc_price = float(poc_global)
            hvn_price = float(hvn_global) if hvn_global is not None else None
            lvn_price = float(lvn_global) if lvn_global is not None else None
            if lvn_price is not None:
                lvns = [(0, lvn_price, 0.0)]
            else:
                lvns = []
        else:
            vp = self._compute_vpvr(df)
            lvns = self._find_lvn_nodes(vp)
            poc_price = float(vp["poc_price"])
            hvn_price = None
            lvn_price = None
        swing = self._detect_last_swing(df, self.gp.swing_lookback)
        if swing is None:
            print('swing 없음')
            return None
        gp_low, gp_high, direction = self._golden_pocket_zone(df, swing)
        zone_mid = 0.5 * (gp_low + gp_high)
        nearest_lvn = self._nearest_lvn_to_price(lvns, zone_mid) if lvns else None
        lvn_price = nearest_lvn[1] if nearest_lvn else None
        
        # volume dry-up (완화된 판정)
        # if not self._volume_dryup(df, self.gp.dryup_lookback, self.gp.dryup_window, self.gp.dryup_frac, self.gp.dryup_k):
        #     print('volume dryup 없음')
        #     return None

        # rejection confirmation (완화된 조건)
        if not self._rejection_confirm(df, gp_low, gp_high, direction, lvn_price, tol,
                                        self.gp.confirm_body_ratio, self.gp.tick, lookback=3, require_k=1):
            print('rejection confirm 없음')
            return None
        last = df.iloc[-1]
        h = float(last['high']); l = float(last['low']); c = float(last['close'])
        gp_width_adj = self.gp.zone_widen_atr * atr_last
        adj_low = gp_low - gp_width_adj
        adj_high = gp_high + gp_width_adj
        if direction == 'long':
            entry = h + self.gp.tick
            stop  = min(l, adj_low, (lvn_price if lvn_price is not None else l)) - self.gp.tick
            stop  = min(stop, c - self.risk.stop_atr_mult * atr_last)
            R = entry - stop
            tp1, tp2 = entry + self.risk.tp_R1 * R, entry + self.risk.tp_R2 * R
            action = "BUY"
        else:
            entry = l - self.gp.tick
            stop  = max(h, adj_high, (lvn_price if lvn_price is not None else h)) + self.gp.tick
            stop  = max(stop, c + self.risk.stop_atr_mult * atr_last)
            R = stop - entry
            tp1, tp2 = entry - self.risk.tp_R1 * R, entry - self.risk.tp_R2 * R
            action = "SELL"
        ctx = {
            "mode": "VPVR_LVN_GP_DRYUP",
            "direction": direction,
            "gp_zone": [float(gp_low), float(gp_high)],
            "lvn_price": float(lvn_price) if lvn_price is not None else None,
            "poc_price": float(poc_price),
            "atr": float(atr_last),
            "tol_atr_mult": float(self.gp.tolerance_atr_mult),
            "dryup": {
                "lookback": self.gp.dryup_lookback,
                "window": self.gp.dryup_window,
                "frac": self.gp.dryup_frac
            },
            "vpvr": {
                "source": "global" if used_global else "local",
                "bin_size": float(vp["bin_size"]) if vp is not None else None,
                "lookback_bars": self.vpvr.lookback_bars
            }
        }
        result = {
            "stage": "ENTRY",
            "action": action,
            "entry": float(entry),
            "stop": float(stop),
            "targets": [float(tp1), float(tp2), float(poc_price)],
            "context": ctx
        }
        return result
