from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

from indicators.global_indicators import get_atr, get_vpvr
from utils.time_manager import get_time_manager

class LVNGoldenPocket:
    """
    VPVR LVN Rejection + Golden Pocket (0.618~0.65) + Volume Dry-up
    - All logic is encapsulated in this single class.
    - Pass a pandas OHLCV DataFrame to `evaluate(df)` to get a signal dict or None.
    """

    @dataclass
    class VPVRConfig:
        lookback_bars: int = 300
        bin_size: Optional[float] = None
        max_bins: int = 80
        use_quote_volume: bool = True
        min_price_tick: float = 0.1

    @dataclass
    class LVNSettings:
        low_percentile: float = 0.40  # 0.30 -> 0.40 (완화)
        local_min: bool = False       # True -> False (완화)
        merge_neighbors: bool = True
        merge_ticks: int = 5          # 3 -> 5 (완화)

    @dataclass
    class GoldenPocketCfg:
        swing_lookback: int = 60      # 120 -> 60 (단축)
        dryup_lookback: int = 20      # 40 -> 20 (단축)
        dryup_window: int = 3         # 5 -> 3 (단축)
        dryup_frac: float = 0.9      # 0.8 -> 0.9 (완화)
        dryup_k: int = 1              # 2 -> 1 (완화)
        tolerance_atr_mult: float = 3.0  # 2.0 -> 3.0 (완화)
        confirm_body_ratio: float = 0.05 # 0.1 -> 0.05 (완화)
        atr_len: int = 14
        tick: float = 0.1
        lvn_max_atr: float = 6.0      # 4.0 -> 6.0 (완화)
        confirm_mode: str = "wick_or_break"  # 'wick' | 'break' | 'wick_or_break'
        zone_widen_atr: float = 0.5   # 0.3 -> 0.5 (완화)

    @dataclass
    class TargetsStopsCfg:
        stop_atr_mult: float = 0.8
        tp_R1: float = 2.5
        tp_R2: float = 4.0

    def __init__(
        self,
        vpvr: Optional['LVNGoldenPocket.VPVRConfig'] = None,
        lvn: Optional['LVNGoldenPocket.LVNSettings'] = None,
        gp: Optional['LVNGoldenPocket.GoldenPocketCfg'] = None,
        risk: Optional['LVNGoldenPocket.TargetsStopsCfg'] = None,
    ):
        self.vpvr = vpvr or LVNGoldenPocket.VPVRConfig()
        self.lvn = lvn or LVNGoldenPocket.LVNSettings()
        self.gp = gp or LVNGoldenPocket.GoldenPocketCfg()
        self.risk = risk or LVNGoldenPocket.TargetsStopsCfg()
        self.tm = get_time_manager()

    # ===== Utilities =====

    def _atr(self, df: pd.DataFrame, n: int) -> pd.Series:
        h, l, c = df['high'], df['low'], df['close']
        prev_c = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        return tr.rolling(n, min_periods=1).mean()

    def _compute_vpvr(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        (기존 로컬 VPVR 계산) — 전역 VPVR이 없을 때만 폴백으로 사용됩니다.
        """
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

        # auto bin size
        if cfg.bin_size is None:
            span = pmax - pmin
            bins = max(10, min(cfg.max_bins, int(span / max(cfg.min_price_tick, span / cfg.max_bins))))
            bin_size = max(cfg.min_price_tick, span / bins)
        else:
            bin_size = max(cfg.min_price_tick, float(cfg.bin_size))

        nbins = int(np.ceil((pmax - pmin) / bin_size)) + 1
        hist  = np.zeros(nbins, dtype=float)

        # allocate volume uniformly across bar range
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
        if not lvns: return self._no_signal_result()
        i = int(np.argmin([abs(p - price) for _, p, _ in lvns]))
        return lvns[i]

    def _detect_last_swing(self, df: pd.DataFrame, lookback: int) -> Optional[Tuple[int, int]]:
        if len(df) < lookback + 1:
            lookback = len(df) - 1
        if lookback < 5:  # 10 -> 5 (완화)
            print("lookback < 5")
            return self._no_signal_result()
        seg = df.tail(lookback)
        idx_low  = int(seg['low' ].idxmin())
        idx_high = int(seg['high'].idxmax())
        if idx_high > idx_low:
            return idx_low, idx_high  # upswing
        else:
            return idx_high, idx_low  # downswing

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

    def _volume_dryup(self, df: pd.DataFrame, sma_len: int, window: int, dry_frac: float, dry_k: int = 3) -> bool:
        """
        dry-up 판정: 최근 window봉 중에서 sma_len 으로 계산한 SMA 대비 dry_frac 이하인 봉이
        최소 dry_k 개 이상이면 dry-up으로 간주.
        (기존 '모두 True' 방식보다 관대함)
        """
        # choose correct volume series
        if 'quote_volume' in df.columns:
            v = df['quote_volume'].astype(float)
        else:
            v = (df['volume'] * df['close']).astype(float)

        sma = v.rolling(sma_len, min_periods=1).mean()
        last_idx = df.tail(window).index
        conds = (v.loc[last_idx] <= dry_frac * sma.loc[last_idx]).to_list()
        sat = sum(1 for c in conds if bool(c))
        
        # 더 유연한 조건: 최소 1개 또는 전체의 50% 이상
        min_required = max(1, min(dry_k, window))
        alt_required = max(1, window // 2)  # 전체의 50%
        return sat >= min_required or sat >= alt_required


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
                       require_k: int = 2,
                       vol_multiplier: Optional[float] = None,
                       allow_proximity: bool = True,
                       allow_wick_only: bool = True
                       ) -> bool:
        """
        수정된 거부 확인 (lookback=3, require_k=2).
        - LONG은 '윅이 zone 상단(zone_high)으로 들어갔는지'를 확인합니다.
        - SHORT은 '윅이 zone 하단(zone_low)으로 들어갔는지'를 확인합니다.
        - allow_wick_only=True 이면 wick/close 만으로도 통과 가능(조건부).
        """
        n = len(df)
        if n == 0:
            return False

        lookback = min(max(1, int(lookback)), n)
        require_k = max(1, int(require_k))

        # optional volume series 준비
        if vol_multiplier is not None:
            vol_series = df['quote_volume'].astype(float)
            vol_sma = vol_series.rolling(max(1, lookback), min_periods=1).mean() if vol_series is not None else None
        else:
            vol_series = vol_sma = None

        seg = df.iloc[-lookback:]
        sat = 0
        debug_rows = []
        zone_mid = 0.5 * (zone_low + zone_high)
        zone_width = max(1e-9, (zone_high - zone_low))

        for idx, row in seg.iterrows():
            o = float(row.get('open', row.get('o', 0.0)))
            h = float(row.get('high', row.get('h', 0.0)))
            l = float(row.get('low', row.get('l', 0.0)))
            c = float(row.get('close', row.get('c', 0.0)))
            rng = max(1e-9, h - l)
            body = abs(c - o) / rng if rng > 0 else 0.0

            # bar가 zone 범위와 겹치는지 (엄격)
            bar_overlaps = (h >= zone_low - tol) and (l <= zone_high + tol)

            # proximity 허용: 바 중간이 zone 중심 근처에 있으면 허용
            mid = 0.5 * (h + l)
            prox_tol = max(tol * 1.5, zone_width * 0.5)
            near_zone_prox = abs(mid - zone_mid) <= prox_tol

            in_zone = bar_overlaps or (allow_proximity and near_zone_prox)

            # LVN 근접성 검사
            near_lvn = True
            if lvn_price is not None:
                mid_bar = (l + h) * 0.5
                near_lvn = (abs(lvn_price - mid_bar) <= tol) or (l - tol <= lvn_price <= h + tol)

            # body 기준 약간 완화
            body_ok = body >= max(0.10, body_ratio_min * 0.75)

            # 방향별 핵심 판정 (수정된 비교 기준)
            if direction == 'long':
                # LONG: price가 위에 있을 때, '윅이 아래로 내려가 zone 상단(zone_high)을 찍고' 다시 위로 복귀해야 함
                wick_through = (l <= zone_high + tol) or (lvn_price is not None and l <= lvn_price + tol)
                close_back = (c > o and c >= zone_high - tick) or (c >= zone_high - (0.5 * tick))
                wick_or_close = (wick_through or close_back)
            else:
                # SHORT: 반대 방향
                wick_through = (h >= zone_low - tol) or (lvn_price is not None and h >= lvn_price - tol)
                close_back = (c < o and c <= zone_low + tick) or (c <= zone_low + (0.5 * tick))
                wick_or_close = (wick_through or close_back)

            ok = False

            # 기본 승인: zone 겹침(or prox) + near_lvn + body_ok + wick_or_close
            if in_zone and near_lvn and body_ok and wick_or_close:
                ok = True
            else:
                # 보완 승인: wick/close가 있고 LVN 근접하면 허용 (단, body 완화 또는 volume 조건 필요)
                if allow_wick_only and near_lvn and wick_or_close:
                    vol_ok = True
                    if vol_series is not None and vol_sma is not None and vol_multiplier is not None:
                        sma_v = float(vol_sma.loc[idx])
                        vol_ok = (sma_v > 0) and (float(vol_series.loc[idx]) >= vol_multiplier * sma_v)
                    # 느슨한 body 허용(더 낮게), 또는 volume이 충분하면 허용
                    if (body >= max(0.08, body_ratio_min * 0.6)) and vol_ok:
                        ok = True

            debug_rows.append({
                "idx": idx, "o": o, "h": h, "l": l, "c": c, "body": round(body, 3),
                "zone_low": zone_low, "zone_high": zone_high, "tol": tol,
                "bar_overlaps": bool(bar_overlaps),
                "near_zone_prox": bool(near_zone_prox),
                "in_zone": bool(in_zone),
                "near_lvn": bool(near_lvn),
                "wick_through": bool(wick_through),
                "close_back": bool(close_back),
                "body_ok": bool(body_ok),
                "ok": bool(ok)
            })

            if ok:
                sat += 1

        return sat >= require_k

    def _no_signal_result(self, **kwargs):
        return {
            "name": "VPVR",
            "action": "HOLD",   
            "timestamp": self.tm.get_current_time(),
            "score": 0.0,
            "context": kwargs
        }
    # ===== Public API =====
    
    def evaluate(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Evaluate on the last bar of df. Returns signal dict or None.
        df: OHLCV DataFrame (open, high, low, close, volume[, quote_volume]) in time order.
        """
        gp = getattr(self, 'gp', None)
        if gp is not None:
            # apply more permissive settings for testing only
            try:
                gp.confirm_body_ratio = 0.10
                gp.prox_tol_mult = getattr(gp, 'prox_tol_mult', 1.0) * 2.0
                gp.prox_zone_frac = getattr(gp, 'prox_zone_frac', 0.6)
                gp.require_k = 0
                gp.allow_wick_only = True
                gp.enable_soft_accept = True
                # if getattr(self, 'debug', False):
                #     print('[VPVR RUNTIME TUNE] temporary GP tuning applied: confirm_body_ratio=0.10 prox_tol_mult*=2 require_k=0 allow_wick_only=True')
            except Exception as e:
                print('[VPVR RUNTIME TUNE] failed to apply runtime tuning:', e)
        else:
            if getattr(self, 'debug', False):
                print('[VPVR RUNTIME TUNE] self.gp not found; skipping runtime tuning')
        # --- end runtime tuning block ---
            
        need = max(self.vpvr.lookback_bars, self.gp.swing_lookback) + 5
        if len(df) < need:
            print("len(df) < need")
            return self._no_signal_result()

        df = df.copy()
        df.index = pd.Index(range(len(df)))

        # 1) ATR & tolerance
        atr_last = float(self._atr(df, self.gp.atr_len).iloc[-1])    
        tol = self.gp.tolerance_atr_mult * atr_last

        # 2) VPVR (전역 우선) & LVN
        poc_global, hvn_global, lvn_global = get_vpvr()

        vp = None
        lvns = []
        used_global = False

        if poc_global is not None:
            # 전역 VPVR 사용
            used_global = True
            poc_price = float(poc_global)
            hvn_price = float(hvn_global) if hvn_global is not None else None
            lvn_price = float(lvn_global) if lvn_global is not None else None

            # make a compatible lvns list (index, price, vol) so other functions can use it
            if lvn_price is not None:
                lvns = [(0, lvn_price, 0.0)]
            else:
                lvns = []
        else:
            # fallback: 로컬 VPVR 계산
            vp = self._compute_vpvr(df)
            lvns = self._find_lvn_nodes(vp)
            poc_price = float(vp["poc_price"])
            hvn_price = None
            lvn_price = None

        # 3) Swing & Golden Pocket zone
        swing = self._detect_last_swing(df, self.gp.swing_lookback)
        if swing is None:
            print(f"[VPVR] swing detection failed -> no signal (lookback={self.gp.swing_lookback}, df_len={len(df)})")
            return {
                "name": "VPVR",
                "action": "HOLD",   
                "timestamp": self.tm.get_current_time(),
                "score": 0.0,
            }
        else:
            print(f"[VPVR] swing detected ✓ (swing={swing})")
            
        gp_low, gp_high, direction = self._golden_pocket_zone(df, swing)

        zone_mid = 0.5 * (gp_low + gp_high)
        nearest_lvn = self._nearest_lvn_to_price(lvns, zone_mid) if lvns else None
        lvn_price = nearest_lvn[1] if nearest_lvn else None

        # 4) Volume dry-up
        dryup_result = self._volume_dryup(df, self.gp.dryup_lookback, self.gp.dryup_window, self.gp.dryup_frac, self.gp.dryup_k)
        if not dryup_result:
            print(f"[VPVR] volume dry-up failed -> no signal (lookback={self.gp.dryup_lookback}, window={self.gp.dryup_window}, frac={self.gp.dryup_frac}, k={self.gp.dryup_k})")
            return self._no_signal_result()
        else:
            print(f"[VPVR] volume dry-up passed ✓")

        # 5) Rejection confirmation
                # --- Extra diagnostics: print GP/LVN/tol and df tail for manual inspection ---
        # print("[VPVR DIAG EXTRA] gp_low, gp_high, lvn_price, tol:", gp_low, gp_high, lvn_price, tol)
        # print("[VPVR DIAG EXTRA] gp.confirm_body_ratio, gp.tick:", getattr(self.gp, "confirm_body_ratio", None), getattr(self.gp, "tick", None))
        # print("[VPVR DIAG EXTRA] last 6 candles:\n", df.tail(6)[["open","high","low","close"]])
        # -------------------------------------------------------------------

        # 5) Rejection confirmation (diagnostic + robust multi-level relaxed fallback)
        saved_debug = getattr(self, 'debug', False)
        self.debug = False
        ok = self._rejection_confirm(df, gp_low, gp_high, direction, lvn_price, tol,
                                        self.gp.confirm_body_ratio, self.gp.tick)
        self.debug = saved_debug
        # print(f"[VPVR DIAG] initial rejection_confirm ok={ok}")
        if not ok:
            # print('[VPVR] normal rejection_confirm failed -> trying multi-level relaxed fallbacks')
            # levels: (tol_mult, body_ratio_mult, lookback, require_k)
            relax_levels = [
                (1.5, 0.5, 3, 1),
                (2.0, 0.4, 2, 1),
                (3.0, 0.2, 2, 0),
                (4.0, 0.1, 1, 0),
                (6.0, 0.05, 1, 0)
            ]
            ok_relaxed = False
            for i, (tol_m, body_m, lookback_k, req_k) in enumerate(relax_levels, start=1):
                try:
                    new_tol = tol * tol_m if tol is not None else tol_m
                except Exception:
                    new_tol = tol_m
                new_body = max(0.005, float(self.gp.confirm_body_ratio) * body_m) if hasattr(self.gp, 'confirm_body_ratio') else max(0.005, 0.02 * body_m)
                # print(f"[VPVR RELAX] attempt {i}: tol*{tol_m:.2f} -> {new_tol:.3f}, body*{body_m:.3f} -> {new_body:.4f}, lookback={lookback_k}, require_k={req_k}")
                try:
                    ok_relaxed = self._rejection_confirm(df, gp_low, gp_high, direction, lvn_price, new_tol,
                                                            new_body, self.gp.tick, lookback=lookback_k, require_k=req_k)
                except TypeError:
                    try:
                        ok_relaxed = self._rejection_confirm(df, gp_low, gp_high, direction, lvn_price, new_tol,
                                                                new_body, self.gp.tick)
                    except Exception as e:
                        # print("[VPVR RELAX] _rejection_confirm call failed:", repr(e))
                        ok_relaxed = False
                # print(f"[VPVR RELAX] attempt {i} result: {ok_relaxed}")
                if ok_relaxed:
                    # print('[VPVR] relaxed rejection_confirm succeeded on attempt', i)
                    ok = True
                    break
            if not ok_relaxed:
                print('[VPVR] all relaxed rejection_confirm attempts failed -> no signal')
                return self._no_signal_result()

        # 6) Orders
# 6) Orders
        last = df.iloc[-1]
        h = float(last['high']); l = float(last['low']); c = float(last['close'])

        if direction == 'long':
            entry = h + self.gp.tick
            stop  = min(l, gp_low, (lvn_price if lvn_price is not None else l)) - self.gp.tick
            stop  = min(stop, c - self.risk.stop_atr_mult * atr_last)
            R = entry - stop
            tp1, tp2 = entry + self.risk.tp_R1 * R, entry + self.risk.tp_R2 * R
            action = "BUY"
        else:
            entry = l - self.gp.tick
            stop  = max(h, gp_high, (lvn_price if lvn_price is not None else h)) + self.gp.tick
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
        # --- SCORING (added) ---
        # compute simple, forgiving score using available vars: c, entry, stop, poc_price, gp_low/gp_high, atr_last
        try:
            prev_c = float(df.iloc[-2]['close']) if len(df)>=2 else float(c)
        except Exception:
            prev_c = float(c)
        pct_move = (float(c) - float(prev_c)) / (prev_c if prev_c!=0 else 1.0)
        gp_mid = 0.5 * (float(gp_low) + float(gp_high)) if (gp_low is not None and gp_high is not None) else float(c)
        zone_width = max(1e-6, float(gp_high) - float(gp_low)) if (gp_low is not None and gp_high is not None) else 1.0
        zone_dist = abs(float(c) - gp_mid)
        # normalize by zone width or ATR
        den = max(zone_width, 2.0 * float(atr_last) if atr_last is not None else zone_width)
        zone_comp = max(0.0, 1.0 - (zone_dist / den))
        # poc proximity
        try:
            poc_comp = 1.0 - (abs(float(c) - float(poc_price)) / (4.0 * float(atr_last))) if poc_price is not None else 0.5
        except Exception:
            poc_comp = 0.5
        poc_comp = max(0.0, min(1.0, poc_comp))
        # lvn proximity
        try:
            lvn_comp = 1.0 - (abs(float(c) - float(lvn_price)) / (6.0 * float(atr_last))) if lvn_price is not None else 0.5
        except Exception:
            lvn_comp = 0.5
        lvn_comp = max(0.0, min(1.0, lvn_comp))
        # pct comp (more sensitive)
        pct_comp = min(1.0, abs(pct_move) / 0.002)
        # relaxed_level if present in local scope
        relaxed_level = locals().get('i', None)
        if relaxed_level is None:
            relaxed_level = 0
        # small penalty for high relaxation
        relax_pen = max(0.0, 1.0 - 0.15 * float(relaxed_level))
        # weights
        W = {'zone':0.30, 'poc':0.25, 'lvn':0.15, 'pct':0.20, 'relax':0.10}
        total_w = sum(W.values()) or 1.0
        raw = W['zone']*zone_comp + W['poc']*poc_comp + W['lvn']*lvn_comp + W['pct']*pct_comp + W['relax']*relax_pen
        score = float(raw) / float(total_w)
        # floor if heavily relaxed
        if relaxed_level and score < 0.10:
            score = 0.10

        components = {'zone':round(zone_comp,3),'poc':round(poc_comp,3),'lvn':round(lvn_comp,3),'pct':round(pct_comp,3),'relax_level':int(relaxed_level)}
        # print(f"[VPVR_SCORE] score={score:.3f} comps={components} pct_move={pct_move:.4f} relaxed_level={relaxed_level}")
        # attach to ctx later by injecting into result
        # --- end SCORING ---

        result = {
            "name": "VPVR",
            "action": action,   
            "entry": float(entry),
            "stop": float(stop),
            "score": score,
            "targets": [float(tp1), float(tp2), float(poc_price)],
            "context": ctx
        }

        # attach scoring metadata
        try:
            result['score'] = float(score)
            result['components'] = components
        except Exception:
            result['score'] = None
            result['components'] = None

        return result

