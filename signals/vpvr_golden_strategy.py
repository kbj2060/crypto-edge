from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from indicators import global_indicators

class LVNGoldenPocket:
    """
    VPVR LVN Rejection + Golden Pocket (0.618~0.65) + Volume Dry-up
    - All logic is encapsulated in this single class.
    - Pass a pandas OHLCV DataFrame to `evaluate(df)` to get a signal dict or None.
    """

    @dataclass
    class VPVRConfig:
        lookback_bars: int = 400
        bin_size: Optional[float] = None
        max_bins: int = 80
        use_quote_volume: bool = True
        min_price_tick: float = 0.1

    @dataclass
    class LVNSettings:
        low_percentile: float = 0.30
        local_min: bool = True
        merge_neighbors: bool = True
        merge_ticks: int = 3

    @dataclass
    class GoldenPocketCfg:
        swing_lookback: int = 180
        dryup_lookback: int = 20
        dryup_window: int = 5         # 4 -> 5
        dryup_frac: float = 0.9      # 0.6 -> 0.75 (완화)
        dryup_k: int = 3              # 최근 N봉 중 최소 k개 만족
        tolerance_atr_mult: float = 0.6  # 0.3 -> 0.5 (완화)
        confirm_body_ratio: float = 0.18 # 0.3 -> 0.25 (조금 완화)
        atr_len: int = 14
        tick: float = 0.1
        lvn_max_atr: float = 4.0      # LVN이 GP중앙에서 4×ATR 이내면 LVN 인정
        confirm_mode: str = "wick_or_break"  # 'wick' | 'break' | 'wick_or_break'
        zone_widen_atr: float = 0.3   # GP 존을 ±(0.2×ATR) 만큼 넓혀 허용

    @dataclass
    class TargetsStopsCfg:
        stop_atr_mult: float = 0.8
        tp_R1: float = 1.2
        tp_R2: float = 2.0

    def __init__(
        self,
        vpvr: Optional['LVNGoldenPocket.VPVRConfig'] = None,
        lvn: Optional['LVNGoldenPocket.LVNSettings'] = None,
        gp: Optional['LVNGoldenPocket.GoldenPocketCfg'] = None,
        risk: Optional['LVNGoldenPocket.TargetsStopsCfg'] = None,
    ):
        self.vpvr = vpvr or LVNGoldenPocket.VPVRConfig()
        self.lvn  = lvn  or LVNGoldenPocket.LVNSettings()
        self.gp   = gp   or LVNGoldenPocket.GoldenPocketCfg()
        self.risk = risk or LVNGoldenPocket.TargetsStopsCfg()

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
        if not lvns: return None
        i = int(np.argmin([abs(p - price) for _, p, _ in lvns]))
        return lvns[i]

    def _detect_last_swing(self, df: pd.DataFrame, lookback: int) -> Optional[Tuple[int, int]]:
        if len(df) < lookback + 1:
            lookback = len(df) - 1
        if lookback < 10:
            return None
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
        # require at least dry_k successes
        return sat >= max(1, min(dry_k, window))


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

        # 디버그 출력 (운영시 주석 처리 가능)
        print(f"DEBUG: sat={sat}/{lookback}, require_k={require_k}")

        return sat >= require_k


    # ===== Public API =====

    def evaluate(self, df: pd.DataFrame, now_utc: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Evaluate on the last bar of df. Returns signal dict or None.
        df: OHLCV DataFrame (open, high, low, close, volume[, quote_volume]) in time order.
        """
        print(f"🔍 [VPVR] 전략 평가 시작 - 데이터 길이: {len(df) if df is not None else 'None'}")
        
        if df is None:
            print(f"❌ [VPVR] 데이터가 None입니다")
            return None
            
        need = max(self.vpvr.lookback_bars, self.gp.swing_lookback) + 5
        if len(df) < need:
            print(f"⚠️ [VPVR] 데이터 부족: {len(df)} < {need} (필요: {need})")
            return None
            
        print(f"✅ [VPVR] 데이터 검증 통과 - 길이: {len(df)}")

        df = df.copy()
        df.index = pd.Index(range(len(df)))

        # 1) ATR & tolerance
        print(f"📊 [VPVR] ATR 계산 시작 - 기간: {self.gp.atr_len}")
        atr_last = global_indicators.get_atr()
        if atr_last is None:
            print("⚠️ ATR 값을 가져올 수 없습니다. (global_indicators.get_atr() 반환값 None)")
            atr_last = float(self._atr(df, self.gp.atr_len).iloc[-1])
        tol = self.gp.tolerance_atr_mult * atr_last
        print(f"📊 [VPVR] ATR: {atr_last:.4f}, 허용오차: {tol:.4f} (ATR × {self.gp.tolerance_atr_mult})")

        # 2) VPVR (전역 우선) & LVN
        print(f"📊 [VPVR] 전역 VPVR 조회 시도...")
        try:
            poc_global, hvn_global, lvn_global = global_indicators.get_vpvr()
        except Exception as e:
            print(f"⚠️ 전역 VPVR 조회 중 오류: {e}")
            poc_global, hvn_global, lvn_global = (None, None, None)

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

            print(f"📊 [GLOBAL VPVR] POC: ${poc_price:.2f}, HVN: {hvn_price if hvn_price else 'N/A'}, LVN: {lvn_price if lvn_price else 'N/A'} (전역 사용)")
        else:
            # fallback: 로컬 VPVR 계산
            print(f"📊 [VPVR] 전역 VPVR 없음 — 로컬 VPVR 계산 (폴백)")
            vp = self._compute_vpvr(df)
            lvns = self._find_lvn_nodes(vp)
            poc_price = float(vp["poc_price"])
            hvn_price = None
            lvn_price = None
            print(f"📊 [LOCAL VPVR] POC: ${poc_price:.2f}, LVN count: {len(lvns)}")

        # 3) Swing & Golden Pocket zone
        print(f"📊 [VPVR] 스윙 분석 시작 - 룩백: {self.gp.swing_lookback}")
        swing = self._detect_last_swing(df, self.gp.swing_lookback)
        if swing is None:
            print(f"❌ [VPVR] 스윙 패턴을 찾을 수 없습니다")
            return None
            
        gp_low, gp_high, direction = self._golden_pocket_zone(df, swing)
        print(f"📊 [VPVR] 골든 포켓 구간: {direction.upper()} - ${gp_low:.2f}~${gp_high:.2f}")
        print(f"   📍 스윙: 인덱스 {swing[0]} → {swing[1]}")

        zone_mid = 0.5 * (gp_low + gp_high)
        nearest_lvn = self._nearest_lvn_to_price(lvns, zone_mid) if lvns else None
        lvn_price = nearest_lvn[1] if nearest_lvn else None
        
        print(f"📊 [VPVR] 골든 포켓 중간: ${zone_mid:.2f}")
        if nearest_lvn:
            print(f"📊 [VPVR] 가장 가까운 LVN: ${lvn_price:.2f}")
        else:
            print(f"📊 [VPVR] LVN이 골든 포켓 근처에 없습니다")

        # 4) Volume dry-up
        print(f"📊 [VPVR] 볼륨 드라이업 검사 - 룩백: {self.gp.dryup_lookback}, 윈도우: {self.gp.dryup_window}, 임계값: {self.gp.dryup_frac}")
        if not self._volume_dryup(df, self.gp.dryup_lookback, self.gp.dryup_window, self.gp.dryup_frac, self.gp.dryup_k):
            print(f"❌ [VPVR] 볼륨 드라이업 조건 불만족")
            return None
        print(f"✅ [VPVR] 볼륨 드라이업 조건 만족")

        # 5) Rejection confirmation
        print(f"📊 [VPVR] 거부 확인 검사 - 방향: {direction.upper()}, 바디 비율 최소: {self.gp.confirm_body_ratio}")
        if not self._rejection_confirm(df, gp_low, gp_high, direction, lvn_price, tol,
                                        self.gp.confirm_body_ratio, self.gp.tick):
            print(f"❌ [VPVR] 거부 확인 조건 불만족")
            return None
        print(f"✅ [VPVR] 거부 확인 조건 만족")

        # 6) Orders
        print(f"📊 [VPVR] 주문 계산 시작 - 방향: {direction.upper()}")
        last = df.iloc[-1]
        h = float(last['high']); l = float(last['low']); c = float(last['close'])
        print(f"   📍 마지막 캔들: H=${h:.2f}, L=${l:.2f}, C=${c:.2f}")

        if direction == 'long':
            entry = h + self.gp.tick
            stop  = min(l, gp_low, (lvn_price if lvn_price is not None else l)) - self.gp.tick
            stop  = min(stop, c - self.risk.stop_atr_mult * atr_last)
            R = entry - stop
            tp1, tp2 = entry + self.risk.tp_R1 * R, entry + self.risk.tp_R2 * R
            action = "BUY"
            print(f"   📍 롱 진입: ${entry:.2f}, 스탑: ${stop:.2f}, 리스크: ${R:.2f}")
            print(f"   📍 목표: TP1=${tp1:.2f}, TP2=${tp2:.2f}")
        else:
            entry = l - self.gp.tick
            stop  = max(h, gp_high, (lvn_price if lvn_price is not None else h)) + self.gp.tick
            stop  = max(stop, c + self.risk.stop_atr_mult * atr_last)
            R = stop - entry
            tp1, tp2 = entry - self.risk.tp_R1 * R, entry - self.risk.tp_R2 * R
            action = "SELL"
            print(f"   📍 숏 진입: ${entry:.2f}, 스탑: ${stop:.2f}, 리스크: ${R:.2f}")
            print(f"   📍 목표: TP1=${tp1:.2f}, TP2=${tp2:.2f}")

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
        
        print(f"🎯 [VPVR] 전략 신호 생성 완료!")
        print(f"   📊 액션: {action}")
        print(f"   📊 진입가: ${result['entry']:.2f}")
        print(f"   📊 스탑로스: ${result['stop']:.2f}")
        print(f"   📊 목표가: TP1=${result['targets'][0]:.2f}, TP2=${result['targets'][1]:.2f}, POC=${result['targets'][2]:.2f}")
        
        return result
