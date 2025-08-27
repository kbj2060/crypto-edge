
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
        dryup_window: int = 4
        dryup_frac: float = 0.6
        tolerance_atr_mult: float = 0.3
        confirm_body_ratio: float = 0.3
        atr_len: int = 14
        tick: float = 0.1

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

    def _volume_dryup(self, df: pd.DataFrame, sma_len: int, window: int, dry_frac: float) -> bool:
        if 'quote_volume' in df.columns:
            v = df['quote_volume'].astype(float)
        else:
            v = (df['volume'] * df['close']).astype(float)
        sma = v.rolling(sma_len, min_periods=1).mean()
        last = df.tail(window).index
        conds = (v.loc[last] <= dry_frac * sma.loc[last]).to_list()
        return all(bool(x) for x in conds)

    def _rejection_confirm(self, df: pd.DataFrame, zone_low: float, zone_high: float, direction: str,
                            lvn_price: Optional[float], tol: float, body_ratio_min: float, tick: float) -> bool:
        last = df.iloc[-1]
        o = float(last['open']); h = float(last['high']); l = float(last['low']); c = float(last['close'])
        rng = max(1e-9, h - l)
        body = abs(c - o) / rng

        in_zone = (min(h, c) >= zone_low - tol) and (max(l, c) <= zone_high + tol) or                   (l <= zone_high + tol and h >= zone_low - tol)

        near_lvn = True
        if lvn_price is not None:
            near_lvn = (abs(lvn_price - (l + h) * 0.5) <= tol) or (l - tol <= lvn_price <= h + tol)

        if direction == 'long':
            wick_through = (l <= zone_low - tick) or (lvn_price is not None and l <= lvn_price - tick)
            close_back   = c > o and c >= zone_low - tick
            return in_zone and near_lvn and wick_through and close_back and (body >= body_ratio_min)
        else:
            wick_through = (h >= zone_high + tick) or (lvn_price is not None and h >= lvn_price + tick)
            close_back   = c < o and c <= zone_high + tick
            return in_zone and near_lvn and wick_through and close_back and (body >= body_ratio_min)

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
        atr_series = self._atr(df, self.gp.atr_len)
        atr_last = float(atr_series.iloc[-1])
        tol = self.gp.tolerance_atr_mult * atr_last
        print(f"📊 [VPVR] ATR: {atr_last:.4f}, 허용오차: {tol:.4f} (ATR × {self.gp.tolerance_atr_mult})")

        # 2) VPVR & LVN
        print(f"📊 [VPVR] VPVR 계산 시작 - 룩백: {self.vpvr.lookback_bars}")
        vp = self._compute_vpvr(df)
        print(f"📊 [VPVR] VPVR 완료 - 가격범위: ${vp['pmin']:.2f}~${vp['pmax']:.2f}, POC: ${vp['poc_price']:.2f}")
        
        lvns = self._find_lvn_nodes(vp)
        print(f"📊 [VPVR] LVN 노드 발견: {len(lvns)}개")
        if lvns:
            for i, (idx, price, vol) in enumerate(lvns[:3]):  # 처음 3개만 출력
                print(f"   📍 LVN{i+1}: ${price:.2f} (인덱스: {idx}, 볼륨: {vol:.0f})")

        poc, hvn, lvn = global_indicators.get_vpvr()
        print(f"📊 [GLOBAL VPVR] POC: ${poc:.2f}, HVN: ${hvn:.2f}, LVN: ${lvn:.2f}")

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
        lvn = self._nearest_lvn_to_price(lvns, zone_mid) if lvns else None
        lvn_price = lvn[1] if lvn else None
        
        print(f"📊 [VPVR] 골든 포켓 중간: ${zone_mid:.2f}")
        if lvn:
            print(f"📊 [VPVR] 가장 가까운 LVN: ${lvn_price:.2f}")
        else:
            print(f"📊 [VPVR] LVN이 골든 포켓 근처에 없습니다")

        # 4) Volume dry-up
        print(f"📊 [VPVR] 볼륨 드라이업 검사 - 룩백: {self.gp.dryup_lookback}, 윈도우: {self.gp.dryup_window}, 임계값: {self.gp.dryup_frac}")
        if not self._volume_dryup(df, self.gp.dryup_lookback, self.gp.dryup_window, self.gp.dryup_frac):
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
            "poc_price": float(vp["poc_price"]),
            "atr": float(atr_last),
            "tol_atr_mult": float(self.gp.tolerance_atr_mult),
            "dryup": {
                "lookback": self.gp.dryup_lookback,
                "window": self.gp.dryup_window,
                "frac": self.gp.dryup_frac
            },
            "vpvr": {
                "bin_size": float(vp["bin_size"]),
                "lookback_bars": self.vpvr.lookback_bars
            }
        }

        result = {
            "stage": "ENTRY",
            "action": action,
            "entry": float(entry),
            "stop": float(stop),
            "targets": [float(tp1), float(tp2), float(vp["poc_price"])],
            "context": ctx
        }
        
        print(f"🎯 [VPVR] 전략 신호 생성 완료!")
        print(f"   📊 액션: {action}")
        print(f"   📊 진입가: ${result['entry']:.2f}")
        print(f"   📊 스탑로스: ${result['stop']:.2f}")
        print(f"   📊 목표가: TP1=${result['targets'][0]:.2f}, TP2=${result['targets'][1]:.2f}, POC=${result['targets'][2]:.2f}")
        
        return result
