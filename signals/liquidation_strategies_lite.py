# liquidation_strategies_lite.py
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from math import sqrt

from utils.time_manager import get_time_manager
from indicators.global_indicators import get_atr, get_vwap
from data.data_manager import get_data_manager

def _usd_from_event(ev: Dict[str, Any]) -> Tuple[str, float]:
    side = str(ev.get('side', '')).lower()
    size = float(ev.get('size', 0))
    price = float(ev.get('price', 0))
    if 'qty_usd' in ev and ev['qty_usd'] is not None:
        usd = float(ev['qty_usd'])
    else:
        usd = size * price
    return side, usd

@dataclass
class BaseLiqConfig:
    lookback_buckets: int = 240
    recency_sec: int = 600     # 더 민감: 최근성 체크 강화
    tick: float = 0.05        # 선물 시장에 맞춰 tick 작게


@dataclass
class FadeConfig(BaseLiqConfig):
    agg_window_sec: int = 60
    min_bucket_notional_usd: float = 4000.0  # 더 작은 청산도 감지
    z_setup: float = 1.0                        # z 기준 완화 (민감)
    lpi_min: float = 0.03                       # LPI 문턱 낮춤
    setup_ttl_min: int = 15                      # SETUP TTL 단축
    vwap_sigma_entry: float = 1.0              # 엔트리 시그마 완화
    atr_stop_mult: float = 0.5                  # 스탑 타이트닝
    tp_R1: float = 0.9
    tp_R2: float = 1.6

class FadeReentryStrategy:
    def __init__(self, cfg: FadeConfig = FadeConfig()):
        self.cfg = cfg
        self.time_manager = get_time_manager()
        self.long_hist = deque(maxlen=cfg.lookback_buckets)
        self.short_hist = deque(maxlen=cfg.lookback_buckets)
        self.mu_long = 0.0; self.sd_long = 1.0
        self.mu_short = 0.0; self.sd_short = 1.0
        self.bucket_log: List[Tuple[datetime, float, float, float]] = []
        self.pending_setup: Optional[Dict[str, Any]] = None

    def _update_stats(self, long_usd: float, short_usd: float) -> None:
        self.long_hist.append(float(long_usd))
        self.short_hist.append(float(short_usd))
        if len(self.long_hist) >= 20:  # 더 빠르게 통계 반영
            self.mu_long = float(np.mean(self.long_hist))
            self.mu_short = float(np.mean(self.short_hist))
            self.sd_long = float(np.std(self.long_hist, ddof=1)) or 1.0
            self.sd_short = float(np.std(self.short_hist, ddof=1)) or 1.0

    def _z_lpi(self, long_usd: float, short_usd: float) -> Tuple[float, float, float]:
        zL_raw = (long_usd - self.mu_long) / max(self.sd_long, 1e-9)
        zS_raw = (short_usd - self.mu_short) / max(self.sd_short, 1e-9)
        zL = max(0.0, zL_raw); zS = max(0.0, zS_raw)
        total = long_usd + short_usd
        lpi = (short_usd - long_usd) / (total + 1e-9)
        return zL, zS, lpi

    def warmup(self, bucket_events: List[Dict[str, Any]]) -> None:
        self.bucket_log = []
        for ev in bucket_events:
            side, usd = _usd_from_event(ev)
            timestamp = self.time_manager.get_timestamp_datetime(ev['timestamp'])
            if side in ('sell', 'long'):
                self._update_stats(float(usd), 0.0)
                self.bucket_log.append((timestamp, float(usd), 0.0, float(usd)))
            elif side in ('buy', 'short'):
                self._update_stats(0.0, float(usd))
                self.bucket_log.append((timestamp, 0.0, float(usd), float(usd)))

    def on_bucket_close(self, bucket_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        now = self.time_manager.get_current_time()
        if not bucket_events:
            self._update_stats(0.0, 0.0)
            return None
        long_usd = short_usd = 0.0
        for ev in bucket_events:
            side, usd = _usd_from_event(ev)
            if side in ('sell', 'long'):
                long_usd += usd
            elif side in ('buy', 'short'):
                short_usd += usd
        total = long_usd + short_usd
        zL, zS, lpi = self._z_lpi(long_usd, short_usd)
        self._update_stats(long_usd, short_usd)
        self.bucket_log.append((now, long_usd, short_usd, total))
        max_z = max(zL, zS)
        if (max_z < self.cfg.z_setup) or (abs(lpi) < self.cfg.lpi_min):
            return None
        side = 'BUY' if zL > zS else 'SELL'
        self.pending_setup = {
            'side': side, 'created': now,
            'expires': now + timedelta(minutes=self.cfg.setup_ttl_min),
            'z': float(max_z), 'lpi': float(lpi), 'bucket_total_usd': float(total)
        }
        return {"stage":"SETUP","action":side,"z":float(max_z),"lpi":float(lpi),
                "bucket_total_usd":float(total),"created":now.isoformat()}

    def on_kline_close_3m(self) -> Optional[Dict[str, Any]]:
        data_manager = get_data_manager()
        df_3m = data_manager.get_latest_data(count=2)
        if df_3m is None or len(df_3m) < 2:
            print(f"🔍 [FADE] 데이터 부족: 필요한 데이터 길이={2}")
            return None

        now = self.time_manager.get_current_time()
        vwap, vwap_std = get_vwap()
        atr = get_atr()
        ps = self.pending_setup
        if not ps or now > ps['expires']:
            print(f"{ps} expired")
            return None

        age = (now - self.bucket_log[-1][0]).total_seconds() if self.bucket_log else None
        if age is None or age > self.cfg.recency_sec:
            print(f"{age} > {self.cfg.recency_sec}")
            return None

        # get OHLC
        prev_c = float(df_3m["close"].iloc[-2])
        prev_low = float(df_3m["low"].iloc[-2])
        prev_high = float(df_3m["high"].iloc[-2])

        last_h = float(df_3m["high"].iloc[-1])
        last_l = float(df_3m["low"].iloc[-1])
        last_c = float(df_3m["close"].iloc[-1])

        # sigma / threshold
        n = float(self.cfg.vwap_sigma_entry) if hasattr(self.cfg, "vwap_sigma_entry") else float(getattr(self.cfg, "cont_range_atr", 1.0))
        # defensive vwap_std
        try:
            vwap_val = float(vwap) if vwap is not None else None
        except Exception:
            vwap_val = None
        try:
            vwap_std_val = float(vwap_std) if vwap_std is not None else None
        except Exception:
            vwap_std_val = None

        # compute a base pct move (fallback)
        pct_move = (last_c - prev_c) / (prev_c if prev_c > 0 else 1.0)
        pct_ok = pct_move >= 0.002  # 0.2% upward move qualifies as reentry support

        # compute thresholds (if vwap available). If not available, rely on pct_ok/wick logic.
        if vwap_val is not None and vwap_std_val is not None:
            threshold_buy = vwap_val - n * vwap_std_val
            threshold_sell = vwap_val + n * vwap_std_val
        else:
            threshold_buy = threshold_sell = None

        # tolerance to avoid tiny noise false negatives
        tol_pct = 0.00025
        tol = max((abs(threshold_buy or threshold_sell) * tol_pct), 0.5)

        # default vol_ok (if you have a vol check earlier, you can replace this)
        vol_ok = True

        # Decide reentry robustly
        reentry = False
        if ps['side'] == 'BUY':
            print(f"prev_c={prev_c}, vwap={vwap_val}, vwap_std={vwap_std_val}, n={n}, last_c={last_c}")
            if threshold_buy is None:
                # fallback: rely on percent move or wick behavior
                reentry = pct_ok or (last_l <= prev_l if (prev_l := prev_low) is not None else False)
            else:
                # close-based crossing (prev closed below threshold_prev and last closed above threshold_now)
                close_cross = (prev_c <= (threshold_buy - tol)) and (last_c >= (threshold_buy - tol))
                # wick cross: either previous or last low touched/breached threshold (tail)
                wick_cross = (prev_low <= (threshold_buy + tol)) or (last_l <= (threshold_buy + tol))
                # allow reentry if close_cross or wick_cross or enough pct move (or volume support)
                reentry = (close_cross or wick_cross or pct_ok or vol_ok)
                # For stricter policy you can require (close_cross or (wick_cross and pct_ok))
            if not reentry:
                print(f"{reentry} not reentry (close_cross={locals().get('close_cross', None)}, wick_cross={locals().get('wick_cross', None)}, pct_ok={pct_ok}, tol={tol})")
                return None

            entry = last_h + self.cfg.tick
            stop = min(last_l, last_c - self.cfg.atr_stop_mult * float(atr)) - self.cfg.tick
            R = entry - stop
            tp1, tp2 = entry + self.cfg.tp_R1 * R, entry + self.cfg.tp_R2 * R

        else:
            print(f"prev_c={prev_c}, vwap={vwap_val}, vwap_std={vwap_std_val}, n={n}, last_c={last_c}")
            if threshold_sell is None:
                reentry = (pct_move <= -0.002) or (last_h >= prev_high)
            else:
                close_cross = (prev_c >= (threshold_sell + tol)) and (last_c <= (threshold_sell + tol))
                wick_cross = (prev_high >= (threshold_sell - tol)) or (last_h >= (threshold_sell - tol))
                reentry = (close_cross or wick_cross or (pct_move <= -0.002) or vol_ok)
            if not reentry:
                print(f"{reentry} not reentry (close_cross={locals().get('close_cross', None)}, wick_cross={locals().get('wick_cross', None)}, pct_move={pct_move}, tol={tol})")
                return None

            entry = last_l - self.cfg.tick
            stop = max(last_h, last_c + self.cfg.atr_stop_mult * float(atr)) + self.cfg.tick
            R = stop - entry
            tp1, tp2 = entry - self.cfg.tp_R1 * R, entry - self.cfg.tp_R2 * R

        # clear pending setup and return structured signal
        return {
            "action": ps['side'],
            "entry": float(entry),
            "stop": float(stop),
            "targets": [float(tp1), float(tp2)],
            "context": {
                "mode": "LIQ_FADE",
                "z": ps.get('z'),
                "lpi": ps.get('lpi'),
                "vwap": float(vwap_val) if vwap_val is not None else None,
                "vwap_std": float(vwap_std_val) if vwap_std_val is not None else None,
                "atr": float(atr) if atr is not None else None,
                "bucket_total_usd": ps.get("bucket_total_usd", None)
            }
        }

@dataclass
class MomentumConfig(BaseLiqConfig):
    cascade_dir_share: float = 0.7
    cascade_z3: float = 2.0
    cont_sigma: float = 0.6
    cont_range_atr: float = 0.6
    vol_mult: Optional[float] = 1.0
    tp_R1: float = 0.9
    tp_R2: float = 1.6
    atr_stop_mult: float = 0.9
    enable_fast_1m: bool = True
    fast_minutes: int = 2
    fast_dir_share: float = 0.45
    fast_zN: float = 1.5
    fast_sigma: float = 0.5
    fast_range_atr1m: float = 0.25

class SqueezeMomentumStrategy:
    def __init__(self, cfg: MomentumConfig):
        self.cfg = cfg
        self.time_manager = get_time_manager()
        self.long_hist = deque(maxlen=cfg.lookback_buckets)
        self.short_hist = deque(maxlen=cfg.lookback_buckets)
        self.mu_long = 0.0; self.sd_long = 1.0
        self.mu_short = 0.0; self.sd_short = 1.0
        self.bucket_log: List[Tuple[datetime, float, float, float]] = []
        self.prev_1m = None

    def _update_stats(self, long_usd: float, short_usd: float) -> None:
        self.long_hist.append(float(long_usd))
        self.short_hist.append(float(short_usd))
        if len(self.long_hist) >= 20:
            self.mu_long = float(np.mean(self.long_hist))
            self.mu_short = float(np.mean(self.short_hist))
            self.sd_long = float(np.std(self.long_hist, ddof=1)) or 1.0
            self.sd_short = float(np.std(self.short_hist, ddof=1)) or 1.0

    def _recent_nonempty_bucket_age(self, now_utc: datetime) -> Optional[float]:
        if not self.bucket_log:
            print("no bucket log")
            return None
        return (now_utc - self.bucket_log[-1][0]).total_seconds()

    def _lastN(self, now: datetime, minutes: int) -> List[Tuple[datetime, float, float, float]]:
        cut = now - timedelta(minutes=minutes)
        return [b for b in self.bucket_log if b[0] > cut]

    def _zN(self, L: float, S: float, N: int) -> Tuple[float, float]:
        muL = N * self.mu_long; muS = N * self.mu_short
        sdL = (N ** 0.5) * self.sd_long; sdS = (N ** 0.5) * self.sd_short
        zL = max(0.0, (L - muL) / max(sdL, 1e-9))
        zS = max(0.0, (S - muS) / max(sdS, 1e-9))
        return zL, zS

    def warmup(self, bucket_events: List[Dict[str, Any]]) -> None:
        self.bucket_log = []
        for ev in bucket_events:
            side, usd = _usd_from_event(ev)
            timestamp = self.time_manager.get_timestamp_datetime(ev['timestamp'])
            if side in ('sell', 'long'):
                self._update_stats(float(usd), 0.0)
                self.bucket_log.append((timestamp, float(usd), 0.0, float(usd)))
            elif side in ('buy', 'short'):
                self._update_stats(0.0, float(usd))
                self.bucket_log.append((timestamp, 0.0, float(usd), float(usd)))

    def on_bucket_close(self, bucket_events: List[Dict[str, Any]]) -> None:
        now = self.time_manager.get_current_time()
        if not bucket_events:
            self._update_stats(0.0, 0.0)
            return None
        long_usd = short_usd = 0.0
        for ev in bucket_events:
            side, usd = _usd_from_event(ev)
            if side in ('sell', 'long'):
                long_usd += usd
            elif side in ('buy', 'short'):
                short_usd += usd
        total = long_usd + short_usd
        self._update_stats(long_usd, short_usd)
        self.bucket_log.append((now, long_usd, short_usd, total))
        return None
    
    def on_kline_close_1m(self, df_1m: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        더 관대하고 디버깅 출력이 풍부한 1분 fast squeeze handler.
        - 여러 하위 조건들을 개별적으로 검사하고 로그로 출력.
        - OR 기반 완화 룰을 적용해서 한두 조건만 충족해도 진행(단, low_confidence 표시).
        """
        if not getattr(self.cfg, "enable_fast_1m", False):
            return None
        now = self.time_manager.get_current_time()
        if df_1m is None or len(df_1m) < 1:
            return None

        # indicators
        try:
            vwap, vwap_std = get_vwap()
        except Exception:
            vwap, vwap_std = (None, None)
        atr_3m = get_atr()
        atr1m = float(atr_3m) / sqrt(3.0) if atr_3m else 0.0

        # bucket recency & counts
        age = self._recent_nonempty_bucket_age(now)
        lastN = self._lastN(now, minutes=int(self.cfg.fast_minutes))
        L = sum(b[1] for b in lastN) if lastN else 0.0
        S = sum(b[2] for b in lastN) if lastN else 0.0
        T = L + S

        # safe prev/last handling
        if self.prev_1m is None:
            self.prev_1m = df_1m.iloc[-1]
            # first call after init -> no signal
            return None
        prev = self.prev_1m
        last = df_1m.iloc[-1]
        self.prev_1m = last

        last_o, last_h, last_l, last_c = float(last['open']), float(last['high']), float(last['low']), float(last['close'])
        prev_c = float(prev['close'])
        prev_h = float(prev['high']); prev_l = float(prev['low'])

        # metrics
        share = max(L, S) / (T if T > 0 else 1.0)
        zL, zS = self._zN(L, S, int(self.cfg.fast_minutes))
        side = 'BUY' if S > L else 'SELL'
        zN = zS if side == 'BUY' else zL
        sigma = float(self.cfg.fast_sigma)

        # --- individual condition checks ---
        # 1) vwap condition (original style)
        vwap_cond = False
        if vwap is not None and vwap_std is not None:
            if side == 'BUY':
                vwap_cond = last_c > (vwap + sigma * vwap_std)
            else:
                vwap_cond = last_c < (vwap - sigma * vwap_std)

        # 2) higher-high / lower-low momentum
        hh_cond = last_h > prev_h if side == 'BUY' else last_l < prev_l

        # 3) rng_ok (enough candle size relative to atr1m) - optional fallback True if atr missing
        if atr1m and atr1m > 0:
            rng_ok = (last_h - last_l) >= max(1e-9, self.cfg.fast_range_atr1m * atr1m)
        else:
            rng_ok = True

        # 4) simple momentum (last close moved in direction)
        mom_cond = (last_c > prev_c) if side == 'BUY' else (last_c < prev_c)

        # 5) percent-move absolute threshold (useful when vwap_std is large)
        price = last_c if last_c > 0 else 1.0
        pct_move = abs((last_c - prev_c) / price)
        pct_cond = pct_move >= 0.002  # 0.3% 기본 허용 (민감하게 조절 가능)

        # 6) share / zN checks (stat thresholds)
        share_ok = share >= getattr(self.cfg, "fast_dir_share", 0.45)
        z_ok = zN >= getattr(self.cfg, "fast_zN", 1.5)

        # --- final cont decision (관대화 로직) ---
        # 우선: (vwap_cond AND rng_ok) -> 강한 신호
        strong = vwap_cond and rng_ok
        # 관대 조건: 모멘텀 OR 퍼센트 무브 OR higher-high
        relaxed = mom_cond or pct_cond or hh_cond
        # If either strong or relaxed true -> allow (but if share/zN 부족하면 low_confidence)
        cont = strong or relaxed

        low_confidence = False
        if cont:
            if not (share_ok and z_ok):
                # 그래도 share/zN이 부족하면 허용하되 low_confidence 플래그
                low_confidence = True
        else:
            # 완전히 실패하면 빠져나감
            return None

        # --- build entry/stop/targets (기존 로직) ---
        if side == 'BUY':
            entry = last_h + self.cfg.tick
            stop = max(last_l, prev_l) - self.cfg.tick
            R = entry - stop
            tp1, tp2 = entry + self.cfg.tp_R1 * R, entry + self.cfg.tp_R2 * R
        else:
            entry = last_l - self.cfg.tick
            stop = min(last_h, prev_h) + self.cfg.tick
            R = stop - entry
            tp1, tp2 = entry - self.cfg.tp_R1 * R, entry - self.cfg.tp_R2 * R

        result = {
            "stage": "ENTRY",
            "action": side,
            "entry": float(entry),
            "stop": float(stop),
            "targets": [float(tp1), float(tp2)],
            "context": {
                "mode": "LIQ_SQUEEZE_FAST_1M",
                "minutes": int(self.cfg.fast_minutes),
                "share": float(share),
                "zN": float(zN),
                "vwap": float(vwap) if vwap is not None else None,
                "vwap_std": float(vwap_std) if vwap_std is not None else None,
                "atr1m": float(atr1m)
            },
            "low_confidence": low_confidence,
            "debug": {
                "vwap_cond": bool(vwap_cond),
                "hh_cond": bool(hh_cond),
                "rng_ok": bool(rng_ok),
                "mom_cond": bool(mom_cond),
                "pct_move": float(pct_move)
            }
        }

        return result
    