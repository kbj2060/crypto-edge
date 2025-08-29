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
    recency_sec: int = 120     # relaxed: allow slightly older buckets to count
    tick: float = 0.02        # smaller tick for futures

@dataclass
class FadeConfig(BaseLiqConfig):
    agg_window_sec: int = 60
    min_bucket_notional_usd: float = 2000.0  # detect smaller liquidations (more permissive)
    z_setup: float = 0.9                       # relaxed z threshold (more sensitive)
    lpi_min: float = 0.02                      # lower LPI threshold
    setup_ttl_min: int = 8                     # longer TTL to reduce missed entries
    vwap_sigma_entry: float = 0.9              # looser sigma for re-entry detection
    atr_stop_mult: float = 0.7                 # slightly larger stop multiplier (or keep tight based on risk)
    tp_R1: float = 0.9
    tp_R2: float = 1.6

class FadeReentryStrategy:
    def __init__(self, cfg: FadeConfig):
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
        if len(self.long_hist) >= 20:  # keep relatively small window for quicker adaptation
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
        # relaxed criteria: smaller z and lpi still allowed
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
            return None
        now = self.time_manager.get_current_time()
        vwap, vwap_std = get_vwap()
        atr = get_atr()
        ps = self.pending_setup
        if not ps or now > ps['expires']:
            return None
        age = (now - self.bucket_log[-1][0]).total_seconds() if self.bucket_log else None
        if age is None or age > self.cfg.recency_sec:
            return None
        prev_c = float(df_3m["close"].iloc[-2])
        last_h = float(df_3m["high"].iloc[-1])
        last_l = float(df_3m["low"].iloc[-1])
        last_c = float(df_3m["close"].iloc[-1])
        n = self.cfg.vwap_sigma_entry
        # relaxed reentry: allow looser reentry patterns
        if ps['side'] == 'BUY':
            # allow either tight reentry OR looser "close >= vwap - n*vwap_std - small_margin"
            reentry = (prev_c <= vwap - n*vwap_std and last_c >= vwap - n*vwap_std) or (last_c >= vwap - (n+0.2)*vwap_std)
            if not reentry:
                return None
            entry = last_h + self.cfg.tick
            stop  = min(last_l, last_c - self.cfg.atr_stop_mult * float(atr)) - self.cfg.tick
            R = entry - stop; tp1, tp2 = entry + self.cfg.tp_R1*R, entry + self.cfg.tp_R2*R
        else:
            reentry = (prev_c > vwap + n*vwap_std and last_c < vwap + n*vwap_std) or (last_c <= vwap + (n+0.2)*vwap_std)
            if not reentry:
                return None
            entry = last_l - self.cfg.tick
            stop  = max(last_h, last_c + self.cfg.atr_stop_mult * float(atr)) + self.cfg.tick
            R = stop - entry; tp1, tp2 = entry - self.cfg.tp_R1*R, entry - self.cfg.tp_R2*R
        self.pending_setup = None
        return {"action":ps['side'],"entry":float(entry),"stop":float(stop),
                "targets":[float(tp1), float(tp2)],
                "context":{"mode":"LIQ_FADE","z":ps['z'],"lpi":ps['lpi'],
                            "vwap":float(vwap),"vwap_std":float(vwap_std),"atr":float(atr),
                            "bucket_total_usd":ps.get("bucket_total_usd", None)}}

# ----------------- Momentum / Squeeze Strategy -----------------

@dataclass
class MomentumConfig(BaseLiqConfig):
    cascade_dir_share: float = 0.6
    cascade_z3: float = 1.5
    cont_sigma: float = 0.45
    cont_range_atr: float = 0.45
    vol_mult: Optional[float] = 1.0
    tp_R1: float = 0.9
    tp_R2: float = 1.4
    atr_stop_mult: float = 0.8
    enable_fast_1m: bool = True
    fast_minutes: int = 2
    fast_dir_share: float = 0.40
    fast_zN: float = 1.2
    fast_sigma: float = 0.4
    fast_range_atr1m: float = 0.20

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
        if not getattr(self.cfg, "enable_fast_1m", False):
            return None
        now = self.time_manager.get_current_time()
        if df_1m is None:
            return None
        vwap, vwap_std = get_vwap()
        atr_3m = get_atr()
        atr1m = float(atr_3m) / sqrt(3.0) if atr_3m else 0.0
        age = self._recent_nonempty_bucket_age(now)
        if age is None or age > self.cfg.recency_sec:
            return None
        N = int(self.cfg.fast_minutes)
        lastN = self._lastN(now, minutes=N)
        if len(lastN) == 0:
            return None
        L = sum(b[1] for b in lastN); S = sum(b[2] for b in lastN); T = L + S
        if T <= 0:
            return None
        share = max(L, S) / T
        zL, zS = self._zN(L, S, N)
        side = 'BUY' if S > L else 'SELL'
        zN = zS if side == 'BUY' else zL
        # relaxed direction/share/z thresholds
        if (share < self.cfg.fast_dir_share) or (zN < self.cfg.fast_zN):
            return None
        if self.prev_1m is None:
            self.prev_1m = df_1m.iloc[-1]
            return None
        prev = self.prev_1m
        last = df_1m.iloc[-1]
        self.prev_1m = last
        last_close = float(last['close']); last_high = float(last['high']); last_low = float(last['low'])
        prev_high = float(prev['high']);  prev_low  = float(prev['low'])
        sigma = float(self.cfg.fast_sigma)
        rng_ok = (last_high - last_low) >= self.cfg.fast_range_atr1m * atr1m if atr1m and self.cfg.fast_range_atr1m>0 else True
        if side == 'BUY':
            cont = (last_close > vwap + sigma * vwap_std) and (last_high > prev_high) and rng_ok
            if not cont:
                return None
            entry = last_high + self.cfg.tick
            stop  = max(last_low, prev_low) - self.cfg.tick
            R = entry - stop; tp1, tp2 = entry + self.cfg.tp_R1*R, entry + self.cfg.tp_R2*R
        else:
            cont = (last_close < vwap - sigma * vwap_std) and (last_low < prev_low) and rng_ok
            if not cont:
                return None
            entry = last_low - self.cfg.tick
            stop  = min(last_high, prev_high) + self.cfg.tick
            R = stop - entry; tp1, tp2 = entry - self.cfg.tp_R1*R, entry - self.cfg.tp_R2*R
        return {"stage":"ENTRY","action":side,"entry":float(entry),"stop":float(stop),
                "targets":[float(tp1), float(tp2)],
                "context":{"mode":"LIQ_SQUEEZE_FAST_1M","minutes":N,"share":float(share),"zN":float(zN),
                            "vwap":float(vwap),"vwap_std":float(vwap_std),"atr1m":float(atr1m)}}
