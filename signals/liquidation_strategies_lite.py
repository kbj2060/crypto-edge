#!/usr/bin/env python3
"""
간소화된 청산 전략들
- Fade Reentry Strategy
- Squeeze Momentum Strategy
"""

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

# ============================================================
# 청산 기반 전략 (심플) - 사용자의 고급 전략 코드 스타일을 따름
# - SELL = 롱 청산, BUY = 숏 청산 (사용자 코드 기준)
# - μ/σ는 "1분 버킷 합계" 기준으로 0 버킷도 포함해 롤링 추정
# - on_bucket_close(): 1분마다 호출 (버킷 닫힘)
# - on_kline_close_1m(): 1분봉 마감(모멘텀 패스트)
# - on_kline_close_3m(): 3분봉 마감(페이드/모멘텀 보조)
# ============================================================

# -------------------- 공통 유틸/설정 --------------------

def _usd_from_event(ev: Dict[str, Any]) -> Tuple[str, float]:
    """이벤트 → (side, usd). SELL=롱 청산, BUY=숏 청산."""
    side = str(ev.get('side', '')).lower()
    size = float(ev.get('size', 0))
    price = float(ev.get('price', 0))
    
    # qty_usd가 있으면 사용, 없으면 계산
    if 'qty_usd' in ev and ev['qty_usd'] is not None:
        usd = float(ev['qty_usd'])
    else:
        usd = size * price
    
    return side, usd

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


@dataclass
class BaseLiqConfig:
    lookback_buckets: int = 240       # μ/σ 추정 4h
    recency_sec: int = 90             # 최근 비어있지 않은 버킷 신선도
    tick: float = 0.1                 # 주문 틱


# -------------------- 페이드(재진입) --------------------

@dataclass
class FadeConfig(BaseLiqConfig):
    agg_window_sec: int = 60
    min_bucket_notional_usd: float = 50000.0
    z_setup: float = 1.8
    lpi_min: float = 0.12
    setup_ttl_min: int = 10
    vwap_sigma_entry: float = 1.8
    atr_stop_mult: float = 0.9
    tp_R1: float = 1.3
    tp_R2: float = 2.0


class FadeReentryStrategy:
    """
    VWAP 밴드 재진입(페이드) — 사용자의 고급 전략에서 단순화.
    버킷(1m)에서 SETUP, 3분봉 마감에서 ENTRY 확정.
    """
    def __init__(self, cfg: FadeConfig):
        self.cfg = cfg
        # Time Manager 초기화
        self.time_manager = get_time_manager()
        # 1분 버킷 롤링 통계(0 버킷 포함)
        self.long_hist = deque(maxlen=cfg.lookback_buckets)
        self.short_hist = deque(maxlen=cfg.lookback_buckets)
        self.mu_long = 0.0; self.sd_long = 1.0
        self.mu_short = 0.0; self.sd_short = 1.0
        # 비어있지 않은 버킷 로그(신선도 체크)
        self.bucket_log: List[Tuple[datetime, float, float, float]] = []
        # 보류 SETUP
        self.pending_setup: Optional[Dict[str, Any]] = None

    # ---- 내부 ----
    def _update_stats(self, long_usd: float, short_usd: float) -> None:
        self.long_hist.append(float(long_usd))
        self.short_hist.append(float(short_usd))
        if len(self.long_hist) >= 30:
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

    def _recent_nonempty_bucket_age(self, now_utc: datetime) -> Optional[float]:
        if not self.bucket_log: return None
        return (now_utc - self.bucket_log[-1][0]).total_seconds()

    def warmup(self, bucket_events: List[Dict[str, Any]]) -> None:
        print(f"🔥 [FADE] 전략 워밍업 시작 - {len(bucket_events)}개 이벤트")
        
        # bucket_log 초기화 및 형태 변환
        self.bucket_log = []
        
        for i, ev in enumerate(bucket_events):
            side, usd = _usd_from_event(ev)
            timestamp = self.time_manager.get_timestamp_datetime(ev['timestamp'])
            if side in ('sell', 'long'):
                self._update_stats(float(usd), 0.0)
                self.bucket_log.append((timestamp, float(usd), 0.0, float(usd)))
            elif side in ('buy', 'short'): 
                self._update_stats(0.0, float(usd))
                self.bucket_log.append((timestamp, 0.0, float(usd), float(usd)))
        
        print(f"✅ [FADE] 워밍업 완료 - 롱 μ={self.mu_long:.0f}, σ={self.sd_long:.0f}, 숏 μ={self.mu_short:.0f}, σ={self.sd_short:.0f}")

    # ---- 1) 버킷 닫힘(1m) ----
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
            'z': float(max_z),'lpi': float(lpi), 'bucket_total_usd': float(total)
        }
        
        print(f"🎯 [FADE] 신호 생성: {side} | Z={max_z:.2f} | LPI={lpi:.3f} | 총액=${total:,.0f}")
        return {"stage":"SETUP","action":side,"z":float(max_z),"lpi":float(lpi),
                "bucket_total_usd":float(total),"created":now.isoformat()}

    # ---- 2) 3분봉 마감 ----
    def on_kline_close_3m(self) -> Optional[Dict[str, Any]]:
        data_manager = get_data_manager()
        df_3m = data_manager.get_latest_data(count=2)
        if df_3m is None or len(df_3m) < 2: 
            return None
        
        now = self.time_manager.get_current_time()
        
        # VWAP 및 VWAP 표준편차
        vwap, vwap_std = get_vwap()
        atr = get_atr()
        
        ps = self.pending_setup
        if not ps or now > ps['expires']:
            print("📊 [FADE] 보류 SETUP 없음 또는 만료")
            return None
            
        age = self._recent_nonempty_bucket_age(now)
        if age is None or age > self.cfg.recency_sec: 
            print("⚠️ [FADE] 최근 버킷 신선도 불만족")
            return None

        prev_c = float(df_3m["close"].iloc[-2])
        last_h = float(df_3m["high"].iloc[-1])
        last_l = float(df_3m["low"].iloc[-1])
        last_c = float(df_3m["close"].iloc[-1])
        
        n = self.cfg.vwap_sigma_entry
        if ps['side'] == 'BUY':
            reentry = (prev_c <= vwap - n*vwap_std) and (last_c >= vwap - n*vwap_std)
            if not reentry: 
                return None
            entry = last_h + self.cfg.tick
            stop  = min(last_l, last_c - self.cfg.atr_stop_mult * float(atr)) - self.cfg.tick
            R = entry - stop; tp1, tp2 = entry + self.cfg.tp_R1*R, entry + self.cfg.tp_R2*R
        else:
            reentry = (prev_c > vwap + n*vwap_std) and (last_c < vwap + n*vwap_std)
            if not reentry: 
                return None
            entry = last_l - self.cfg.tick
            stop  = max(last_h, last_c + self.cfg.atr_stop_mult * float(atr)) + self.cfg.tick
            R = stop - entry; tp1, tp2 = entry - self.cfg.tp_R1*R, entry - self.cfg.tp_R2*R

        self.pending_setup = None
        print(f"🎯 [FADE] ENTRY 신호 생성: {ps['side']} | 진입=${entry:.2f} | 손절=${stop:.2f} | 목표1=${tp1:.2f} | 목표2=${tp2:.2f}")
        return {"action":ps['side'],"entry":float(entry),"stop":float(stop),
                "targets":[float(tp1), float(tp2)],
                "context":{"mode":"LIQ_FADE","z":ps['z'],"lpi":ps['lpi'],
                            "vwap":float(vwap),"vwap_std":float(vwap_std),"atr":float(atr),
                            "bucket_total_usd":ps.get("bucket_total_usd", None)}}


# -------------------- 스퀴즈(모멘텀) --------------------

@dataclass
class MomentumConfig(BaseLiqConfig):
    # 3분 보조 패스
    cascade_dir_share: float = 0.80
    cascade_z3: float = 2.6
    cont_sigma: float = 0.8
    cont_range_atr: float = 0.8
    vol_mult: Optional[float] = 1.2
    tp_R1: float = 1.0
    tp_R2: float = 1.8
    atr_stop_mult: float = 1.0
    # 1분 패스트 패스
    enable_fast_1m: bool = True
    fast_minutes: int = 2
    fast_dir_share: float = 0.65
    fast_zN: float = 2.0
    fast_sigma: float = 0.6
    fast_range_atr1m: float = 0.35

class SqueezeMomentumStrategy:
    """
    캐스케이드 모멘텀(스퀴즈) — 1분 패스트 + 3분 보조
    """
    def __init__(self, cfg: MomentumConfig):
        self.cfg = cfg
        # Time Manager 초기화
        self.time_manager = get_time_manager()
        self.long_hist = deque(maxlen=cfg.lookback_buckets)
        self.short_hist = deque(maxlen=cfg.lookback_buckets)
        self.mu_long = 0.0; self.sd_long = 1.0
        self.mu_short = 0.0; self.sd_short = 1.0
        self.bucket_log: List[Tuple[datetime, float, float, float]] = []
        self.prev_1m = None

    # ---- 내부 ----
    def _update_stats(self, long_usd: float, short_usd: float) -> None:
        self.long_hist.append(float(long_usd))
        self.short_hist.append(float(short_usd))
        if len(self.long_hist) >= 30:
            self.mu_long = float(np.mean(self.long_hist))
            self.mu_short = float(np.mean(self.short_hist))
            self.sd_long = float(np.std(self.long_hist, ddof=1)) 
            self.sd_short = float(np.std(self.short_hist, ddof=1))

    def _recent_nonempty_bucket_age(self, now_utc: datetime) -> Optional[float]:
        if not self.bucket_log: return None
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
        print(f"🔥 [SQUEEZE] 전략 워밍업 시작 - {len(bucket_events)}개 이벤트")
        # bucket_log 초기화 및 형태 변환
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
        
        print(f"✅ [SQUEEZE] 워밍업 완료 - 롱 μ={self.mu_long:.0f}, σ={self.sd_long:.0f}, 숏 μ={self.mu_short:.0f}, σ={self.sd_short:.0f}")

    # ---- 1) 버킷 닫힘(1m) ----
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

    # ---- 2) 1분봉 마감(패스트) ----
    def on_kline_close_1m(self, df_1m: pd.DataFrame) -> Optional[Dict[str, Any]]:
        if not getattr(self.cfg, "enable_fast_1m", False): 
            return None
        
        now = self.time_manager.get_current_time()
        if df_1m is None: 
            return None

        # VWAP 및 VWAP 표준편차
        vwap, vwap_std = get_vwap()
        atr_3m = get_atr()
        atr1m = float(atr_3m) / sqrt(3.0)

        age = self._recent_nonempty_bucket_age(now)
        if age is None or age > self.cfg.recency_sec: 
            return None

        N = int(self.cfg.fast_minutes)
        lastN = self._lastN(now, minutes=N)
        if len(lastN) == 0: 
            return None
        
        L = sum(b[1] for b in lastN); S = sum(b[2] for b in lastN); T = L + S
        if T <= 0: 
            print(f"⚠️ [SQUEEZE] 총 청산 금액 0: 롱=${L:,.0f}, 숏=${S:,.0f}")
            return None
        
        share = max(L, S) / T
        zL, zS = self._zN(L, S, N)
        side = 'BUY' if S > L else 'SELL'
        zN = zS if side == 'BUY' else zL
                
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

        rng_ok = (last_high - last_low) >= self.cfg.fast_range_atr1m * atr1m

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

    # ---- 3) 3분봉 마감(보조) ----
    def on_kline_close_3m(self) -> Optional[Dict[str, Any]]:
        data_manager = get_data_manager()
        df_3m = data_manager.get_latest_data(count=2)
        if df_3m is None or len(df_3m) < 2: return None
        now = self.time_manager.get_current_time()
        
        # VWAP 및 VWAP 표준편차
        vwap, vwap_std = get_vwap()
        atr = get_atr()
        
        # ATR
        age = self._recent_nonempty_bucket_age(now)
        if age > self.cfg.recency_sec: 
            print(f"⚠️ [SQUEEZE] 3M 버킷 데이터 오래됨: age={age:.1f}s > {self.cfg.recency_sec}s")
            return None

        cut = now - timedelta(minutes=3)
        last3 = [b for b in self.bucket_log if b[0] > cut]
        if len(last3) == 0: return None
        L3 = sum(b[1] for b in last3); S3 = sum(b[2] for b in last3); T3 = L3 + S3
        if T3 <= 0: return None
        share = max(L3, S3) / T3

        mu3L, mu3S = 3*self.mu_long, 3*self.mu_short
        sd3L, sd3S = (3**0.5)*self.sd_long, (3**0.5)*self.sd_short
        zL3 = max(0.0, (L3 - mu3L) / max(sd3L, 1e-9)); zS3 = max(0.0, (S3 - mu3S) / max(sd3S, 1e-9))
        side = 'BUY' if S3 > L3 else 'SELL'; z3 = zS3 if side == 'BUY' else zL3
        if (share < self.cfg.cascade_dir_share) or (z3 < self.cfg.cascade_z3): return None

        prev = df_3m.iloc[-2]; last = df_3m.iloc[-1]
        rng = float(last['high'] - last['low'])
        atrv = float(atr)
        vol_ok = True
        if ('quote_volume' in df_3m.columns) and (self.cfg.vol_mult is not None):
            ma20 = float(df_3m['quote_volume'].rolling(20).mean().iloc[-1])
            if ma20 > 0:
                vol_ok = float(last['quote_volume']) >= self.cfg.vol_mult * ma20

        if side == 'BUY':
            cont = (float(last['close']) > vwap + self.cfg.cont_sigma*vwap_std) and (float(last['high']) > float(prev['high'])) and (rng >= self.cfg.cont_range_atr * atrv) and vol_ok
            if not cont: return None
            entry = float(last['high']) + self.cfg.tick
            stop  = max(float(last['low']), float(prev['low'])) - self.cfg.tick
            R = entry - stop; tp1, tp2 = entry + self.cfg.tp_R1*R, entry + self.cfg.tp_R2*R
        else:
            cont = (float(last['close']) < vwap - self.cfg.cont_sigma*vwap_std) and (float(last['low']) < float(prev['low'])) and (rng >= self.cfg.cont_range_atr * atrv) and vol_ok
            if not cont: return None
            entry = float(last['low']) - self.cfg.tick
            stop  = min(float(last['high']), float(prev['high'])) + self.cfg.tick
            R = stop - entry; tp1, tp2 = entry - self.cfg.tp_R1*R, entry - self.cfg.tp_R2*R

        return {"stage":"ENTRY","action":side,"entry":float(entry),"stop":float(stop),
                "targets":[float(tp1), float(tp2)],
                "context":{"mode":"LIQ_SQUEEZE_3M","share":float(share),"z3":float(z3),
                            "vwap":float(vwap),"vwap_std":float(vwap_std),"atr":float(atr)}}
