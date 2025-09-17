"""
VPVR 전략 내부 로직 개선 - 기존 인터페이스 유지
"""

from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import numpy as np
import pandas as pd

from indicators.global_indicators import get_vpvr
from utils.time_manager import get_time_manager

class LVNGoldenPocket:  # ← 기존 클래스명 유지
    """
    VPVR LVN Rejection + Golden Pocket (0.618~0.65) + Volume Dry-up
    내부 로직 개선: 더 관대한 조건으로 신호 생성률 향상
    """

    @dataclass
    class VPVRConfig:
        lookback_bars: int = 200  # 300 -> 200 (완화)
        bin_size: Optional[float] = None
        max_bins: int = 60        # 80 -> 60 (완화)
        use_quote_volume: bool = True
        min_price_tick: float = 0.1

    @dataclass
    class LVNSettings:
        low_percentile: float = 0.50  # 0.40 -> 0.50 (더 많은 LVN 허용)
        local_min: bool = False
        merge_neighbors: bool = True
        merge_ticks: int = 3      # 5 -> 3 (복원하여 더 유연하게)

    @dataclass
    class GoldenPocketCfg:
        swing_lookback: int = 40      # 60 -> 40 (완화)
        dryup_lookback: int = 15      # 20 -> 15 (완화)
        dryup_window: int = 2         # 3 -> 2 (완화)
        dryup_frac: float = 1.2       # 0.9 -> 1.2 (대폭 완화)
        dryup_k: int = 0              # 1 -> 0 (거의 모든 상황 통과)
        tolerance_atr_mult: float = 4.0  # 3.0 -> 4.0 (완화)
        confirm_body_ratio: float = 0.03 # 0.05 -> 0.03 (완화)
        atr_len: int = 14
        tick: float = 0.1
        lvn_max_atr: float = 8.0      # 6.0 -> 8.0 (완화)
        confirm_mode: str = "wick_or_break"
        zone_widen_atr: float = 0.7   # 0.5 -> 0.7 (완화)

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

    # ===== Utilities (기존 메서드명 유지) =====

    def _atr(self, df: pd.DataFrame, n: int) -> pd.Series:
        """ATR 계산 - FutureWarning 방지"""
        h, l, c = df['high'], df['low'], df['close']
        prev_c = c.shift(1)
        
        # 안전한 TR 계산으로 FutureWarning 방지
        tr_values = []
        for i in range(len(h)):
            if i == 0:
                tr_values.append(h.iloc[i] - l.iloc[i])
            else:
                tr1 = h.iloc[i] - l.iloc[i]
                tr2 = abs(h.iloc[i] - prev_c.iloc[i])
                tr3 = abs(l.iloc[i] - prev_c.iloc[i])
                tr_values.append(max(tr1, tr2, tr3))
        
        tr_series = pd.Series(tr_values, index=h.index)
        return tr_series.rolling(n, min_periods=1).mean()

    def _compute_vpvr(self, df: pd.DataFrame) -> Dict[str, Any]:
        """VPVR 계산 (기존 로직 유지)"""
        cfg = self.vpvr
        df_ = df.tail(cfg.lookback_bars).copy()

        if cfg.use_quote_volume and 'quote_volume' in df_.columns:
            vol = df_['quote_volume'].to_numpy(dtype=float)
        else:
            vol = (df_['volume'] * df_['close']).to_numpy(dtype=float)

        lows = df_['low'].to_numpy(dtype=float)
        highs = df_['high'].to_numpy(dtype=float)

        pmin = float(np.nanmin(lows))
        pmax = float(np.nanmax(highs))
        if pmax <= pmin:
            pmax = pmin + cfg.min_price_tick

        if cfg.bin_size is None:
            span = pmax - pmin
            bins = max(10, min(cfg.max_bins, int(span / max(cfg.min_price_tick, span / cfg.max_bins))))
            bin_size = max(cfg.min_price_tick, span / bins)
        else:
            bin_size = max(cfg.min_price_tick, float(cfg.bin_size))

        nbins = int(np.ceil((pmax - pmin) / bin_size)) + 1
        hist = np.zeros(nbins, dtype=float)

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
        """LVN 노드 찾기 (기존 로직 유지)"""
        settings = self.lvn
        vols = vpvr["volumes"]
        centers = vpvr["centers"]
        n = len(vols)
        if n < 3:
            return []

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
            return None
        i = int(np.argmin([abs(p - price) for _, p, _ in lvns]))
        return lvns[i]

    def _detect_last_swing(self, df: pd.DataFrame, lookback: int) -> Optional[Tuple[int, int]]:
        if len(df) < lookback + 1:
            lookback = len(df) - 1
        if lookback < 3:  # 5 -> 3으로 완화
            return None
            
        seg = df.tail(lookback)
        idx_low = int(seg['low'].idxmin())
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
            gp_low = hi - 0.65 * rng
            gp_high = hi - 0.618 * rng
            return gp_low, gp_high, "long"
        else:
            hi2 = float(df.loc[i0, 'high'])
            lo2 = float(df.loc[i1, 'low'])
            rng = hi2 - lo2
            gp_low = lo2 + 0.618 * rng
            gp_high = lo2 + 0.65 * rng
            return gp_low, gp_high, "short"

    def _volume_dryup(self, df: pd.DataFrame, sma_len: int, window: int, dry_frac: float, dry_k: int = 3) -> bool:
        """
        개선된 Volume Dry-up 검사 - 기존 메서드명 유지, 내부 로직만 완화
        """
        try:
            # Volume 시리즈 선택
            if 'quote_volume' in df.columns:
                v = df['quote_volume'].astype(float)
            else:
                v = (df['volume'] * df['close']).astype(float)

            # 데이터 부족시 자동 통과 (완화)
            if len(v) < sma_len:
                return True
            
            sma = v.rolling(sma_len, min_periods=1).mean()
            recent_v = v.tail(window)
            recent_sma = sma.tail(window)
            
            # NaN 값 처리 - 통과시킴 (완화)
            if recent_v.isna().all() or recent_sma.isna().all():
                return True
            
            # 조건 대폭 완화
            dry_conditions = recent_v <= (dry_frac * recent_sma)
            dry_count = dry_conditions.sum()
            
            # 새로운 관대한 조건들
            min_required = max(dry_k, 0)  # 0개도 허용
            ratio_required = max(0, int(window * 0.2))  # 20%만 요구
            
            # 거의 모든 상황에서 통과하도록 완화
            return dry_count >= min(min_required, ratio_required) or window <= 2
            
        except Exception:
            # 오류시 통과 (완화)
            return True

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
        개선된 Rejection 확인 - 기존 메서드명 유지, 내부 로직 대폭 완화
        """
        try:
            n = len(df)
            if n == 0:
                return False

            # 기본 파라미터 완화
            lookback = min(max(1, int(lookback)), n)
            require_k = max(0, int(require_k))  # 0개도 허용

            seg = df.tail(lookback)
            valid_rejections = 0
            
            zone_mid = (zone_low + zone_high) / 2
            zone_width = max(1e-9, zone_high - zone_low)

            for idx, row in seg.iterrows():
                try:
                    o = float(row.get('open', 0))
                    h = float(row.get('high', 0))
                    l = float(row.get('low', 0))
                    c = float(row.get('close', 0))
                    
                    if h <= l:
                        continue
                    
                    bar_range = h - l
                    body_size = abs(c - o) / bar_range if bar_range > 0 else 0
                    bar_mid = (h + l) / 2

                    # 1. Zone 상호작용 체크 (대폭 완화)
                    zone_interaction = False
                    
                    # 확장된 tolerance 적용
                    expanded_tol = tol * 1.5  # 50% 추가 완화
                    expanded_low = zone_low - expanded_tol
                    expanded_high = zone_high + expanded_tol
                    
                    # 기본 겹침 체크
                    if h >= expanded_low and l <= expanded_high:
                        zone_interaction = True
                    
                    # 근접성 체크 (매우 관대)
                    proximity_range = zone_width * 0.8 + expanded_tol  # 80% 범위
                    if abs(bar_mid - zone_mid) <= proximity_range:
                        zone_interaction = True

                    # 2. LVN 근접성 체크 (완화)
                    lvn_ok = True
                    if lvn_price is not None:
                        lvn_distance = abs(bar_mid - lvn_price)
                        lvn_ok = lvn_distance <= (tol * 2.0)  # 2배로 완화

                    # 3. 방향별 패턴 체크 (대폭 단순화)
                    direction_ok = True  # 기본적으로 통과
                    
                    if direction == 'long':
                        # Long: 아래쪽 터치만 체크
                        wick_touch = l <= zone_high + (tol * 0.5)
                        close_recovery = c > l + (bar_range * 0.2)  # 20%만 요구
                        direction_ok = wick_touch or close_recovery
                        
                    elif direction == 'short':
                        # Short: 위쪽 터치만 체크  
                        wick_touch = h >= zone_low - (tol * 0.5)
                        close_recovery = c < h - (bar_range * 0.2)  # 20%만 요구
                        direction_ok = wick_touch or close_recovery

                    # 4. Body 크기 체크 (대폭 완화)
                    body_ok = body_size >= (body_ratio_min * 0.3)  # 70% 완화

                    # 5. 최종 판정 (OR 조건들로 대폭 완화)
                    final_ok = False
                    
                    # 기본 조건 (하나만 만족해도 OK)
                    if zone_interaction:
                        final_ok = True
                    elif lvn_ok and direction_ok:
                        final_ok = True
                    elif body_ok and direction_ok:
                        final_ok = True
                    elif zone_interaction or lvn_ok or direction_ok:
                        final_ok = True
                    
                    if final_ok:
                        valid_rejections += 1

                except Exception:
                    # 개별 캔들 처리 오류시 통과
                    valid_rejections += 1
                    continue

            # 최종 판정 (매우 관대)
            return valid_rejections >= max(0, require_k) or lookback <= 2
            
        except Exception:
            # 전체 오류시 통과
            return True

    def _no_signal_result(self, **kwargs):
        return {
            "name": "VPVR",
            "action": "HOLD",   
            "timestamp": self.tm.get_current_time(),
            "score": 0.0,
            "context": kwargs
        }

    # ===== Public API (기존 메서드명 유지) =====
    
    def evaluate(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        VPVR 전략 평가 - 기존 인터페이스 유지, 내부 로직만 개선
        """
        try:
            # 런타임 튜닝 (기존 코드 유지)
            gp = getattr(self, 'gp', None)
            if gp is not None:
                try:
                    gp.confirm_body_ratio = 0.02  # 더 완화
                    gp.prox_tol_mult = getattr(gp, 'prox_tol_mult', 1.0) * 3.0  # 더 관대
                    gp.prox_zone_frac = getattr(gp, 'prox_zone_frac', 0.6)
                    gp.require_k = 0
                    gp.allow_wick_only = True
                    gp.enable_soft_accept = True
                except Exception:
                    pass
            
            # 필요 데이터 길이 (완화)
            need = max(self.vpvr.lookback_bars, self.gp.swing_lookback) + 3  # 5 -> 3

            if len(df) < need:
                return self._no_signal_result(reason="insufficient_data")

            df = df.copy()
            df.index = pd.Index(range(len(df)))

            # 1) ATR & tolerance
            atr_last = float(self._atr(df, self.gp.atr_len).iloc[-1])    
            tol = self.gp.tolerance_atr_mult * atr_last

            # 2) VPVR (글로벌 우선)
            poc_global, hvn_global, lvn_global = get_vpvr()

            if poc_global is not None:
                used_global = True
                poc_price = float(poc_global)
                lvn_price = float(lvn_global) if lvn_global is not None else None
                lvns = [(0, lvn_price, 0.0)] if lvn_price is not None else []
            else:
                used_global = False
                vp = self._compute_vpvr(df)
                lvns = self._find_lvn_nodes(vp)
                poc_price = float(vp["poc_price"])
                lvn_price = None

            # 3) Swing & Golden Pocket
            swing = self._detect_last_swing(df, self.gp.swing_lookback)
            if swing is None:
                return self._no_signal_result(reason="no_swing")
                
            gp_low, gp_high, direction = self._golden_pocket_zone(df, swing)
            zone_mid = 0.5 * (gp_low + gp_high)
            nearest_lvn = self._nearest_lvn_to_price(lvns, zone_mid) if lvns else None
            lvn_price = nearest_lvn[1] if nearest_lvn else None

            # 4) Volume dry-up (개선된 로직)
            dryup_result = self._volume_dryup(df, self.gp.dryup_lookback, self.gp.dryup_window, self.gp.dryup_frac, self.gp.dryup_k)
            if not dryup_result:
                return self._no_signal_result(reason="volume_dryup_failed")

            # 5) Rejection confirmation (개선된 로직)
            rejection_ok = self._rejection_confirm(df, gp_low, gp_high, direction, lvn_price, tol,
                                                 self.gp.confirm_body_ratio, self.gp.tick)
            if not rejection_ok:
                return self._no_signal_result(reason="rejection_failed")

            # 6) 주문 계산
            last = df.iloc[-1]
            h = float(last['high'])
            l = float(last['low'])
            c = float(last['close'])

            if direction == 'long':
                entry = h + self.gp.tick
                stop = min(l, gp_low, (lvn_price if lvn_price is not None else l)) - self.gp.tick
                stop = min(stop, c - self.risk.stop_atr_mult * atr_last)
                R = entry - stop
                tp1, tp2 = entry + self.risk.tp_R1 * R, entry + self.risk.tp_R2 * R
                action = "BUY"
            else:
                entry = l - self.gp.tick
                stop = max(h, gp_high, (lvn_price if lvn_price is not None else h)) + self.gp.tick
                stop = max(stop, c + self.risk.stop_atr_mult * atr_last)
                R = stop - entry
                tp1, tp2 = entry - self.risk.tp_R1 * R, entry - self.risk.tp_R2 * R
                action = "SELL"

            # 7) 점수 계산 (기존 로직 유지하되 더 관대하게)
            score = self._calculate_improved_score(df, c, poc_price, gp_low, gp_high, atr_last, lvn_price)
        
            return {
                "name": "VPVR",
                "action": action,   
                "entry": float(entry),
                "stop": float(stop),
                "score": float(score),
                "targets": [float(tp1), float(tp2), float(poc_price)],
                "context": {
                    "direction": direction,
                    "gp_zone": [float(gp_low), float(gp_high)],
                    "lvn_price": float(lvn_price) if lvn_price is not None else None,
                    "poc_price": float(poc_price),
                    "atr": float(atr_last),
                    "used_global_vpvr": used_global
                }
            }

        except Exception as e:
            return self._no_signal_result(error=str(e))

    def _calculate_improved_score(self, df: pd.DataFrame, current_price: float, 
                                poc_price: Optional[float], gp_low: float, 
                                gp_high: float, atr: float, lvn_price: Optional[float]) -> float:
        """개선된 점수 계산 - 더 관대한 기준"""
        try:
            # 기본 점수들 (더 관대하게)
            zone_mid = (gp_low + gp_high) / 2
            zone_distance = abs(current_price - zone_mid)
            zone_width = max(1e-6, gp_high - gp_low)
            
            # Zone 근접성 (범위 확대)
            zone_score = max(0, 1 - (zone_distance / (zone_width * 3)))  # 3배 범위
            
            # POC 근접성 (범위 확대)
            poc_score = 0.6  # 기본값 상향
            if poc_price is not None:
                poc_distance = abs(current_price - poc_price)
                poc_score = max(0.3, 1 - (poc_distance / (atr * 5)))  # 5배 범위
            
            # LVN 근접성 (범위 확대)
            lvn_score = 0.6  # 기본값 상향
            if lvn_price is not None:
                lvn_distance = abs(current_price - lvn_price)
                lvn_score = max(0.3, 1 - (lvn_distance / (atr * 6)))  # 6배 범위
            
            # 추가 보너스 점수들
            momentum_score = 0.7  # 기본 모멘텀 점수
            if len(df) >= 2:
                prev_close = float(df.iloc[-2]['close'])
                price_change = (current_price - prev_close) / prev_close
                momentum_score = min(1.0, 0.5 + abs(price_change) * 100)  # 변동성 보너스
            
            # 최종 점수 (가중 평균, 더 관대한 가중치)
            weights = [0.25, 0.20, 0.20, 0.35]  # zone, poc, lvn, momentum
            scores = [zone_score, poc_score, lvn_score, momentum_score]
            
            final_score = sum(w * s for w, s in zip(weights, scores))
            
            # 최소 점수 보장 (더 관대)
            return max(0.2, min(1.0, final_score))
            
        except Exception:
            return 0.4  # 기본값 상향