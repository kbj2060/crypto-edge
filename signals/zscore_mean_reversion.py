# zscore_mean_reversion.py
# Put this file in your project (e.g., signals/zscore_mean_reversion.py)
# Usage:
#   from zscore_mean_reversion import ZScoreConfig, ZScoreMeanReversion
#   cfg = ZScoreConfig()
#   bot = ZScoreMeanReversion(cfg)
#   signal = bot.on_kline_close_3m(df)   # df must have 'close' (and optionally 'high','low','volume')

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from managers.data_manager import get_data_manager


@dataclass
class ZScoreConfig:
    # 단기 전략 최적화 (3분봉 기준)
    window: int = 60                 # 80 → 60 (3시간, 단기용)
    use_log: bool = True             # if True use log-price -> log-returns; else pct_change
    z_thresh: float = 0.8            # 1.2 → 0.8 (더 민감한 신호)
    exit_z: float = 0.3              # 0.5 → 0.3 (더 빠른 청산)
    atr_period: int = 14
    stop_atr_mult: float = 1.5       # 2.0 → 1.5 (더 타이트한 스탑)
    take_profit_atr_mult: Optional[float] = 2.0  # 2.5 → 2.0 (더 빠른 목표)
    min_volume: float = 0.0
    min_history: int = 120           # 200 → 120 (10시간, 단기용)
    mode: str = "price"              # unused other modes kept for compatibility
    vwap_window: int = 200           # 300 → 200 (10시간, 단기용)
    # new options
    use_ewma: bool = True
    ewma_span: Optional[int] = None
    z_scale: float = 1.2             # 1.0 → 1.2 (민감도 증가)
    pre_signal_pct: float = 0.6      # 0.75 → 0.6 (더 빠른 사전 신호)
    debug: bool = False               # 디버깅 활성화


class ZScoreMeanReversion:
    def __init__(self, cfg: Optional[ZScoreConfig] = ZScoreConfig()):
        self.config = cfg or ZScoreConfig()
        self.data_manager = get_data_manager()

    @staticmethod
    def compute_zscore(series: pd.Series, window: int) -> pd.Series:
        """
        Compute Z-score on a return-like series (e.g., log-returns or pct-change).
        Uses rolling mean & rolling std with shift(1) to avoid lookahead.
        Small epsilon replaces zero/NaN stds to avoid division by zero.
        """
        x = series.astype(float).copy()
        mean = x.rolling(window, min_periods=1).mean().shift(1)
        std = x.rolling(window, min_periods=1).std(ddof=0).shift(1)
        std = std.replace(0, np.nan).fillna(1e-8)
        z = (x - mean) / std
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        z = z.clip(lower=-100.0, upper=100.0)
        return z

    @staticmethod
    def _zscore_to_signal(last_z: float, cfg: ZScoreConfig) -> Dict[str, Any]:
        """Convert last z into (action, score) with pre-signal logic (단기 전략 최적화)."""
        absz = abs(last_z)
        zt = float(cfg.z_thresh)
        pre_pct = float(cfg.pre_signal_pct)
        result = {"action": "HOLD", "score": 0.0}

        if absz >= zt:
            # full trigger (단기 전략용 완화된 조건)
            act = "BUY" if last_z < 0 else "SELL"
            # continuous score: base 0.6 then scale with additional z beyond threshold
            extra = max(0.0, (absz - zt) / max(1e-9, zt))
            score = min(1.0, 0.6 + 0.4 * extra)  # 0.5 → 0.6 (더 높은 기본 점수)
            conf = 0.8 if absz >= (zt * 1.5) else 0.6  # 1.25 → 1.5 (더 관대한 조건)
            result.update({"action": act, "score": float(score)})
            
            if cfg.debug:
                print(f"[ZSCORE] 강한 신호: {act} | Z={last_z:.3f} | Score={score:.3f} | Conf={conf:.3f}")
                
        elif absz >= (zt * pre_pct):
            # pre-signal: 단기 전략에서는 더 높은 점수 부여
            pre_score = float(max(0.1, 0.5 * (absz / zt)))  # 0.05 → 0.1, 0.4 → 0.5
            result.update({"action": "HOLD", "score": pre_score})
            
            if cfg.debug:
                print(f"[ZSCORE] 사전 신호: Z={last_z:.3f} | Score={pre_score:.3f}")
        else:
            result.update({"action": "HOLD", "score": 0.0})
            
        return result

    def _prepare_input_series(self, df: pd.DataFrame) -> pd.Series:
        """Return the series (returns-style) that we'll compute z-score on."""
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        # clean price series
        price = df["close"].astype(float).replace(0, np.nan).ffill().bfill()

        if self.config.use_log:
            # log-price -> log-return
            logp = np.log(price)
            ret = logp.diff().fillna(0.0)
            return ret
        else:
            # percent change
            return price.pct_change().fillna(0.0)

    def on_kline_close_3m(self) -> Dict[str, Any]:
        """
        Main entry (단기 전략 최적화).
        Input: df (pandas DataFrame) with at least 'close'; optionally 'volume','high','low'.
        Output: dict with keys: name, action(BUY/SELL/HOLD), score, entry, stop, tp, context
        """
        # basic checks
        df = self.data_manager.get_latest_data(count=self.config.min_history)
        if len(df) < max(self.config.min_history, self.config.window + 2):
            if self.config.debug:
                print(f"[ZSCORE] 데이터 부족: {len(df)} < {max(self.config.min_history, self.config.window + 2)}")
            return {
                "name": "ZSCORE_MEAN_REVERSION",
                "action": "HOLD",
                "score": 0.0,
                "entry": None,
                "stop": None,
                "tp": None,
                "context": {"reason": "not_enough_history", "n": len(df)}
            }

        # volume gate if applicable
        if "quote_volume" in df.columns and self.config.min_volume and df["quote_volume"].iloc[-1] < self.config.min_volume:
            if self.config.debug:
                print(f"[ZSCORE] 볼륨 부족: {df['quote_volume'].iloc[-1]} < {self.config.min_volume}")
            return {
                "name": "ZSCORE_MEAN_REVERSION",
                "action": "HOLD",
                "score": 0.0,
                "entry": df["close"].iloc[-1],
                "stop": None,
                "tp": None,
                "context": {"reason": "low_volume", "current_volume": float(df["quote_volume"].iloc[-1])}
            }

        # prepare input series (returns-like)
        try:
            input_series = self._prepare_input_series(df)
            if self.config.debug:
                print(f"[ZSCORE] 입력 시리즈 준비 완료: {len(input_series)}개 데이터")
        except Exception as e:
            if self.config.debug:
                print(f"[ZSCORE] 입력 시리즈 준비 실패: {e}")
            return {
                "name": "ZSCORE_MEAN_REVERSION",
                "action": "HOLD",
                "score": 0.0,
                "entry": df["close"].iloc[-1] if "close" in df.columns else None,
                "stop": None,
                "tp": None,
                "context": {"reason": "prepare_input_failed", "error": str(e)}
            }

        # compute z using EWMA or rolling
        if self.config.use_ewma:
            span = int(self.config.ewma_span) if self.config.ewma_span else self.config.window
            mean = input_series.ewm(span=span, adjust=False).mean().shift(1)
            std = input_series.ewm(span=span, adjust=False).std(bias=False).shift(1).replace(0, np.nan).fillna(1e-8)
            z = (input_series - mean) / std
            z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-100.0, 100.0)
            if self.config.debug:
                print(f"[ZSCORE] EWMA 계산 완료: span={span}")
        else:
            z = self.compute_zscore(input_series, self.config.window)
            if self.config.debug:
                print(f"[ZSCORE] Rolling Z-Score 계산 완료: window={self.config.window}")

        # optional scaling (단기 전략용 민감도 증가)
        if getattr(self.config, "z_scale", 1.0) != 1.0:
            z = z * float(self.config.z_scale)
            if self.config.debug:
                print(f"[ZSCORE] Z-Score 스케일링 적용: {self.config.z_scale}")

        last_z = float(z.iloc[-1])
        last_price = float(df["close"].iloc[-1])
        
        # approximate ATR (simple proxy if hi/lo available)
        if "high" in df.columns and "low" in df.columns:
            tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1)
            tr = tr.max(axis=1)
            atr = tr.rolling(self.config.atr_period, min_periods=1).mean().iloc[-1]
        else:
            atr = float(df["close"].diff().abs().rolling(self.config.atr_period, min_periods=1).mean().iloc[-1])

        if self.config.debug:
            print(f"[ZSCORE] 현재 상태: Z={last_z:.3f}, Price={last_price:.4f}, ATR={atr:.4f}")

        # convert z -> signal (with pre-signal)
        sig = self._zscore_to_signal(last_z, self.config)

        # produce entry/stop/tp using ATR if real trade
        entry = last_price
        stop = None
        tp = None
        if sig["action"] == "BUY":
            stop = entry - self.config.stop_atr_mult * atr
            tp = entry + (self.config.take_profit_atr_mult * atr if self.config.take_profit_atr_mult else None)
            if self.config.debug:
                print(f"[ZSCORE] BUY 신호: Entry={entry:.4f}, Stop={stop:.4f}, TP={tp:.4f}")
        elif sig["action"] == "SELL":
            stop = entry + self.config.stop_atr_mult * atr
            tp = entry - (self.config.take_profit_atr_mult * atr if self.config.take_profit_atr_mult else None)
            if self.config.debug:
                print(f"[ZSCORE] SELL 신호: Entry={entry:.4f}, Stop={stop:.4f}, TP={tp:.4f}")

        # map action to canonical BUY/SELL/HOLD
        action = sig["action"]

        return {
            "name": "ZSCORE_MEAN_REVERSION",
            "action": action,
            "score": float(sig["score"]),
            "entry": float(entry),
            "stop": float(stop) if stop is not None else None,
            "tp": float(tp) if tp is not None else None,
            "context": {
                "last_z": last_z,
                "z_threshold": float(self.config.z_thresh),
                "atr": float(atr),
                "mode": "returns" if self.config.use_log else "pct",
                "window": int(self.config.window),
                "z_scale": float(self.config.z_scale),
                "pre_signal_pct": float(self.config.pre_signal_pct)
            }
        }
