import numpy as np
import pandas as pd

def price_volume_profile(df: pd.DataFrame, bins: int = 50, lookback: int = 300):
    """
    OHLCV를 이용해 최근 lookback 구간의 가격/거래량을 가격구간(bins)으로 히스토그램 근사.
    - 각 캔들의 거래량을 high~low 구간에 균등 분배(간이 근사).
    - 반환: (bin_edges, vol_hist)
    """
    use = df.iloc[-lookback:].copy() if len(df) >= lookback else df.copy()
    highs = use["high"].to_numpy()
    lows = use["low"].to_numpy()
    vols = use["volume"].to_numpy()

    price_min = float(np.min(lows))
    price_max = float(np.max(highs))
    edges = np.linspace(price_min, price_max, bins + 1)
    hist = np.zeros(bins, dtype=float)

    for h, l, v in zip(highs, lows, vols):
        if h <= l or v <= 0:
            continue
        # 해당 캔들이 커버하는 구간의 bin index
        idx_low = np.searchsorted(edges, l, side="right") - 1
        idx_high = np.searchsorted(edges, h, side="right") - 1
        idx_low = max(0, min(idx_low, bins - 1))
        idx_high = max(0, min(idx_high, bins - 1))
        width = max(1, idx_high - idx_low + 1)
        per_bin = v / width
        hist[idx_low:idx_high+1] += per_bin

    return edges, hist

def vpvr_key_levels(df: pd.DataFrame, bins: int = 50, lookback: int = 300, topn: int = 3):
    edges, hist = price_volume_profile(df, bins=bins, lookback=lookback)
    centers = (edges[:-1] + edges[1:]) / 2.0
    order = np.argsort(hist)[::-1]
    levels = [(centers[i], hist[i]) for i in order[:topn]]
    return levels  # [(price_level, volume), ...]
