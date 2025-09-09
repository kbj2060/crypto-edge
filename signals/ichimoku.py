
class IchimokuCfg:
    # Scalping-friendly Ichimoku settings (suitable for 1m-5m bars)
    tenkan_period: int = 7          # conversion line (fast)
    kijun_period: int = 22          # base line (slow)
    senkou_b_period: int = 44       # leading span B (longer term)
    displacement: int = 22          # cloud shift
    require_price_above_cloud: bool = False  # only take long signals above cloud
    require_cloud_thickness: float = 0.001  # minimal cloud thickness (in price units) to consider
    require_chikou_confirm: bool = False    # whether to require chikou span confirmation for scalping (often False to be responsive)
    max_holding_minutes: int = 30            # scalping max hold
    min_body_ratio: float = 0.06             # require decent candle body for cross confirmation
    use_cross_confirm: bool = False            # require Tenkan/Kijun cross to confirm entries
    debug: bool = False


# ichimoku_htf.py
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# 프로젝트 내 기존 유틸 사용 (당신의 다른 전략 파일들과 동일한 방식)
from data.binance_dataloader import BinanceDataLoader
from utils.time_manager import get_time_manager


# --------- 계산 유틸 ---------
def _highest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).max()

def _lowest(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).min()

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # 표준 ATR 계산 (고가/저가/종가 전일)
    high = pd.to_numeric(df["high"].astype(float))
    low = pd.to_numeric(df["low"].astype(float))
    close = pd.to_numeric(df["close"].astype(float))
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _ichimoku(df: pd.DataFrame, tenkan, kijun, senkou_b, shift):
    """
    반환:
      - tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_span
    주의: senkou_a/b는 forward shift(미래로)된 값을 그리지만,
          여기서는 '현재 봉 기준의 클라우드 레벨' 비교에 사용할 수 있도록
          계산 값 자체(시프트 전)를 함께 리턴합니다.
    """
    high = pd.to_numeric(df["high"].astype(float))
    low = pd.to_numeric(df["low"].astype(float))
    close = pd.to_numeric(df["close"].astype(float))

    tenkan_sen = (_highest(high, tenkan) + _lowest(low, tenkan)) / 2.0
    kijun_sen = (_highest(high, kijun) + _lowest(low, kijun)) / 2.0
    senkou_a_base = (tenkan_sen + kijun_sen) / 2.0
    senkou_b_base = (_highest(high, senkou_b) + _lowest(low, senkou_b)) / 2.0

    # 시프트 적용 버전(그림용)은 필요 시 사용 가능
    senkou_a = senkou_a_base.shift(shift)
    senkou_b = senkou_b_base.shift(shift)
    chikou_span = close.shift(-shift)

    return {
        "tenkan": tenkan_sen,
        "kijun": kijun_sen,
        "senkou_a_base": senkou_a_base,  # 현재 봉 기준 비교용
        "senkou_b_base": senkou_b_base,
        "senkou_a": senkou_a,            # 참고용
        "senkou_b": senkou_b,
        "chikou": chikou_span
    }

def _cloud_state(price: float, sa: float, sb: float) -> str:
    """
    현재가가 클라우드 대비 어디에 있는지 간단 판정
    """
    top = max(sa, sb)
    bottom = min(sa, sb)
    if price > top:
        return "ABOVE"
    elif price < bottom:
        return "BELOW"
    else:
        return "INSIDE"


# --------- 설정 ---------
@dataclass
class IchimokuHTFCfg:
    symbol: str = "ETHUSDC"
    main_interval: str = "4h"    # 주 신호: 4h (장기 전략에 적합)
    confirm_interval: Optional[str] = "1d"  # 1일봉 보조 필터 (장기 확인)
    # 표준 일목 파라미터 (4시간봉에 맞게 조정)
    tenkan: int = 9              # 전환선 (36시간)
    kijun: int = 26              # 기준선 (104시간)
    senkou_b: int = 52           # 선행스팬B (208시간)
    shift: int = 26              # 구름 시프트 (104시간)
    # 데이터 범위 (장기 분석)
    main_lookback_days: int = 60     # 4시간봉 60일 (240개 봉)
    confirm_lookback_days: int = 180 # 1일봉 180일
    # 위험관리 (장기 보유 4-24시간, 레버리지 5배)
    atr_period: int = 14
    atr_stop_mult: float = 2.0       # 손절폭: ATR x 2.0 (장기용으로 완화)
    tp_R1: float = 3.0               # 1차 목표: 3R (장기용으로 증가)
    tp_R2: float = 5.0               # 2차 목표: 5R (장기용으로 증가)
    min_score: float = 0.3           # 신호 최소 점수 완화 (0.5 -> 0.3)
    debug: bool = False               # 디버깅 활성화


class Ichimoku:
    """
    Ichimoku Cloud 고타임프레임 전략
    - 조건(롱/BULL):
        1) 현재가가 클라우드 "위"(ABOVE)
        2) 전환선(tenkan) > 기준선(kijun)
        3) 치코스팬(chikou)이 가격 위(=지연 확인)
      단축: 조건 2,3 중 1개만 충족해도 점수는 낮게 LONG 가능
    - 숏/BEAR는 반대 조건
    - 보조 필터(옵션): confirm_interval(4h)의 클라우드 방향이 동일하면 점수 가산
    - 출력: 기존 전략들과 같은 딕셔너리 포맷
    """
    def __init__(self, cfg: IchimokuHTFCfg = IchimokuHTFCfg()):
        self.cfg = cfg
        self.loader = BinanceDataLoader()
        self.tm = get_time_manager()

    def _load_ohlcv(self, interval: str, days: int) -> Optional[pd.DataFrame]:
        end_time = self.tm.get_current_time()
        start_time = end_time - timedelta(days=days)
        df = self.loader.fetch_data(
            interval=interval,
            symbol=self.cfg.symbol,
            start_time=start_time,
            end_time=end_time
        )
        if df is None or len(df) < max(self.cfg.kijun, self.cfg.senkou_b) + self.cfg.shift + 10:
            return None
        # 보정: 인덱스 정리
        df = df.copy()
        df["open"] = pd.to_numeric(df["open"].astype(float))
        df["high"] = pd.to_numeric(df["high"].astype(float))
        df["low"] = pd.to_numeric(df["low"].astype(float))
        df["close"] = pd.to_numeric(df["close"].astype(float))
        return df

    def _score_direction(self, price: float, ichi: Dict[str, pd.Series]) -> Dict[str, Any]:
        # 현재 봉의 값들
        tenkan = float(ichi["tenkan"].iloc[-1])
        kijun  = float(ichi["kijun"].iloc[-1])
        sa     = float(ichi["senkou_a_base"].iloc[-1])
        sb     = float(ichi["senkou_b_base"].iloc[-1])
        chikou = float(ichi["chikou"].iloc[-1]) if not np.isnan(ichi["chikou"].iloc[-1]) else np.nan

        cloud_pos = _cloud_state(price, sa, sb)
        bull = 0.0
        bear = 0.0

        # 클라우드 위치 가중 (완화)
        if cloud_pos == "ABOVE":
            bull += 0.4  # 0.5 -> 0.4
        elif cloud_pos == "BELOW":
            bear += 0.4  # 0.5 -> 0.4
        else:
            # INSIDE → 양쪽 약하게 (완화)
            bull += 0.2  # 0.15 -> 0.2
            bear += 0.2  # 0.15 -> 0.2

        # 전환선/기준선 (완화)
        tenkan_kijun_diff = abs(tenkan - kijun) / price  # 상대적 차이
        if tenkan > kijun:
            bull += 0.3 + min(0.2, tenkan_kijun_diff * 100)  # 0.25 -> 0.3 + 차이 보너스
        elif tenkan < kijun:
            bear += 0.3 + min(0.2, tenkan_kijun_diff * 100)  # 0.25 -> 0.3 + 차이 보너스

        # 치코스팬 (가격 위/아래) (완화)
        if not np.isnan(chikou):
            chikou_diff = abs(chikou - price) / price  # 상대적 차이
            if chikou > price:
                bull += 0.2 + min(0.1, chikou_diff * 50)  # 0.25 -> 0.2 + 차이 보너스
            elif chikou < price:
                bear += 0.2 + min(0.1, chikou_diff * 50)  # 0.25 -> 0.2 + 차이 보너스

        # 추가 점수: 구름 두께 (구름이 두꺼울수록 신호 강화)
        cloud_thickness = abs(sa - sb) / price
        if cloud_thickness > 0.01:  # 1% 이상 구름 두께
            if cloud_pos == "ABOVE":
                bull += 0.1
            elif cloud_pos == "BELOW":
                bear += 0.1

        # 정규화
        bull = min(1.0, max(0.0, bull))
        bear = min(1.0, max(0.0, bear))

        if bull > bear:
            action = "BUY"
            score = bull
        elif bear > bull:
            action = "SELL"
            score = bear
        else:
            action = "HOLD"
            score = 0.0

        if self.cfg.debug:
            print(f"[ICHIMOKU] 가격: {price:.2f}, 구름: {cloud_pos}, 전환선: {tenkan:.2f}, 기준선: {kijun:.2f}")
            print(f"[ICHIMOKU] Bull: {bull:.3f}, Bear: {bear:.3f}, 액션: {action}, 점수: {score:.3f}")

        return {
            "action": action,
            "score": float(score),
            "cloud_pos": cloud_pos,
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_a": sa,
            "senkou_b": sb,
            "chikou": chikou
        }

    def _is_hour_candle_close(self, hours: str) -> bool:
        """현재 시간이 시간봉 마감 시간인지 체크 (4h: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00)"""
        try:
            current_time = self.tm.get_current_time()
            current_hour = current_time.hour
            current_minute = current_time.minute

            # 1일봉 처리 (1d, 24h)
            if hours == '1d' or hours == '24h':
                return current_hour == 0 and current_minute == 0
            # 시간봉 처리 (4h, 1h 등)
            elif hours.endswith('h'):
                hours_int = int(hours[:-1])
                if hours_int == 4:
                    return current_hour % 4 == 0 and current_minute == 0
                else:
                    return current_hour % hours_int == 0 and current_minute == 0
            else:
                raise Exception(f"시간봉 마감 시간 체크 오류: {hours}는 지원되지 않는 형식입니다.")
                
        except Exception as e:
            print(f"시간봉 마감 시간 체크 오류: {e}")
            return False
        
    def on_kline_close_htf(self) -> Optional[Dict[str, Any]]:
        # 4시간봉 마감 체크 (4시간마다만 신호 생성)
        is_candle_close = self._is_hour_candle_close(self.cfg.main_interval)
        if not is_candle_close:
            if self.cfg.debug:
                print(f"[ICHIMOKU] 4시간봉 마감 대기 중... (이전 완성된 봉으로 분석)")
        
        # 1) 메인 TF(4h) 로드 및 일목 계산 (완성된 봉만 사용)
        df_main = self._load_ohlcv(self.cfg.main_interval, self.cfg.main_lookback_days)
        if df_main is None:
            if self.cfg.debug:
                print(f"[ICHIMOKU] 4시간봉 데이터 로드 실패")
            return None
        
        # 현재 마감 시간이 아니면 마지막 데이터 제거 (진행 중인 봉 제거)
        if not is_candle_close:
            df_main = df_main.iloc[:-1].copy()  # 마지막 행 제거
            if self.cfg.debug:
                print(f"[ICHIMOKU] 진행 중인 4시간봉 제거됨")
        
        # 마지막 4시간봉 데이터 확인
        if self.cfg.debug:
            current_time = self.tm.get_current_time()
            print(f"[ICHIMOKU] 현재 시간: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"[ICHIMOKU] 4시간봉 마감 여부: {is_candle_close}")
            print(f"[ICHIMOKU] 처리된 4시간봉 개수: {len(df_main)}")
            
            # 마지막 3개 봉의 시간 정보 출력
            if len(df_main) >= 3:
                print(f"[ICHIMOKU] 마지막 3개 완성된 4시간봉:")
                for i in range(-3, 0):
                    if 'timestamp' in df_main.columns:
                        timestamp = df_main.iloc[i]['timestamp']
                        close_price = df_main.iloc[i]['close']
                        print(f"  [{i}] 시간: {timestamp}, 종가: {close_price:.2f}")
                    else:
                        close_price = df_main.iloc[i]['close']
                        print(f"  [{i}] 종가: {close_price:.2f}")
        
        # 완성된 봉만 사용
        ichi_main = _ichimoku(df_main, self.cfg.tenkan, self.cfg.kijun, self.cfg.senkou_b, self.cfg.shift)
        price = float(df_main["close"].iloc[-1])  # 마지막 완성된 봉
        base = self._score_direction(price, ichi_main)
        
        if self.cfg.debug:
            status = "현재 마감" if is_candle_close else "이전 완성봉"
            print(f"[ICHIMOKU] 4시간봉 분석 완료 ({status}) - 액션: {base['action']}, 점수: {base['score']:.3f}")
            print(f"[ICHIMOKU] 사용된 가격: {price:.2f} (마지막 완성된 4시간봉)")

        # 2) 보조 TF(1d) 확인(옵션) - 완성된 봉만 사용
        confirm_dir = None
        if self.cfg.confirm_interval:
            is_confirm_close = self._is_hour_candle_close(hours=self.cfg.confirm_interval)
            df_c = self._load_ohlcv(self.cfg.confirm_interval, self.cfg.confirm_lookback_days)
            if df_c is not None:
                # 현재 마감 시간이 아니면 마지막 데이터 제거 (진행 중인 봉 제거)
                if not is_confirm_close:
                    df_c = df_c.iloc[:-1].copy()  # 마지막 행 제거
                    if self.cfg.debug:
                        print(f"[ICHIMOKU] 진행 중인 1일봉 제거됨")
                
                # 1일봉 데이터 확인
                if self.cfg.debug:
                    print(f"[ICHIMOKU] 1일봉 마감 여부: {is_confirm_close}")
                    print(f"[ICHIMOKU] 처리된 1일봉 개수: {len(df_c)}")
                    
                    # 마지막 3개 봉의 시간 정보 출력
                    if len(df_c) >= 3:
                        print(f"[ICHIMOKU] 마지막 3개 완성된 1일봉:")
                        for i in range(-3, 0):
                            if 'timestamp' in df_c.columns:
                                timestamp = df_c.iloc[i]['timestamp']
                                close_price = df_c.iloc[i]['close']
                                print(f"  [{i}] 시간: {timestamp}, 종가: {close_price:.2f}")
                            else:
                                close_price = df_c.iloc[i]['close']
                                print(f"  [{i}] 종가: {close_price:.2f}")
                
                # 완성된 봉만 사용
                ichi_c = _ichimoku(df_c, self.cfg.tenkan, self.cfg.kijun, self.cfg.senkou_b, self.cfg.shift)
                price_c = float(df_c["close"].iloc[-1])  # 마지막 완성된 봉
                confirm = self._score_direction(price_c, ichi_c)
                confirm_dir = confirm["action"]
                
                if self.cfg.debug:
                    confirm_status = "현재 마감" if is_confirm_close else "이전 완성봉"
                    print(f"[ICHIMOKU] 1일봉 사용된 가격: {price_c:.2f} ({confirm_status})")
                
                # 동방향이면 점수 가산, 반대면 감산 (완화)
                if confirm["action"] == base["action"] and confirm["action"] != "HOLD":
                    base["score"] = min(1.0, base["score"] + 0.2)  # 0.15 -> 0.2
                    if self.cfg.debug:
                        print(f"[ICHIMOKU] 1일봉 확인: 동방향 신호 강화 (+0.2)")
                elif confirm["action"] != "HOLD" and base["action"] != "HOLD" and confirm["action"] != base["action"]:
                    base["score"] = max(0.0, base["score"] - 0.15)  # 0.2 -> 0.15 (완화)
                    if self.cfg.debug:
                        print(f"[ICHIMOKU] 1일봉 확인: 반대방향 신호 약화 (-0.15)")
            else:
                if self.cfg.debug:
                    print(f"[ICHIMOKU] 1일봉 데이터 로드 실패")
        else:
            if self.cfg.debug:
                print(f"[ICHIMOKU] 1일봉 확인 비활성화")

        # 3) ATR 기반 손절/목표 (4h 기준) - 항상 마지막 완성된 봉 사용
        atr_series = _atr(df_main, self.cfg.atr_period)
        atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0
        stop = None
        tp1 = None
        tp2 = None
        if base["action"] == "BUY" and atr > 0:
            stop = price - (atr * self.cfg.atr_stop_mult)
            r = price - stop
            tp1 = price + r * self.cfg.tp_R1
            tp2 = price + r * self.cfg.tp_R2
        elif base["action"] == "SELL" and atr > 0:
            stop = price + (atr * self.cfg.atr_stop_mult)
            r = stop - price
            tp1 = price - r * self.cfg.tp_R1
            tp2 = price - r * self.cfg.tp_R2

        # 4) 최종 출력 (프로젝트 표준 딕셔너리)
        action = base["action"]
        score = float(base["score"])
        if score < self.cfg.min_score:
            action = "HOLD"
            if self.cfg.debug:
                print(f"[ICHIMOKU] 점수 부족으로 HOLD: {score:.3f} < {self.cfg.min_score}")
        else:
            if self.cfg.debug:
                print(f"[ICHIMOKU] 신호 생성: {action}, 점수: {score:.3f}")

        result = {
            "name": f"ICHIMOKU",
            "action": action,
            "score": score if action != "HOLD" else 0.0,
            "timestamp": self.tm.get_current_time(),
            "context": {
                "price": price,
                "cloud_pos": base["cloud_pos"],
                "tenkan": base["tenkan"],
                "kijun": base["kijun"],
                "senkou_a": base["senkou_a"],
                "senkou_b": base["senkou_b"],
                "chikou": base["chikou"],
                "confirm_tf": self.cfg.confirm_interval,
                "confirm_dir": confirm_dir,
                "atr": atr,
                "stop": stop,
                "tp_R1": tp1,
                "tp_R2": tp2
            }
        }
        
        return result
