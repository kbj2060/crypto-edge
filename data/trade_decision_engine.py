from typing import Dict, Any, Optional
from datetime import datetime
import math
from utils.time_manager import get_time_manager
from indicators.global_indicators import get_atr


class TradeDecisionEngine:
    """거래 결정을 담당하는 클래스"""
    
    def __init__(self):
        self.time_manager = get_time_manager()

    def decide_trade_realtime(
        self,
        signals: Dict[str, Dict[str, Any]],
        *,
        account_balance: float = 10000.0,
        base_risk_pct: float = 0.005,
        leverage: float = 20,
        weights: Optional[Dict[str, float]] = None,
        open_threshold: float = 0.5,
        immediate_threshold: float = 0.75,
        confirm_threshold: float = 0.45,
        confirm_window_sec: int = 180,
        session_priority: bool = True,
        news_event: bool = False,
    ) -> Dict[str, Any]:
        """실시간 거래 결정"""
        
        # 기본 가중치 설정
        priority_order = [
            "HTF_TREND_15M",      
            "ORDERFLOW_CVD",       
            "RSI_DIV",      
            "MACD_HISTOGRAM",              
            "FUNDING_RATE",
            "LIQUIDITY_GRAB",      
            "OI_DELTA",      
            "VWAP_PINBALL",    
            "SESSION",           
            "VPVR",
            "VOL_SPIKE",             
        ]

        default_weights = {
            # 핵심 추세/모멘텀 (45%)
            "HTF_TREND_15M":    0.15,  # 상위 추세 확인
            "ORDERFLOW_CVD":    0.15,  # 스마트머니 추적  
            "RSI_DIV":          0.15,  # 모멘텀 전환 신호
            
            # 신규 핵심 지표 (30%)
            "MACD_HISTOGRAM":   0.12,  # 🆕 모멘텀 가속도 (가장 중요)
            "FUNDING_RATE":     0.08,  # 🆕 시장 심리 (크립토 특화)
            "LIQUIDITY_GRAB":   0.06,  # 🆕 유동성 사냥 패턴
            "OI_DELTA":         0.04,  # 🆕 선물 미결제약정 변화
            
            # 레벨/세션 기반 (20%)
            "VWAP_PINBALL":     0.08,  # 동적 지지/저항
            "SESSION":          0.07,  # 세션 오프닝 레인지  
            "VPVR":             0.05,  # 볼륨 프로파일 레벨
            
            # 볼륨/변동성 (5%)
            "VOL_SPIKE":        0.05,  # 볼륨 스파이크
            
            # 제거 대상
            # "VPVR_MICRO":      제거 (VPVR와 중복)
            # "EMA_TREND_15M":   제거 (HTF_TREND_15M와 중복) 
            # "ZSCORE_MEAN_REVERSION": 제거 (VWAP_PINBALL과 유사)
            # "BB_SQUEEZE":      제거 (VOL_SPIKE로 대체)
            # "ICHIMOKU":        제거 (단타에 부적합)
        }

        if weights is None:
            weights = default_weights.copy()
        else:
            for k, v in default_weights.items():
                weights.setdefault(k, v)

        # after building raw (or before signed calc)
        ovf_strategy = signals.get("ORDERFLOW_CVD")
        vwap_strategy = signals.get("VWAP_PINBALL")
        if ovf_strategy and vwap_strategy and ovf_strategy.get("action") in ("BUY","SELL") and vwap_strategy.get("action") in ("BUY","SELL"):
            if ovf_strategy.get("action") != vwap_strategy.get("action"):
                weights["VWAP_PINBALL"] = 0.0


        now = self.time_manager.get_current_time()

        # 전략별 가중 점수 계산
        signed, raw, used_weight_sum = self._calculate_weighted_scores(signals, weights)

        # 가중치가 없으면 HOLD
        if used_weight_sum <= 0:
            return self._create_hold_decision(raw)

        net = sum(signed.values()) / max(1e-9, used_weight_sum)

        # 세션 오버라이드 체크
        session_override, session_action = self._check_session_override(raw, session_priority, immediate_threshold)

        # 확인 신호 계산
        agree_counts = self._calculate_agreement_counts(raw, confirm_threshold, confirm_window_sec, now)

        # 충돌 감지
        oppositions = self._detect_oppositions(raw)

        # 거래 규모 계산
        recommended_scale = self._calculate_trade_scale(net, oppositions, raw, used_weight_sum)

        # 최종 결정
        action, reason = self._make_final_decision(
            session_override, session_action, raw, net, open_threshold, agree_counts
        )

        # 포지션 크기 계산
        sizing = self._calculate_sizing(
            action, raw, priority_order, account_balance, base_risk_pct, leverage, recommended_scale
        )

        return {
            "action": action,
            "net_score": round(net, 4),
            "raw": raw,
            "reason": "; ".join(reason),
            "recommended_trade_scale": round(recommended_scale, 3),
            "sizing": sizing,
            "oppositions": oppositions,
            "agree_counts": agree_counts,
            "meta": {"timestamp_utc": now.isoformat(), "used_weight_sum": used_weight_sum}
        }

    def _calculate_weighted_scores(self, signals, weights):
        """가중 점수 계산"""
        signed = {}
        raw = {}
        used_weight_sum = 0.0

        for name, s in signals.items():
            print(name, s)
            name = name.upper()
            action = (s.get("action")).upper()
            score = float(s.get("score"))
            w = float(weights.get(name))
            
            # 부호 있는 값 계산
            sign = 0
            if action == "BUY":
                sign = 1
            elif action == "SELL":
                sign = -1
            val = sign * score * w
            signed[name] = val
            raw[name] = {
                "action": action if action else None,
                "score": score,
                "weight": w,
                "entry": s.get("entry"),
                "stop": s.get("stop"),
                "timestamp": self.time_manager.get_current_time()
            }
            if w > 0:
                used_weight_sum += w
                
        return signed, raw, used_weight_sum

    def _create_hold_decision(self, raw):
        """HOLD 결정 생성"""
        return {
            "action": "HOLD",
            "net_score": 0.0,
            "reason": "no recognized weighted strategies",
            "recommended_trade_scale": 0.0,
            "sizing": {"qty": None, "risk_usd": 0.0, "entry_used": None, "stop_used": None},
            "raw": raw
        }

    def _check_session_override(self, raw, session_priority, immediate_threshold):
        """세션 오버라이드 체크"""
        session_override = False
        session_action = None
        
        if session_priority:
            session_rec = raw.get("SESSION")
            if session_rec:
                sess_act = session_rec.get("action")
                sess_score = float(session_rec.get("score") or 0.0)
                
                if sess_act in ("BUY", "SELL") and sess_score >= immediate_threshold:
                    # 반대 신호 체크
                    opp_strong = False
                    for nm, r in raw.items():
                        if nm == "SESSION": 
                            continue
                        if (r.get("action") and r.get("action") != sess_act and 
                            float(r.get("score") or 0.0) >= 0.60):
                            opp_strong = True
                            break
                    
                    if not opp_strong:
                        session_override = True
                        session_action = sess_act
                        
        return session_override, session_action

    def _calculate_agreement_counts(self, raw, confirm_threshold, confirm_window_sec, now):
        """동의 신호 개수 계산"""
        agree_counts = {"BUY": 0, "SELL": 0}
        
        for nm, r in raw.items():
            act = r.get("action")
            if act not in ("BUY", "SELL"):
                continue
            sc = float(r.get("score") or 0.0)
            ts = r.get("timestamp")
            
            # 시간 기반 확인
            if ts is not None and isinstance(ts, datetime):
                if abs((now - ts).total_seconds()) > confirm_window_sec:
                    continue
            if sc >= confirm_threshold:
                agree_counts[act] += 1
                
        return agree_counts

    def _detect_oppositions(self, raw):
        """충돌 신호 감지"""
        oppositions = []
        for nm, r in raw.items():
            act = r.get("action")
            sc = float(r.get("score") or 0.0)
            if act in ("BUY", "SELL") and sc >= 0.5:
                oppositions.append((nm, act, sc))
        return oppositions

    def _calculate_trade_scale(self, net, oppositions, raw, used_weight_sum):
        """거래 규모 계산"""
        # 기본 규모
        base_scale = min(1.0, max(0.0, abs(net) / 0.75))
        
        # 충돌 페널티
        if len(oppositions) >= 2:
            conflict_penalty = 0.25
        elif len(oppositions) == 1:
            conflict_penalty = 0.6
        else:
            conflict_penalty = 1.0
            
        return max(0.0, min(1.0, base_scale * conflict_penalty))

    def _make_final_decision(self, session_override, session_action, raw, net, open_threshold, agree_counts):
        """최종 결정"""
        action = "HOLD"
        reason = []
        
        if session_override:
            action = "LONG" if session_action == "BUY" else "SHORT"
            session_rec = raw.get("SESSION")
            reason.append(f"SESSION strong override (score={session_rec.get('score')})")
        else:
            if net >= open_threshold:
                action = "LONG"
                reason.append(f"net_score {net:.3f} >= open_threshold {open_threshold}")
            elif net <= -open_threshold:
                action = "SHORT"
                reason.append(f"net_score {net:.3f} <= -open_threshold {-open_threshold}")
            else:
                # 조건부 진입
                if net > 0 and agree_counts["BUY"] >= 1 and net >= (open_threshold * 0.6):
                    action = "LONG"
                    reason.append(f"conditional LONG: net {net:.3f}, confirmations {agree_counts['BUY']}")
                elif net < 0 and agree_counts["SELL"] >= 1 and abs(net) >= (open_threshold * 0.6):
                    action = "SHORT"
                    reason.append(f"conditional SHORT: net {net:.3f}, confirmations {agree_counts['SELL']}")
                else:
                    action = "HOLD"
                    reason.append(f"net_score too small ({net:.3f}) or no confirmations")
                    
        return action, reason

    def _calculate_sizing(self, action, raw, priority_order, account_balance, base_risk_pct, leverage, recommended_scale):
        """포지션 크기 계산"""
        entry_used = None
        stop_used = None
        selected_strategy = None

        # 우선순위에 따라 전략 선택
        for pname in priority_order:
            r = raw.get(pname)
            if r and r.get("action") and r.get("action") in ("BUY", "SELL"):
                if action == "HOLD":
                    selected_strategy = pname
                    break
                if (action == "LONG" and r.get("action") == "BUY") or (action == "SHORT" and r.get("action") == "SELL"):
                    selected_strategy = pname
                    break
                    
        if selected_strategy:
            r = raw.get(selected_strategy)
            entry_used = r.get("entry")
            stop_used = r.get("stop")

        # ATR을 사용한 폴백
        if (entry_used is None or stop_used is None):
            try:
                atr_val = float(get_atr())
                any_price = self._find_any_price(raw)
                
                if entry_used is None and any_price is not None:
                    entry_used = any_price
                if stop_used is None and any_price is not None:
                    stop_used = self._calculate_stop_with_atr(entry_used, atr_val, action)
            except Exception:
                pass

        # 수량 계산
        qty = self._calculate_quantity(entry_used, stop_used, action, account_balance, base_risk_pct, leverage, recommended_scale)

        return {
            "qty": float(qty) if qty is not None else None,
            "risk_usd": round(float(account_balance * base_risk_pct), 4),
            "entry_used": float(entry_used) if entry_used is not None else None,
            "stop_used": float(stop_used) if stop_used is not None else None,
            "recommended_scale": round(recommended_scale, 3)
        }

    def _find_any_price(self, raw):
        """어떤 가격이라도 찾기"""
        for nm, r in raw.items():
            if r.get("entry") is not None:
                return float(r.get("entry"))
        for nm, r in raw.items():
            if r.get("score", 0) > 0:
                any_price = r.get("entry") or r.get("stop")
                if any_price is not None:
                    return float(any_price)
        return None

    def _calculate_stop_with_atr(self, entry_used, atr_val, action):
        """ATR을 사용한 스탑 계산"""
        if atr_val is None or math.isnan(atr_val):
            atr_val = max(1.0, 0.5 * abs(entry_used) * 0.001)
            
        if action == "LONG":
            return entry_used - 1.5 * atr_val
        elif action == "SHORT":
            return entry_used + 1.5 * atr_val
        else:
            return None

    def _calculate_quantity(self, entry_used, stop_used, action, account_balance, base_risk_pct, leverage, recommended_scale):
        """수량 계산"""
        qty = None
        risk_usd = account_balance * float(base_risk_pct)
        
        if (entry_used is not None and stop_used is not None and 
            entry_used != stop_used and action in ("LONG", "SHORT")):
            distance = abs(entry_used - stop_used)
            if distance > 0:
                qty = risk_usd / distance
                qty = qty * recommended_scale * leverage
        else:
            qty = None
            
        return qty
