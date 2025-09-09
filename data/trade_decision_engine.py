from typing import Dict, Any, Optional, List
from datetime import datetime
import math
from utils.time_manager import get_time_manager
from indicators.global_indicators import get_atr


class TradeDecisionEngine:
    """거래 결정을 담당하는 클래스 - 독립적 다중 포지션 지원"""
    
    # 전략 카테고리 정의
    STRATEGY_CATEGORIES = {
        "SHORT_TERM": {  # 3분봉 기반, 5-30분 보유
            "strategies": ["MACD_HISTOGRAM", "RSI_DIV", "VOL_SPIKE", "ORDERFLOW_CVD", 
                            "SESSION", "LIQUIDITY_GRAB", "VWAP_PINBALL"],
            "weight": 0.60,  # 전체 가중치의 60%
            "timeframe": "3m",
            "max_holding_minutes": 30,
            "leverage": 20,
            "risk_multiplier": 1.0
        },
        "MEDIUM_TERM": {  # 15분봉 기반, 1-4시간 보유
            "strategies": ["MULTI_TIMEFRAME", "HTF_TREND_15M", "BOLLINGER_SQUEEZE"],
            "weight": 0.25,  # 전체 가중치의 25%
            "timeframe": "15m",
            "max_holding_minutes": 240,
            "leverage": 10,
            "risk_multiplier": 0.7
        },
        "LONG_TERM": {  # 1시간봉+ 기반, 4-24시간 보유
            "strategies": ["OI_DELTA", "FUNDING_RATE", "VPVR", "ICHIMOKU"],
            "weight": 0.15,  # 전체 가중치의 15%
            "timeframe": "1h+",
            "max_holding_minutes": 1440,
            "leverage": 5,
            "risk_multiplier": 0.5
        }
    }
    
    def __init__(self):
        self.time_manager = get_time_manager()

    def decide_trade_realtime(
        self,
        signals: Dict[str, Dict[str, Any]],
        *,
        account_balance: float = 10000.0,
        base_risk_pct: float = 0.005,
        open_threshold: float = 0.25,
        immediate_threshold: float = 0.75,
        confirm_threshold: float = 0.3,
        confirm_window_sec: int = 180,
        session_priority: bool = True,
        news_event: bool = False,
    ) -> Dict[str, Any]:
        """실시간 거래 결정 - 독립적 다중 포지션 지원"""
        
        # 각 카테고리별로 독립적으로 결정
        decisions = {}
        
        for category_name, category_config in self.STRATEGY_CATEGORIES.items():
            decisions[category_name] = self._decide_category_trade(
                signals, category_name, category_config,
                account_balance, base_risk_pct, open_threshold,
                immediate_threshold, confirm_threshold, confirm_window_sec,
                session_priority, news_event
            )
        
        # 포지션 충돌 체크
        conflicts = self._check_position_conflicts(decisions)
        
        return {
            "decisions": decisions,
            "conflicts": conflicts,
            "meta": {
                "timestamp_utc": self.time_manager.get_current_time().isoformat(),
                "total_categories": len(decisions),
                "active_positions": sum(1 for d in decisions.values() if d["action"] != "HOLD")
            }
        }
    
    def _decide_category_trade(
        self, 
        signals: Dict[str, Dict[str, Any]], 
        category_name: str, 
        category_config: Dict[str, Any],
        account_balance: float,
        base_risk_pct: float,
        open_threshold: float,
        immediate_threshold: float,
        confirm_threshold: float,
        confirm_window_sec: int,
        session_priority: bool,
        news_event: bool
    ) -> Dict[str, Any]:
        """카테고리별 독립 거래 결정"""
        
        # 해당 카테고리 신호만 필터링
        category_signals = {k: v for k, v in signals.items() 
                           if k in category_config["strategies"]}
        
        if not category_signals:
            return self._create_category_hold_decision(category_name, category_config)
        
        # 카테고리 내 가중치 계산
        category_weights = self._calculate_category_weights(category_signals, category_config)
        
        # 카테고리별 점수 계산
        signed, raw, used_weight_sum = self._calculate_weighted_scores(category_signals, category_weights)
        
        if used_weight_sum <= 0:
            return self._create_category_hold_decision(category_name, category_config)
        
        net_score = sum(signed.values()) / max(1e-9, used_weight_sum)
        
        # 세션 오버라이드 체크 (단기 전략에만 적용)
        session_override = False
        session_action = None
        if category_name == "SHORT_TERM" and session_priority:
            session_override, session_action = self._check_session_override(raw, session_priority, immediate_threshold)
        
        # 확인 신호 계산
        now = self.time_manager.get_current_time()
        agree_counts = self._calculate_agreement_counts(raw, confirm_threshold, confirm_window_sec, now)
        
        # 최종 결정
        action, reason = self._make_category_decision(
            session_override, session_action, raw, net_score, open_threshold, agree_counts
        )
        
        # 포지션 크기 계산
        sizing = self._calculate_category_sizing(
            action, raw, category_config, account_balance, base_risk_pct
        )
        
        return {
            "action": action,
            "category": category_name,
            "net_score": round(net_score, 4),
            "raw": raw,
            "reason": "; ".join(reason),
            "sizing": sizing,
            "max_holding_minutes": category_config["max_holding_minutes"],
            "leverage": category_config["leverage"],
            "strategies_used": list(category_signals.keys()),
            "meta": {
                "timestamp_utc": now.isoformat(),
                "used_weight_sum": used_weight_sum,
                "timeframe": category_config["timeframe"]
            }
        }
    
    def _calculate_category_weights(self, category_signals: Dict[str, Dict[str, Any]], category_config: Dict[str, Any]) -> Dict[str, float]:
        """카테고리 내 가중치 계산"""
        strategies = category_config["strategies"]
        total_weight = category_config["weight"]
        
        # 카테고리 내 균등 분배 (필요시 개별 가중치 설정 가능)
        weight_per_strategy = total_weight / len(strategies) if strategies else 0
        
        weights = {}
        for strategy in strategies:
            if strategy in category_signals:
                weights[strategy] = weight_per_strategy
        
        return weights
    
    def _create_category_hold_decision(self, category_name: str, category_config: Dict[str, Any]) -> Dict[str, Any]:
        """카테고리별 HOLD 결정 생성"""
        return {
            "action": "HOLD",
            "category": category_name,
            "net_score": 0.0,
            "reason": f"no {category_name.lower()} strategies active",
            "sizing": {"qty": None, "risk_usd": 0.0, "entry_used": None, "stop_used": None},
            "max_holding_minutes": category_config["max_holding_minutes"],
            "leverage": category_config["leverage"],
            "strategies_used": [],
            "meta": {
                "timestamp_utc": self.time_manager.get_current_time().isoformat(),
                "timeframe": category_config["timeframe"]
            }
        }
    
    def _make_category_decision(self, session_override: bool, session_action: Optional[str], raw: Dict, net_score: float, open_threshold: float, agree_counts: Dict) -> tuple:
        """카테고리별 최종 결정"""
        action = "HOLD"
        reason = []
        
        if session_override and session_action is not None:
            action = "LONG" if session_action == "BUY" else "SHORT"
            session_rec = raw.get("SESSION")
            reason.append(f"SESSION strong override (score={session_rec.get('score')})")
        else:
            if net_score >= open_threshold:
                action = "LONG"
                reason.append(f"net_score {net_score:.3f} >= open_threshold {open_threshold}")
            elif net_score <= -open_threshold:
                action = "SHORT"
                reason.append(f"net_score {net_score:.3f} <= -open_threshold {-open_threshold}")
            else:
                # 조건부 진입
                if net_score > 0 and agree_counts["BUY"] >= 1 and net_score >= (open_threshold * 0.6):
                    action = "LONG"
                    reason.append(f"conditional LONG: net {net_score:.3f}, confirmations {agree_counts['BUY']}")
                elif net_score < 0 and agree_counts["SELL"] >= 1 and abs(net_score) >= (open_threshold * 0.6):
                    action = "SHORT"
                    reason.append(f"conditional SHORT: net {net_score:.3f}, confirmations {agree_counts['SELL']}")
                else:
                    action = "HOLD"
                    reason.append(f"net_score too small ({net_score:.3f}) or no confirmations")
                    
        return action, reason
    
    def _calculate_category_sizing(self, action: str, raw: Dict, category_config: Dict[str, Any], account_balance: float, base_risk_pct: float) -> Dict[str, Any]:
        """카테고리별 포지션 크기 계산"""
        if action == "HOLD":
            return {"qty": None, "risk_usd": 0.0, "entry_used": None, "stop_used": None}
        
        # 카테고리별 리스크 조정
        adjusted_risk_pct = base_risk_pct * category_config["risk_multiplier"]
        leverage = category_config["leverage"]
        
        # 진입/손절가 찾기
        entry_used = None
        stop_used = None
        
        for strategy_name, strategy_data in raw.items():
            if strategy_data.get("action") in ("BUY", "SELL"):
                if (action == "LONG" and strategy_data.get("action") == "BUY") or \
                   (action == "SHORT" and strategy_data.get("action") == "SELL"):
                    entry_used = strategy_data.get("entry")
                    stop_used = strategy_data.get("stop")
                    break
        
        # ATR을 사용한 폴백
        if entry_used is None or stop_used is None:
            try:
                atr_val = float(get_atr())
                any_price = self._find_any_price(raw)
                
                if entry_used is None and any_price is not None:
                    entry_used = any_price
                if stop_used is None and any_price is not None:
                    stop_used = self._calculate_stop_with_atr(entry_used, atr_val, action)
            except Exception:
                pass
        
        # 거래 규모 계산 (카테고리별 기본값)
        recommended_scale = 1.0  # 카테고리별로 기본 100% 규모
        
        # 수량 계산
        qty = self._calculate_quantity(entry_used, stop_used, action, account_balance, adjusted_risk_pct, leverage, recommended_scale)
        
        return {
            "qty": float(qty) if qty is not None else None,
            "risk_usd": round(float(account_balance * adjusted_risk_pct), 4),
            "entry_used": float(entry_used) if entry_used is not None else None,
            "stop_used": float(stop_used) if stop_used is not None else None,
            "leverage": leverage,
            "risk_multiplier": category_config["risk_multiplier"]
        }
    
    def _check_position_conflicts(self, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """포지션 충돌 체크"""
        long_categories = [k for k, v in decisions.items() if v["action"] == "LONG"]
        short_categories = [k for k, v in decisions.items() if v["action"] == "SHORT"]
        
        conflicts = []
        
        # 단기와 장기가 반대 방향일 때 경고
        if "SHORT_TERM" in long_categories and "LONG_TERM" in short_categories:
            conflicts.append("SHORT_TERM_LONG_vs_LONG_TERM_SHORT")
        elif "SHORT_TERM" in short_categories and "LONG_TERM" in long_categories:
            conflicts.append("SHORT_TERM_SHORT_vs_LONG_TERM_LONG")
        
        # 중기와 장기가 반대 방향일 때 경고
        if "MEDIUM_TERM" in long_categories and "LONG_TERM" in short_categories:
            conflicts.append("MEDIUM_TERM_LONG_vs_LONG_TERM_SHORT")
        elif "MEDIUM_TERM" in short_categories and "LONG_TERM" in long_categories:
            conflicts.append("MEDIUM_TERM_SHORT_vs_LONG_TERM_LONG")
        
        return {
            "has_conflicts": len(conflicts) > 0,
            "conflict_types": conflicts,
            "long_categories": long_categories,
            "short_categories": short_categories
        }

    def _calculate_weighted_scores(self, signals, weights):
        """가중 점수 계산"""
        signed = {}
        raw = {}
        used_weight_sum = 0.0

        for name, s in signals.items():
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
