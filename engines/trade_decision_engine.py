from typing import Dict, Any, Optional
from datetime import datetime
import math
from utils.time_manager import get_time_manager
from indicators.global_indicators import get_atr
from engines.short_term_synergy_engine import ShortTermSynergyEngine, SynergyConfig
from engines.medium_term_synergy_engine import MediumTermSynergyEngine, MediumTermConfig
from engines.long_term_synergy_engine import LongTermSynergyEngine, LongTermConfig


class TradeDecisionEngine:
    """거래 결정을 담당하는 클래스 - 독립적 다중 포지션 지원"""
    
    # 전략 카테고리 정의
    STRATEGY_CATEGORIES = {
        "SHORT_TERM": {  # 3분봉 기반, 5-30분 보유
            "strategies": ["VOL_SPIKE", "ORDERFLOW_CVD", "VPVR_MICRO",
                          "SESSION", "LIQUIDITY_GRAB", "VWAP_PINBALL", "ZSCORE_MEAN_REVERSION"],
            "weight": 0.60,
            "timeframe": "3m",
            "max_holding_minutes": 30,
            "leverage": 20,
            "risk_multiplier": 1.0
        },
        "MEDIUM_TERM": {  # 15분봉 기반, 1-4시간 보유
            "strategies": ["MULTI_TIMEFRAME", "HTF_TREND", "BOLLINGER_SQUEEZE", 
                          "SUPPORT_RESISTANCE", "EMA_CONFLUENCE"],
            "weight": 0.25,
            "timeframe": "15m",
            "max_holding_minutes": 240,
            "leverage": 10,
            "risk_multiplier": 0.7
        },
        "LONG_TERM": {  # 1시간봉+ 기반, 4-24시간 보유
            "strategies": ["OI_DELTA", "VPVR", "ICHIMOKU", "FUNDING_RATE"],
            "weight": 0.15,
            "timeframe": "1h+",
            "max_holding_minutes": 1440,
            "leverage": 5,
            "risk_multiplier": 0.5
        }
    }
    
    # 고정 스키마 정의 - 항상 동일한 구조 보장
    FIXED_SCHEMA = {
        "action": "HOLD",
        "category": "",
        "net_score": 0.0,
        "raw": {
            # 모든 가능한 전략들을 고정으로 정의
            "vol_spike": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "orderflow_cvd": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "vpvr_micro": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "session": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "liquidity_grab": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "vwap_pinball": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "zscore_mean_reversion": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "multi_timeframe": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "htf_trend": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "bollinger_squeeze": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "support_resistance": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "ema_confluence": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "oi_delta": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "vpvr": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "ichimoku": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None},
            "funding_rate": {"action": "HOLD", "score": 0.0, "entry": None, "stop": None, "weight": 0.0, "timestamp": None}
        },
        "reason": "",
        "sizing": {
            "qty": None,
            "risk_usd": 0.0,
            "entry_used": None,
            "stop_used": None,
            "leverage": 0,
            "risk_multiplier": 0.0
        },
        "max_holding_minutes": 0,
        "leverage": 0,
        "strategies_used": [],
        "meta": {
            "timestamp_utc": "",
            "timeframe": "",
            "used_weight_sum": 0.0,
            "synergy_meta": {
                "confidence": "LOW",
                "market_context": "NEUTRAL",
                "conflicts_detected": [],
                "bonuses_applied": [],
                "buy_score": 0.0,
                "sell_score": 0.0,
                "signals_used": 0,
                "institutional_bias": "NEUTRAL",
                "macro_trend_strength": "WEAK"
            }
        }
    }
    
    def __init__(self):
        self.time_manager = get_time_manager()
        # 각 카테고리별 시너지 엔진 초기화
        self.short_term_engine = ShortTermSynergyEngine(SynergyConfig())
        self.medium_term_engine = MediumTermSynergyEngine(MediumTermConfig())
        self.long_term_engine = LongTermSynergyEngine(LongTermConfig())

    def _create_fixed_decision_schema(self, category_name: str) -> Dict[str, Any]:
        """고정 스키마 기반으로 결정 객체 생성"""
        import copy
        decision = copy.deepcopy(self.FIXED_SCHEMA)
        decision["category"] = category_name
        decision["meta"]["timestamp_utc"] = self.time_manager.get_current_time().isoformat()
        decision["meta"]["timeframe"] = self.STRATEGY_CATEGORIES[category_name]["timeframe"]
        return decision

    def _normalize_single_decision(self, decision: Dict[str, Any], category_name: str) -> Dict[str, Any]:
        """단일 결정을 고정 스키마로 정규화"""
        # 고정 스키마로 시작
        normalized = self._create_fixed_decision_schema(category_name)
        
        # 실제 값들로 업데이트 (안전하게)
        self._safe_update_dict(normalized, decision, ["action", "net_score", "reason"])
        
        # raw 섹션 업데이트 - 모든 전략 키가 항상 존재하도록 보장
        if "raw" in decision and isinstance(decision["raw"], dict):
            for strategy_name, strategy_data in decision["raw"].items():
                normalized_strategy = strategy_name.lower().replace("_or", "")  # session_or -> session
                if normalized_strategy in normalized["raw"]:
                    # 항상 모든 필드를 업데이트 (None 값도 포함)
                    if isinstance(strategy_data, dict):
                        for field in ["action", "score", "entry", "stop", "weight", "timestamp"]:
                            if field in strategy_data:
                                normalized["raw"][normalized_strategy][field] = strategy_data[field]
                            # 필드가 없어도 기본값 유지됨
        
        # 모든 전략 키에 대해 명시적으로 기본값 보장
        for strategy_key in normalized["raw"]:
            strategy_obj = normalized["raw"][strategy_key]
            # 필수 필드들이 모두 있는지 확인하고 없으면 기본값 설정
            required_fields = {
                "action": "HOLD",
                "score": 0.0,
                "entry": None,
                "stop": None,
                "weight": 0.0,
                "timestamp": None
            }
            for field, default_val in required_fields.items():
                if field not in strategy_obj:
                    strategy_obj[field] = default_val
        
        # sizing 섹션 업데이트
        if "sizing" in decision and isinstance(decision["sizing"], dict):
            self._safe_update_dict(normalized["sizing"], decision["sizing"],
                                 ["qty", "risk_usd", "entry_used", "stop_used", "leverage", "risk_multiplier"])
        
        # sizing의 필수 필드들 보장
        required_sizing_fields = {
            "qty": None,
            "risk_usd": 0.0,
            "entry_used": None,
            "stop_used": None,
            "leverage": 0,
            "risk_multiplier": 0.0
        }
        for field, default_val in required_sizing_fields.items():
            if field not in normalized["sizing"]:
                normalized["sizing"][field] = default_val
        
        # meta 섹션 업데이트
        if "meta" in decision and isinstance(decision["meta"], dict):
            self._safe_update_dict(normalized["meta"], decision["meta"],
                                 ["used_weight_sum"])
            
            # synergy_meta 업데이트
            if "synergy_meta" in decision["meta"] and isinstance(decision["meta"]["synergy_meta"], dict):
                self._safe_update_dict(normalized["meta"]["synergy_meta"], decision["meta"]["synergy_meta"],
                                     ["confidence", "market_context", "conflicts_detected", "bonuses_applied",
                                      "buy_score", "sell_score", "signals_used", "institutional_bias", "macro_trend_strength"])
        
        # synergy_meta의 필수 필드들 보장
        required_synergy_fields = {
            "confidence": "LOW",
            "market_context": "NEUTRAL",
            "conflicts_detected": [],
            "bonuses_applied": [],
            "buy_score": 0.0,
            "sell_score": 0.0,
            "signals_used": 0,
            "institutional_bias": "NEUTRAL",
            "macro_trend_strength": "WEAK"
        }
        for field, default_val in required_synergy_fields.items():
            if field not in normalized["meta"]["synergy_meta"]:
                normalized["meta"]["synergy_meta"][field] = default_val
        
        # 기타 필드들 업데이트
        for field in ["max_holding_minutes", "leverage", "strategies_used"]:
            if field in decision:
                normalized[field] = decision[field]
            elif field == "strategies_used" and field not in normalized:
                normalized[field] = []
        
        return normalized

    def _safe_update_dict(self, target: Dict[str, Any], source: Dict[str, Any], allowed_keys: list) -> None:
        """안전하게 딕셔너리 업데이트 (허용된 키만) - None 값도 명시적으로 설정"""
        for key in allowed_keys:
            if key in source:
                # None 값도 명시적으로 설정하여 키 누락 방지
                target[key] = source[key]
            # source에 키가 없어도 target의 기본값 유지 (이미 고정 스키마에서 설정됨)

    def _normalize_decisions_schema(self, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """모든 카테고리 결정을 고정 스키마로 정규화"""
        normalized = {}
        
        # 모든 카테고리에 대해 고정 스키마 적용
        for category_name in self.STRATEGY_CATEGORIES.keys():
            if category_name.lower() in decisions:
                # 실제 결정이 있는 경우
                normalized[category_name.lower()] = self._normalize_single_decision(
                    decisions[category_name.lower()], category_name
                )
            else:
                # 결정이 없는 경우 기본 HOLD 결정 생성
                normalized[category_name.lower()] = self._create_category_hold_decision(category_name)
        
        return normalized

    def decide_trade_realtime(
        self,
        signals: Dict[str, Dict[str, Any]],
        *,
        account_balance: float = 10000.0,
        base_risk_pct: float = 0.005,
        open_threshold: float = 0.15,
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
            decisions[category_name.lower()] = self._decide_category_trade(
                signals, category_name, category_config,
                account_balance, base_risk_pct, open_threshold,
                immediate_threshold, confirm_threshold, confirm_window_sec,
                session_priority, news_event
            )
        
        # 포지션 충돌 체크
        conflicts = self._check_position_conflicts(decisions)
        
        # 고정 스키마로 정규화 (핵심!)
        normalized_decisions = self._normalize_decisions_schema(decisions)
        
        # 고정 conflicts 스키마
        normalized_conflicts = {
            "has_conflicts": conflicts.get("has_conflicts", False),
            "conflict_types": conflicts.get("conflict_types", []),
            "long_categories": conflicts.get("long_categories", []),
            "short_categories": conflicts.get("short_categories", []),
            "conflict_details": conflicts.get("conflict_details", [])
        }
        
        # 고정 meta 스키마
        normalized_meta = {
            "timestamp_utc": self.time_manager.get_current_time().isoformat(),
            "total_categories": 3,  # 고정값
            "active_positions": sum(1 for d in normalized_decisions.values() if d["action"] != "HOLD"),
            "session_priority": session_priority,
            "news_event": news_event
        }
        
        return {
            "decisions": normalized_decisions,
            "conflicts": normalized_conflicts,
            "meta": normalized_meta
        }
    
    def _create_category_hold_decision(self, category_name: str) -> Dict[str, Any]:
        """카테고리별 HOLD 결정 생성 (고정 스키마 사용)"""
        decision = self._create_fixed_decision_schema(category_name)
        decision["reason"] = f"no {category_name.lower()} strategies active"
        decision["max_holding_minutes"] = self.STRATEGY_CATEGORIES[category_name]["max_holding_minutes"]
        decision["leverage"] = self.STRATEGY_CATEGORIES[category_name]["leverage"]
        return decision

    # 나머지 메서드들은 기존과 동일하게 유지...
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
                            if k.upper() in category_config["strategies"]}
        
        if not category_signals:
            return self._create_category_hold_decision(category_name)
        
        # 각 카테고리별 시너지 엔진 사용
        if category_name == "SHORT_TERM":
            return self._decide_short_term_with_synergy(
                category_signals, category_config, account_balance, base_risk_pct,
                open_threshold, immediate_threshold, confirm_threshold, confirm_window_sec,
                session_priority, news_event
            )
        elif category_name == "MEDIUM_TERM":
            return self._decide_medium_term_with_synergy(
                category_signals, category_config, account_balance, base_risk_pct,
                open_threshold, immediate_threshold, confirm_threshold, confirm_window_sec,
                session_priority, news_event
            )
        elif category_name == "LONG_TERM":
            return self._decide_long_term_with_synergy(
                category_signals, category_config, account_balance, base_risk_pct,
                open_threshold, immediate_threshold, confirm_threshold, confirm_window_sec,
                session_priority, news_event
            )
        
        # 기본 로직 (백업용)
        return self._create_category_hold_decision(category_name)

    # 시너지 엔진 메서드들도 고정 스키마 반환하도록 수정
    def _decide_short_term_with_synergy(
        self,
        category_signals: Dict[str, Dict[str, Any]],
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
        """ShortTermSynergyEngine을 사용한 단기 전략 결정"""
        
        # 고정 스키마로 시작
        decision = self._create_fixed_decision_schema("SHORT_TERM")
        
        # 시너지 엔진 실행
        synergy_result = self.short_term_engine.calculate_synergy_score(category_signals)
        
        # 결과 업데이트
        decision["net_score"] = round(synergy_result['net_score'], 4)
        
        # 액션 결정
        action = synergy_result['action']
        if action == 'BUY':
            decision["action"] = 'LONG'
            decision["reason"] = f"synergy BUY: net_score={synergy_result['net_score']:.3f}, confidence={synergy_result['confidence']}"
        elif action == 'SELL':
            decision["action"] = 'SHORT'
            decision["reason"] = f"synergy SELL: net_score={synergy_result['net_score']:.3f}, confidence={synergy_result['confidence']}"
        else:
            decision["action"] = 'HOLD'
            decision["reason"] = f"synergy HOLD: net_score={synergy_result['net_score']:.3f}, confidence={synergy_result['confidence']}"
        
        # raw 신호 업데이트 (고정 스키마의 해당 키만) - 모든 필드 보장
        for name, signal_data in category_signals.items():
            normalized_name = name.lower().replace("_or", "")
            if normalized_name in decision["raw"]:
                # 모든 필드를 명시적으로 설정 (None 값도 포함)
                decision["raw"][normalized_name]["action"] = signal_data.get("action", "HOLD")
                decision["raw"][normalized_name]["score"] = signal_data.get("score", 0.0)
                decision["raw"][normalized_name]["entry"] = signal_data.get("entry")  # None도 허용
                decision["raw"][normalized_name]["stop"] = signal_data.get("stop")    # None도 허용
                decision["raw"][normalized_name]["weight"] = 1.0
                decision["raw"][normalized_name]["timestamp"] = self.time_manager.get_current_time()
        
        # 사용되지 않은 전략들도 기본값으로 명시적 설정
        for strategy_key in decision["raw"]:
            if not any(name.lower().replace("_or", "") == strategy_key for name in category_signals.keys()):
                decision["raw"][strategy_key]["action"] = "HOLD"
                decision["raw"][strategy_key]["score"] = 0.0
                decision["raw"][strategy_key]["entry"] = None
                decision["raw"][strategy_key]["stop"] = None
                decision["raw"][strategy_key]["weight"] = 0.0
                decision["raw"][strategy_key]["timestamp"] = None
        
        # 포지션 크기 계산
        sizing = self._calculate_category_sizing(
            decision["action"], decision["raw"], category_config, account_balance, base_risk_pct
        )
        decision["sizing"].update(sizing)
        
        # 기타 필드 업데이트
        decision["max_holding_minutes"] = category_config["max_holding_minutes"]
        decision["leverage"] = category_config["leverage"]
        decision["strategies_used"] = list(category_signals.keys())
        
        # 시너지 메타데이터
        decision["meta"]["synergy_meta"].update({
            "confidence": synergy_result['confidence'],
            "market_context": synergy_result['market_context'],
            "conflicts_detected": synergy_result['conflicts_detected'],
            "buy_score": synergy_result.get('buy_score', 0),
            "sell_score": synergy_result.get('sell_score', 0),
            "signals_used": synergy_result.get('signals_used', 0)
        })
        
        return decision

    def _decide_medium_term_with_synergy(
        self,
        category_signals: Dict[str, Dict[str, Any]],
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
        """MediumTermSynergyEngine을 사용한 중기 전략 결정"""
        
        # 고정 스키마로 시작
        decision = self._create_fixed_decision_schema("MEDIUM_TERM")
        
        # 시너지 엔진 실행
        synergy_result = self.medium_term_engine.calculate_synergy_score(category_signals)
        
        # 결과 업데이트
        decision["net_score"] = round(synergy_result['net_score'], 4)
        
        # 액션 결정
        action = synergy_result['action']
        if action == 'BUY':
            decision["action"] = 'LONG'
            decision["reason"] = f"medium synergy BUY: net_score={synergy_result['net_score']:.3f}, confidence={synergy_result['confidence']}"
        elif action == 'SELL':
            decision["action"] = 'SHORT'
            decision["reason"] = f"medium synergy SELL: net_score={synergy_result['net_score']:.3f}, confidence={synergy_result['confidence']}"
        else:
            decision["action"] = 'HOLD'
            decision["reason"] = f"medium synergy HOLD: net_score={synergy_result['net_score']:.3f}, confidence={synergy_result['confidence']}"
        
        # raw 신호 업데이트 (고정 스키마의 해당 키만) - 모든 필드 보장
        for name, signal_data in category_signals.items():
            normalized_name = name.lower()
            if normalized_name in decision["raw"]:
                # 모든 필드를 명시적으로 설정 (None 값도 포함)
                decision["raw"][normalized_name]["action"] = signal_data.get("action", "HOLD")
                decision["raw"][normalized_name]["score"] = signal_data.get("score", 0.0)
                decision["raw"][normalized_name]["entry"] = signal_data.get("entry")  # None도 허용
                decision["raw"][normalized_name]["stop"] = signal_data.get("stop")    # None도 허용
                decision["raw"][normalized_name]["weight"] = 1.0
                decision["raw"][normalized_name]["timestamp"] = self.time_manager.get_current_time()
        
        # 사용되지 않은 전략들도 기본값으로 명시적 설정
        for strategy_key in decision["raw"]:
            if not any(name.lower() == strategy_key for name in category_signals.keys()):
                decision["raw"][strategy_key]["action"] = "HOLD"
                decision["raw"][strategy_key]["score"] = 0.0
                decision["raw"][strategy_key]["entry"] = None
                decision["raw"][strategy_key]["stop"] = None
                decision["raw"][strategy_key]["weight"] = 0.0
                decision["raw"][strategy_key]["timestamp"] = None
        
        # 포지션 크기 계산
        sizing = self._calculate_category_sizing(
            decision["action"], decision["raw"], category_config, account_balance, base_risk_pct
        )
        decision["sizing"].update(sizing)
        
        # 기타 필드 업데이트
        decision["max_holding_minutes"] = category_config["max_holding_minutes"]
        decision["leverage"] = category_config["leverage"]
        decision["strategies_used"] = list(category_signals.keys())
        
        # 시너지 메타데이터
        decision["meta"]["synergy_meta"].update({
            "confidence": synergy_result['confidence'],
            "market_context": synergy_result['market_context'],
            "conflicts_detected": synergy_result['conflicts_detected'],
            "bonuses_applied": synergy_result.get('bonuses_applied', []),
            "buy_score": synergy_result.get('buy_score', 0),
            "sell_score": synergy_result.get('sell_score', 0),
            "signals_used": synergy_result.get('signals_used', 0)
        })
        
        return decision

    def _decide_long_term_with_synergy(
        self,
        category_signals: Dict[str, Dict[str, Any]],
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
        """LongTermSynergyEngine을 사용한 장기 전략 결정"""
        
        # 고정 스키마로 시작
        decision = self._create_fixed_decision_schema("LONG_TERM")
        
        # 시너지 엔진 실행
        synergy_result = self.long_term_engine.calculate_synergy_score(category_signals)
        
        # 결과 업데이트
        decision["net_score"] = round(synergy_result['net_score'], 4)
        
        # 액션 결정
        action = synergy_result['action']
        if action == 'BUY':
            decision["action"] = 'LONG'
            decision["reason"] = f"long synergy BUY: net_score={synergy_result['net_score']:.3f}, confidence={synergy_result['confidence']}"
        elif action == 'SELL':
            decision["action"] = 'SHORT'
            decision["reason"] = f"long synergy SELL: net_score={synergy_result['net_score']:.3f}, confidence={synergy_result['confidence']}"
        else:
            decision["action"] = 'HOLD'
            decision["reason"] = f"long synergy HOLD: net_score={synergy_result['net_score']:.3f}, confidence={synergy_result['confidence']}"
        
        # raw 신호 업데이트 (고정 스키마의 해당 키만) - 모든 필드 보장  
        for name, signal_data in category_signals.items():
            normalized_name = name.lower()
            if normalized_name in decision["raw"]:
                # 모든 필드를 명시적으로 설정 (None 값도 포함)
                decision["raw"][normalized_name]["action"] = signal_data.get("action", "HOLD")
                decision["raw"][normalized_name]["score"] = signal_data.get("score", 0.0)
                decision["raw"][normalized_name]["entry"] = signal_data.get("entry")  # None도 허용
                decision["raw"][normalized_name]["stop"] = signal_data.get("stop")    # None도 허용
                decision["raw"][normalized_name]["weight"] = 1.0
                decision["raw"][normalized_name]["timestamp"] = self.time_manager.get_current_time()
        
        # 사용되지 않은 전략들도 기본값으로 명시적 설정
        for strategy_key in decision["raw"]:
            if not any(name.lower() == strategy_key for name in category_signals.keys()):
                decision["raw"][strategy_key]["action"] = "HOLD"
                decision["raw"][strategy_key]["score"] = 0.0
                decision["raw"][strategy_key]["entry"] = None
                decision["raw"][strategy_key]["stop"] = None
                decision["raw"][strategy_key]["weight"] = 0.0
                decision["raw"][strategy_key]["timestamp"] = None
        
        # 포지션 크기 계산
        sizing = self._calculate_category_sizing(
            decision["action"], decision["raw"], category_config, account_balance, base_risk_pct
        )
        decision["sizing"].update(sizing)
        
        # 기타 필드 업데이트
        decision["max_holding_minutes"] = category_config["max_holding_minutes"]
        decision["leverage"] = category_config["leverage"]
        decision["strategies_used"] = list(category_signals.keys())
        
        # 시너지 메타데이터
        decision["meta"]["synergy_meta"].update({
            "confidence": synergy_result['confidence'],
            "market_context": synergy_result['market_context'],
            "conflicts_detected": synergy_result['conflicts_detected'],
            "bonuses_applied": synergy_result.get('bonuses_applied', []),
            "buy_score": synergy_result.get('buy_score', 0),
            "sell_score": synergy_result.get('sell_score', 0),
            "signals_used": synergy_result.get('signals_used', 0),
            "institutional_bias": synergy_result.get('meta', {}).get('institutional_bias', 'NEUTRAL'),
            "macro_trend_strength": synergy_result.get('meta', {}).get('macro_trend_strength', 'WEAK')
        })
        
        return decision

    # 나머지 필수 메서드들...
    def _calculate_category_sizing(self, action: str, raw: Dict, category_config: Dict[str, Any], 
                                 account_balance: float, base_risk_pct: float) -> Dict[str, Any]:
        """카테고리별 포지션 크기 계산"""
        sizing = {
            "qty": None,
            "risk_usd": 0.0,
            "entry_used": None,
            "stop_used": None,
            "leverage": category_config["leverage"],
            "risk_multiplier": category_config["risk_multiplier"]
        }
        
        if action == "HOLD":
            return sizing
        
        # 리스크 계산
        adjusted_risk_pct = base_risk_pct * category_config["risk_multiplier"]
        sizing["risk_usd"] = round(account_balance * adjusted_risk_pct, 4)
        
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
        
        sizing["entry_used"] = float(entry_used) if entry_used is not None else None
        sizing["stop_used"] = float(stop_used) if stop_used is not None else None
        
        # 수량 계산
        if entry_used is not None and stop_used is not None:
            qty = self._calculate_quantity(
                entry_used, stop_used, action, account_balance, 
                adjusted_risk_pct, category_config["leverage"], 1.0
            )
            sizing["qty"] = float(qty) if qty is not None else None
        
        return sizing

    def _find_any_price(self, raw: Dict) -> Optional[float]:
        """어떤 가격이라도 찾기"""
        for strategy_name, strategy_data in raw.items():
            if strategy_data.get("entry") is not None:
                return float(strategy_data.get("entry"))
        for strategy_name, strategy_data in raw.items():
            if strategy_data.get("score", 0) > 0:
                any_price = strategy_data.get("entry") or strategy_data.get("stop")
                if any_price is not None:
                    return float(any_price)
        return None

    def _calculate_stop_with_atr(self, entry_used: float, atr_val: float, action: str) -> Optional[float]:
        """ATR을 사용한 스탑 계산"""
        if atr_val is None or math.isnan(atr_val):
            atr_val = max(1.0, 0.5 * abs(entry_used) * 0.001)
            
        if action == "LONG":
            return entry_used - 1.5 * atr_val
        elif action == "SHORT":
            return entry_used + 1.5 * atr_val
        else:
            return None

    def _calculate_quantity(self, entry_used: float, stop_used: float, action: str, 
                          account_balance: float, adjusted_risk_pct: float, 
                          leverage: int, scale: float) -> Optional[float]:
        """수량 계산"""
        if entry_used == stop_used:
            return None
            
        distance = abs(entry_used - stop_used)
        if distance <= 0:
            return None
            
        risk_usd = account_balance * adjusted_risk_pct
        qty = risk_usd / distance
        qty = qty * scale * leverage
        
        return qty

    def _check_position_conflicts(self, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """포지션 충돌 체크"""
        long_categories = [k for k, v in decisions.items() if v["action"] == "LONG"]
        short_categories = [k for k, v in decisions.items() if v["action"] == "SHORT"]
        
        conflicts = []
        conflict_details = []
        
        # 단기와 장기가 반대 방향일 때 경고
        if "short_term" in long_categories and "long_term" in short_categories:
            conflicts.append("SHORT_TERM_LONG_vs_LONG_TERM_SHORT")
            conflict_details.append({
                "type": "directional_conflict",
                "categories": ["short_term", "long_term"],
                "directions": ["LONG", "SHORT"]
            })
        elif "short_term" in short_categories and "long_term" in long_categories:
            conflicts.append("SHORT_TERM_SHORT_vs_LONG_TERM_LONG")
            conflict_details.append({
                "type": "directional_conflict", 
                "categories": ["short_term", "long_term"],
                "directions": ["SHORT", "LONG"]
            })
        
        # 중기와 장기가 반대 방향일 때 경고
        if "medium_term" in long_categories and "long_term" in short_categories:
            conflicts.append("MEDIUM_TERM_LONG_vs_LONG_TERM_SHORT")
            conflict_details.append({
                "type": "directional_conflict",
                "categories": ["medium_term", "long_term"], 
                "directions": ["LONG", "SHORT"]
            })
        elif "medium_term" in short_categories and "long_term" in long_categories:
            conflicts.append("MEDIUM_TERM_SHORT_vs_LONG_TERM_LONG")
            conflict_details.append({
                "type": "directional_conflict",
                "categories": ["medium_term", "long_term"],
                "directions": ["SHORT", "LONG"]
            })
        
        return {
            "has_conflicts": len(conflicts) > 0,
            "conflict_types": conflicts,
            "long_categories": long_categories,
            "short_categories": short_categories,
            "conflict_details": conflict_details
        }