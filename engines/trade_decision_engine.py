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
    
    def __init__(self):
        self.time_manager = get_time_manager()
        # 각 카테고리별 시너지 엔진 초기화
        self.short_term_engine = ShortTermSynergyEngine(SynergyConfig())
        self.medium_term_engine = MediumTermSynergyEngine(MediumTermConfig())
        self.long_term_engine = LongTermSynergyEngine(LongTermConfig())


    def _normalize_single_decision(self, decision: Dict[str, Any], category_name: str) -> Dict[str, Any]:
        """단일 결정을 정규화"""
        # 기본 스키마로 시작
        normalized = {
            "action": decision.get("action", "HOLD"),
            "net_score": decision.get("net_score", 0.0),
            "category": category_name,
            "max_holding_minutes": self.STRATEGY_CATEGORIES[category_name]["max_holding_minutes"],
            "leverage": self.STRATEGY_CATEGORIES[category_name]["leverage"],
            "strategies_used": decision.get("strategies_used", []),
            "raw": decision.get("raw", {}),
            "sizing": decision.get("sizing", {}),
            "meta": {
                "timestamp_utc": self.time_manager.get_current_time().isoformat(),
                "timeframe": self.STRATEGY_CATEGORIES[category_name]["timeframe"],
                "used_weight_sum": 0.0,
                "synergy_meta": decision.get("meta", {}).get("synergy_meta", {})
            }
        }
        
        # 추가 필드 업데이트
        if "sizing" in decision:
            normalized["sizing"].update(decision["sizing"])
        
        if "meta" in decision and "synergy_meta" in decision["meta"]:
            normalized["meta"]["synergy_meta"].update(decision["meta"]["synergy_meta"])
        
        return normalized


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
        
        # 강화학습용 결정 데이터로 변환
        rl_decisions = self._convert_to_rl_decisions(decisions)
        
        return {
            "decisions": rl_decisions,
            "conflicts": conflicts,
        }
    
    def _convert_to_rl_decisions(self, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """결정 데이터를 강화학습용 수치 정보로 변환"""
        rl_decisions = {}
        
        for category, decision in decisions.items():
            # 액션을 수치로 변환
            action_value = {"LONG": 1.0, "SHORT": -1.0, "HOLD": 0.0}.get(decision.get("action", "HOLD"), 0.0)
            
            # 신뢰도를 수치로 변환
            confidence = decision.get("meta", {}).get("synergy_meta", {}).get("confidence", "LOW")
            confidence_value = {"HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2}.get(confidence, 0.2)
            
            # 시장 상황을 수치로 변환
            market_context = decision.get("meta", {}).get("synergy_meta", {}).get("market_context", "NEUTRAL")
            context_value = self._convert_market_context_to_value(market_context)
            
            # 카테고리별 특화 메타데이터 수치화
            synergy_meta = decision.get("meta", {}).get("synergy_meta", {})
            category_specific_values = self._extract_category_specific_values(category, synergy_meta)
            
            rl_decisions[category] = {
                # 기본 수치 정보
                "action_value": action_value,
                "net_score": round(decision.get("net_score", 0.0), 3),
                "confidence_value": confidence_value,
                "market_context_value": context_value,
                
                # 시너지 메타데이터 수치
                "buy_score": round(synergy_meta.get("buy_score", 0.0), 3),
                "sell_score": round(synergy_meta.get("sell_score", 0.0), 3),
                "signals_used": synergy_meta.get("signals_used", 0),
                "conflicts_detected_count": len(synergy_meta.get("conflicts_detected", [])),
                "bonuses_applied_count": len(synergy_meta.get("bonuses_applied", [])),
                
                # 카테고리별 특화 수치
                **category_specific_values,
                
                # 포지션 크기 정보
                "leverage": decision.get("leverage", 0),
                "risk_multiplier": decision.get("sizing", {}).get("risk_multiplier", 0.0),
                "risk_usd": round(decision.get("sizing", {}).get("risk_usd", 0.0), 2),
                
                # 전략 사용 정보
                "strategies_count": len(decision.get("strategies_used", [])),
                "max_holding_minutes": decision.get("max_holding_minutes", 0)
            }
        
        return rl_decisions
    
    def _convert_market_context_to_value(self, context: str) -> float:
        """시장 상황을 수치로 변환"""
        context_mapping = {
            "TRENDING": 0.8,
            "RANGING": 0.3,
            "BREAKOUT": 0.9,
            "CONSOLIDATION": 0.4,
            "STRONG_TREND": 0.9,
            "REVERSAL_ZONE": 0.6,
            "INSTITUTIONAL_ACCUMULATION": 0.8,
            "DISTRIBUTION_PHASE": 0.2,
            "MACRO_TREND": 0.7,
            "RANGE_BOUND": 0.3,
            "VOLATILITY_EXPANSION": 0.6,
            "NEUTRAL": 0.5
        }
        return context_mapping.get(context, 0.5)
    
    def _extract_category_specific_values(self, category: str, synergy_meta: Dict[str, Any]) -> Dict[str, float]:
        """카테고리별 특화 메타데이터를 수치로 변환"""
        if category == "short_term":
            return {
                "momentum_strength": self._convert_strength_to_value(synergy_meta.get("momentum_strength", "WEAK")),
                "reversion_potential": self._convert_potential_to_value(synergy_meta.get("reversion_potential", "LOW"))
            }
        elif category == "medium_term":
            return {
                "trend_strength": self._convert_strength_to_value(synergy_meta.get("trend_strength", "WEAK")),
                "consolidation_level": self._convert_level_to_value(synergy_meta.get("consolidation_level", "NEUTRAL"))
            }
        elif category == "long_term":
            return {
                "institutional_bias": self._convert_bias_to_value(synergy_meta.get("institutional_bias", "NEUTRAL")),
                "macro_trend_strength": self._convert_strength_to_value(synergy_meta.get("macro_trend_strength", "WEAK"))
            }
        else:
            return {}
    
    def _convert_strength_to_value(self, strength: str) -> float:
        """강도 문자열을 수치로 변환"""
        strength_mapping = {"STRONG": 0.8, "MEDIUM": 0.5, "WEAK": 0.2}
        return strength_mapping.get(strength, 0.2)
    
    def _convert_potential_to_value(self, potential: str) -> float:
        """잠재력 문자열을 수치로 변환"""
        potential_mapping = {"HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2}
        return potential_mapping.get(potential, 0.2)
    
    def _convert_level_to_value(self, level: str) -> float:
        """수준 문자열을 수치로 변환"""
        level_mapping = {"HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2, "NEUTRAL": 0.5}
        return level_mapping.get(level, 0.5)
    
    def _convert_bias_to_value(self, bias: str) -> float:
        """편향 문자열을 수치로 변환"""
        bias_mapping = {"BULLISH": 0.8, "WEAK_BULLISH": 0.6, "NEUTRAL": 0.5, "WEAK_BEARISH": 0.4, "BEARISH": 0.2}
        return bias_mapping.get(bias, 0.5)
    
    def _create_category_hold_decision(self, category_name: str) -> Dict[str, Any]:
        """카테고리별 HOLD 결정 생성"""
        return {
            "action": "HOLD",
            "net_score": 0.0,
            "category": category_name,
            "max_holding_minutes": self.STRATEGY_CATEGORIES[category_name]["max_holding_minutes"],
            "leverage": self.STRATEGY_CATEGORIES[category_name]["leverage"],
            "strategies_used": [],
            "raw": {},
            "sizing": {},
            "meta": {
                "timestamp_utc": self.time_manager.get_current_time().isoformat(),
                "timeframe": self.STRATEGY_CATEGORIES[category_name]["timeframe"],
                "used_weight_sum": 0.0,
                "synergy_meta": {}
            }
        }

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
        
        # 시너지 엔진 실행
        synergy_result = self.short_term_engine.calculate_synergy_score(category_signals)
        
        # 시너지 결과를 기반으로 결정 생성
        decision = {
            "action": 'LONG' if synergy_result['action'] == 'BUY' else 'SHORT' if synergy_result['action'] == 'SELL' else 'HOLD',
            "net_score": round(synergy_result['net_score'], 4),
            "category": "SHORT_TERM",
            "max_holding_minutes": category_config["max_holding_minutes"],
            "leverage": category_config["leverage"],
            "strategies_used": list(category_signals.keys()),
            "raw": {},
            "sizing": {},
            "meta": {
                "timestamp_utc": self.time_manager.get_current_time().isoformat(),
                "timeframe": category_config["timeframe"],
                "used_weight_sum": 0.0,
                "synergy_meta": {}
            }
        }
        
        # raw 신호 업데이트
        for name, signal_data in category_signals.items():
            normalized_name = name.lower().replace("_or", "")
            decision["raw"][normalized_name] = {
                "action": signal_data.get("action", "HOLD"),
                "score": signal_data.get("score", 0.0),
                "entry": signal_data.get("entry"),
                "stop": signal_data.get("stop"),
                "weight": 1.0,
                "timestamp": self.time_manager.get_current_time()
            }
        
        
        # 포지션 크기 계산
        sizing = self._calculate_category_sizing(
            decision["action"], decision["raw"], category_config, account_balance, base_risk_pct
        )
        decision["sizing"].update(sizing)
        
        # 기타 필드 업데이트
        decision["max_holding_minutes"] = category_config["max_holding_minutes"]
        decision["leverage"] = category_config["leverage"]
        decision["strategies_used"] = list(category_signals.keys())
        
        # 시너지 메타데이터 업데이트
        decision["meta"]["synergy_meta"].update({
            "confidence": synergy_result['confidence'],
            "market_context": synergy_result['market_context'],
            "conflicts_detected": synergy_result['conflicts_detected'],
            "bonuses_applied": synergy_result.get('bonuses_applied', []),
            "buy_score": synergy_result.get('buy_score', 0),
            "sell_score": synergy_result.get('sell_score', 0),
            "signals_used": synergy_result.get('signals_used', 0),
            "momentum_strength": synergy_result.get('meta', {}).get('momentum_strength', 'WEAK'),
            "reversion_potential": synergy_result.get('meta', {}).get('reversion_potential', 'LOW')
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
        
        # 시너지 엔진 실행
        synergy_result = self.medium_term_engine.calculate_synergy_score(category_signals)
        
        # 시너지 결과를 기반으로 결정 생성
        decision = {
            "action": 'LONG' if synergy_result['action'] == 'BUY' else 'SHORT' if synergy_result['action'] == 'SELL' else 'HOLD',
            "net_score": round(synergy_result['net_score'], 4),
            "category": "MEDIUM_TERM",
            "max_holding_minutes": category_config["max_holding_minutes"],
            "leverage": category_config["leverage"],
            "strategies_used": list(category_signals.keys()),
            "raw": {},
            "sizing": {},
            "meta": {
                "timestamp_utc": self.time_manager.get_current_time().isoformat(),
                "timeframe": category_config["timeframe"],
                "used_weight_sum": 0.0,
                "synergy_meta": {}
            }
        }
        
        # raw 신호 업데이트
        for name, signal_data in category_signals.items():
            normalized_name = name.lower()
            decision["raw"][normalized_name] = {
                "action": signal_data.get("action", "HOLD"),
                "score": signal_data.get("score", 0.0),
                "entry": signal_data.get("entry"),
                "stop": signal_data.get("stop"),
                "weight": 1.0,
                "timestamp": self.time_manager.get_current_time()
            }
        
        
        # 포지션 크기 계산
        sizing = self._calculate_category_sizing(
            decision["action"], decision["raw"], category_config, account_balance, base_risk_pct
        )
        decision["sizing"].update(sizing)
        
        # 기타 필드 업데이트
        decision["max_holding_minutes"] = category_config["max_holding_minutes"]
        decision["leverage"] = category_config["leverage"]
        decision["strategies_used"] = list(category_signals.keys())
        
        # 시너지 메타데이터 업데이트
        decision["meta"]["synergy_meta"].update({
            "confidence": synergy_result['confidence'],
            "market_context": synergy_result['market_context'],
            "conflicts_detected": synergy_result['conflicts_detected'],
            "bonuses_applied": synergy_result.get('bonuses_applied', []),
            "buy_score": synergy_result.get('buy_score', 0),
            "sell_score": synergy_result.get('sell_score', 0),
            "signals_used": synergy_result.get('signals_used', 0),
            "trend_strength": synergy_result.get('meta', {}).get('trend_strength', 'WEAK'),
            "consolidation_level": synergy_result.get('meta', {}).get('consolidation_level', 'NEUTRAL')
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
        
        # 시너지 엔진 실행
        synergy_result = self.long_term_engine.calculate_synergy_score(category_signals)
        
        # 시너지 결과를 기반으로 결정 생성
        decision = {
            "action": 'LONG' if synergy_result['action'] == 'BUY' else 'SHORT' if synergy_result['action'] == 'SELL' else 'HOLD',
            "net_score": round(synergy_result['net_score'], 4),
            "category": "LONG_TERM",
            "max_holding_minutes": category_config["max_holding_minutes"],
            "leverage": category_config["leverage"],
            "strategies_used": list(category_signals.keys()),
            "raw": {},
            "sizing": {},
            "meta": {
                "timestamp_utc": self.time_manager.get_current_time().isoformat(),
                "timeframe": category_config["timeframe"],
                "used_weight_sum": 0.0,
                "synergy_meta": {}
            }
        }
        
        # raw 신호 업데이트
        for name, signal_data in category_signals.items():
            normalized_name = name.lower()
            decision["raw"][normalized_name] = {
                "action": signal_data.get("action", "HOLD"),
                "score": signal_data.get("score", 0.0),
                "entry": signal_data.get("entry"),
                "stop": signal_data.get("stop"),
                "weight": 1.0,
                "timestamp": self.time_manager.get_current_time()
            }
        
        
        # 포지션 크기 계산
        sizing = self._calculate_category_sizing(
            decision["action"], decision["raw"], category_config, account_balance, base_risk_pct
        )
        decision["sizing"].update(sizing)
        
        # 기타 필드 업데이트
        decision["max_holding_minutes"] = category_config["max_holding_minutes"]
        decision["leverage"] = category_config["leverage"]
        decision["strategies_used"] = list(category_signals.keys())
        
        # 시너지 메타데이터 업데이트
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
        """포지션 충돌 체크 - 강화학습 데이터 최적화"""
        long_categories = [k for k, v in decisions.items() if v["action"] == "LONG"]
        short_categories = [k for k, v in decisions.items() if v["action"] == "SHORT"]
        hold_categories = [k for k, v in decisions.items() if v["action"] == "HOLD"]
        
        # 기본 충돌 정보
        conflicts = []
        conflict_details = []
        conflict_severity = 0.0
        directional_consensus = 0.0
        
        # 1. 방향성 충돌 감지 및 심각도 계산
        directional_conflicts = []
        
        # 단기 vs 장기 충돌
        if "short_term" in long_categories and "long_term" in short_categories:
            conflicts.append("SHORT_TERM_LONG_vs_LONG_TERM_SHORT")
            directional_conflicts.append({
                "type": "directional_conflict",
                "categories": ["short_term", "long_term"],
                "directions": ["LONG", "SHORT"],
                "severity": 0.8,  # 단기-장기 충돌은 높은 심각도
                "timeframe_conflict": True
            })
        elif "short_term" in short_categories and "long_term" in long_categories:
            conflicts.append("SHORT_TERM_SHORT_vs_LONG_TERM_LONG")
            directional_conflicts.append({
                "type": "directional_conflict", 
                "categories": ["short_term", "long_term"],
                "directions": ["SHORT", "LONG"],
                "severity": 0.8,
                "timeframe_conflict": True
            })
        
        # 중기 vs 장기 충돌
        if "medium_term" in long_categories and "long_term" in short_categories:
            conflicts.append("MEDIUM_TERM_LONG_vs_LONG_TERM_SHORT")
            directional_conflicts.append({
                "type": "directional_conflict",
                "categories": ["medium_term", "long_term"], 
                "directions": ["LONG", "SHORT"],
                "severity": 0.6,  # 중기-장기 충돌은 중간 심각도
                "timeframe_conflict": True
            })
        elif "medium_term" in short_categories and "long_term" in long_categories:
            conflicts.append("MEDIUM_TERM_SHORT_vs_LONG_TERM_LONG")
            directional_conflicts.append({
                "type": "directional_conflict",
                "categories": ["medium_term", "long_term"],
                "directions": ["SHORT", "LONG"],
                "severity": 0.6,
                "timeframe_conflict": True
            })
        
        # 단기 vs 중기 충돌
        if "short_term" in long_categories and "medium_term" in short_categories:
            conflicts.append("SHORT_TERM_LONG_vs_MEDIUM_TERM_SHORT")
            directional_conflicts.append({
                "type": "directional_conflict",
                "categories": ["short_term", "medium_term"],
                "directions": ["LONG", "SHORT"],
                "severity": 0.4,  # 단기-중기 충돌은 낮은 심각도
                "timeframe_conflict": False
            })
        elif "short_term" in short_categories and "medium_term" in long_categories:
            conflicts.append("SHORT_TERM_SHORT_vs_MEDIUM_TERM_LONG")
            directional_conflicts.append({
                "type": "directional_conflict",
                "categories": ["short_term", "medium_term"],
                "directions": ["SHORT", "LONG"],
                "severity": 0.4,
                "timeframe_conflict": False
            })
        
        # 2. 신뢰도 기반 충돌 분석
        confidence_conflicts = []
        for conflict in directional_conflicts:
            categories = conflict["categories"]
            conflict_scores = []
            confidence_scores = []
            
            for cat in categories:
                if cat in decisions:
                    decision = decisions[cat]
                    conflict_scores.append(decision.get("net_score", 0.0))
                    confidence = decision.get("meta", {}).get("synergy_meta", {}).get("confidence", "LOW")
                    # 신뢰도를 수치로 변환
                    confidence_value = {"HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2}.get(confidence, 0.2)
                    confidence_scores.append(confidence_value)
            
            # 신뢰도가 높은 카테고리들 간의 충돌은 더 심각
            if len(confidence_scores) >= 2:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                if avg_confidence > 0.6:  # 높은 신뢰도
                    conflict["severity"] *= 1.5
                    conflict["high_confidence_conflict"] = True
                else:
                    conflict["high_confidence_conflict"] = False
            
            confidence_conflicts.append(conflict)
        
        # 3. 전체 심각도 계산
        if directional_conflicts:
            conflict_severity = max([c["severity"] for c in directional_conflicts])
            # 여러 충돌이 있을 때 누적 효과
            if len(directional_conflicts) > 1:
                conflict_severity = min(1.0, conflict_severity * (1 + 0.2 * (len(directional_conflicts) - 1)))
        
        # 4. 방향성 컨센서스 계산
        active_categories = len(long_categories) + len(short_categories)
        if active_categories > 0:
            # 같은 방향의 카테고리 비율
            if len(long_categories) > len(short_categories):
                directional_consensus = len(long_categories) / active_categories
            elif len(short_categories) > len(long_categories):
                directional_consensus = len(short_categories) / active_categories
            else:
                directional_consensus = 0.5  # 균등 분할
        
        # 5. 리스크 지표 계산
        risk_indicators = {
            "max_leverage_used": max([decisions[cat].get("leverage", 0) for cat in long_categories + short_categories], default=0),
            "total_exposure": len(long_categories) + len(short_categories),
            "timeframe_diversity": len(set([decisions[cat].get("meta", {}).get("timeframe", "") for cat in long_categories + short_categories])),
            "hold_ratio": len(hold_categories) / len(decisions) if decisions else 0
        }
        
        # 6. 강화학습을 위한 보상/페널티 신호
        rl_signals = {
            "conflict_penalty": -conflict_severity,  # 충돌 시 음수 보상
            "consensus_bonus": directional_consensus * 0.5,  # 컨센서스 시 양수 보상
            "diversity_bonus": risk_indicators["timeframe_diversity"] * 0.1,  # 다양성 보상
            "risk_penalty": -min(1.0, risk_indicators["max_leverage_used"] / 20.0) * 0.3  # 과도한 레버리지 페널티
        }
        
        return {
            # 강화학습용 핵심 수치 정보
            "conflict_severity": round(conflict_severity, 3),
            "directional_consensus": round(directional_consensus, 3),
            "conflict_ratio": round(len(conflicts) / max(1, active_categories - 1) if active_categories > 1 else 0, 3),
            "active_categories": active_categories,
            "hold_ratio": round(risk_indicators["hold_ratio"], 3),
            
            # 리스크 지표
            "max_leverage_used": risk_indicators["max_leverage_used"],
            "total_exposure": risk_indicators["total_exposure"],
            "timeframe_diversity": risk_indicators["timeframe_diversity"],
            
            # 강화학습 보상/페널티 신호
            "conflict_penalty": round(rl_signals["conflict_penalty"], 3),
            "consensus_bonus": round(rl_signals["consensus_bonus"], 3),
            "diversity_bonus": round(rl_signals["diversity_bonus"], 3),
            "risk_penalty": round(rl_signals["risk_penalty"], 3),
            
            # 포지션 분포
            "long_count": len(long_categories),
            "short_count": len(short_categories),
            "hold_count": len(hold_categories)
        }