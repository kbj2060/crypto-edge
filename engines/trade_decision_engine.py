from typing import Dict, Any, Optional
from datetime import datetime
import math
from managers.time_manager import get_time_manager
from indicators.global_indicators import get_atr, get_all_indicators
from engines.short_term_synergy_engine import ShortTermSynergyEngine, SynergyConfig
from engines.medium_term_synergy_engine import MediumTermSynergyEngine, MediumTermConfig
from engines.long_term_synergy_engine import LongTermSynergyEngine, LongTermConfig
from engines.meta_labeling_engine import MetaLabelingEngine


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
    
    def __init__(self, use_meta_labeling: bool = True):
        self.time_manager = get_time_manager()
        # 각 카테고리별 시너지 엔진 초기화
        self.short_term_engine = ShortTermSynergyEngine(SynergyConfig())
        self.medium_term_engine = MediumTermSynergyEngine(MediumTermConfig())
        self.long_term_engine = LongTermSynergyEngine(LongTermConfig())
        
        # 메타 라벨링 엔진 초기화
        self.use_meta_labeling = use_meta_labeling
        self.meta_labeling_engine = None
        if use_meta_labeling:
            # 모델 성능이 낮으므로 더 관대한 기준 사용
            # ROC-AUC 0.549, Precision 17.4% → confidence_threshold 낮춤
            self.meta_labeling_engine = MetaLabelingEngine(
                confidence_threshold=0.5  # 50% 이상 확률일 때 거래 실행 (더 관대)
            )
            # 기존 모델이 있으면 로드
            if self.meta_labeling_engine.load_model():
                print(f"✅ 메타 라벨링 모델 로드 완료 (임계값: {self.meta_labeling_engine.confidence_threshold:.0%})")
            else:
                print("⚠️ 메타 라벨링 모델 로드 실패 - 기본 휴리스틱 사용")


    def _normalize_single_decision(self, decision: Dict[str, Any], category_name: str) -> Dict[str, Any]:
        """단일 결정을 정규화"""
        # 기본 스키마로 시작
        normalized = {
            "action": decision.get("action", "HOLD"),
            "net_score": decision.get("net_score", 0.0),
            "reason": decision.get("reason", ""),  # reason 필드 추가
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
        """
        실시간 거래 결정 - Meta-Guided Consensus 아키텍처
        
        1단계: 각 카테고리별 결정 생성
        2단계: 카테고리 종합 및 충돌 분석
        3단계: 메타 라벨링으로 최종 실행 여부 결정
        4단계: 단일 최종 결정 반환
        """
        
        # 1단계: 각 카테고리별로 독립적으로 결정
        category_decisions = {}
        for category_name, category_config in self.STRATEGY_CATEGORIES.items():
            category_decisions[category_name.lower()] = self._decide_category_trade(
                signals, category_name, category_config,
                account_balance, base_risk_pct, open_threshold,
                immediate_threshold, confirm_threshold, confirm_window_sec,
                session_priority, news_event
            )
        
        # 2단계: 포지션 충돌 체크
        conflicts = self._check_position_conflicts(category_decisions)
        
        # 3단계: Meta-Guided Consensus - 모든 카테고리를 종합하여 최종 결정 생성
        final_decision = self._build_consensus_decision(
            category_decisions, conflicts, account_balance, base_risk_pct
        )
        
        # 4단계: 메타 라벨링으로 최종 실행 여부 검증
        if self.use_meta_labeling and self.meta_labeling_engine:
            final_decision = self._apply_meta_guided_consensus(
                final_decision, category_decisions, conflicts
            )
        
        return {
            "final_decision": final_decision,
            "category_decisions": self._normalize_decisions_schema(category_decisions),
            "conflicts": conflicts,
        }
    
    def _create_category_hold_decision(self, category_name: str) -> Dict[str, Any]:
        """카테고리별 HOLD 결정 생성"""
        return {
            "action": "HOLD",
            "net_score": 0.0,
            "reason": "신호 없음",
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
        
        # 이유 생성
        reason_parts = []
        if synergy_result.get('confidence'):
            reason_parts.append(f"신뢰도: {synergy_result['confidence']}")
        if synergy_result.get('market_context'):
            reason_parts.append(f"시장상황: {synergy_result['market_context']}")
        if synergy_result.get('signals_used', 0) > 0:
            reason_parts.append(f"신호 {synergy_result['signals_used']}개 사용")
        reason = ", ".join(reason_parts) if reason_parts else "단기 전략 분석"
        
        # 시너지 결과를 기반으로 결정 생성
        decision = {
            "action": 'LONG' if synergy_result['action'] == 'BUY' else 'SHORT' if synergy_result['action'] == 'SELL' else 'HOLD',
            "net_score": round(synergy_result['net_score'], 4),
            "reason": reason,
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
        
        # 이유 생성
        reason_parts = []
        if synergy_result.get('confidence'):
            reason_parts.append(f"신뢰도: {synergy_result['confidence']}")
        if synergy_result.get('market_context'):
            reason_parts.append(f"시장상황: {synergy_result['market_context']}")
        if synergy_result.get('signals_used', 0) > 0:
            reason_parts.append(f"신호 {synergy_result['signals_used']}개 사용")
        reason = ", ".join(reason_parts) if reason_parts else "중기 전략 분석"
        
        # 시너지 결과를 기반으로 결정 생성
        decision = {
            "action": 'LONG' if synergy_result['action'] == 'BUY' else 'SHORT' if synergy_result['action'] == 'SELL' else 'HOLD',
            "net_score": round(synergy_result['net_score'], 4),
            "reason": reason,
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
        
        # 이유 생성
        reason_parts = []
        if synergy_result.get('confidence'):
            reason_parts.append(f"신뢰도: {synergy_result['confidence']}")
        if synergy_result.get('market_context'):
            reason_parts.append(f"시장상황: {synergy_result['market_context']}")
        if synergy_result.get('signals_used', 0) > 0:
            reason_parts.append(f"신호 {synergy_result['signals_used']}개 사용")
        reason = ", ".join(reason_parts) if reason_parts else "장기 전략 분석"
        
        # 시너지 결과를 기반으로 결정 생성
        decision = {
            "action": 'LONG' if synergy_result['action'] == 'BUY' else 'SHORT' if synergy_result['action'] == 'SELL' else 'HOLD',
            "net_score": round(synergy_result['net_score'], 4),
            "reason": reason,
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
        """포지션 충돌 체크"""
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
        
        return {
            "has_conflicts": len(conflicts) > 0,
            "conflict_severity": round(conflict_severity, 3),
            "directional_consensus": round(directional_consensus, 3),
            "conflict_ratio": round(len(conflicts) / max(1, active_categories - 1) if active_categories > 1 else 0, 3),
            "active_categories": active_categories,
            "hold_ratio": round(risk_indicators["hold_ratio"], 3),
            "conflict_types": conflicts,
            "long_categories": long_categories,
            "short_categories": short_categories,
            
            # 리스크 지표
            "max_leverage_used": risk_indicators["max_leverage_used"],
            "total_exposure": risk_indicators["total_exposure"],
            "timeframe_diversity": risk_indicators["timeframe_diversity"],
            
            # 포지션 분포
            "long_count": len(long_categories),
            "short_count": len(short_categories),
            "hold_count": len(hold_categories)
        }
    
    def _build_consensus_decision(
        self,
        category_decisions: Dict[str, Dict[str, Any]],
        conflicts: Dict[str, Any],
        account_balance: float,
        base_risk_pct: float
    ) -> Dict[str, Any]:
        """
        모든 카테고리를 종합하여 최종 결정 생성 (Meta-Guided Consensus)
        
        Args:
            category_decisions: 카테고리별 결정 딕셔너리
            conflicts: 충돌 정보
            account_balance: 계좌 잔액
            base_risk_pct: 기본 리스크 비율
            
        Returns:
            최종 결정 딕셔너리
        """
        # 카테고리별 점수 수집
        category_scores = {}
        category_actions = {}
        category_confidences = {}
        total_weight = 0.0
        weighted_score = 0.0
        
        for category_name, category_config in self.STRATEGY_CATEGORIES.items():
            cat_key = category_name.lower()
            decision = category_decisions.get(cat_key, {})
            
            action = decision.get("action", "HOLD")
            net_score = decision.get("net_score", 0.0)
            confidence = decision.get("meta", {}).get("synergy_meta", {}).get("confidence", "LOW")
            weight = category_config.get("weight", 0.0)
            
            category_actions[cat_key] = action
            category_scores[cat_key] = net_score
            category_confidences[cat_key] = confidence
            
            # HOLD가 아닌 경우에만 가중치 적용
            if action != "HOLD":
                # 신뢰도에 따른 가중치 조정
                confidence_multiplier = {"HIGH": 1.2, "MEDIUM": 1.0, "LOW": 0.8}.get(confidence, 1.0)
                adjusted_weight = weight * confidence_multiplier
                weighted_score += net_score * adjusted_weight
                total_weight += adjusted_weight
        
        # 최종 점수 계산
        if total_weight > 0:
            consensus_score = weighted_score / total_weight
        else:
            consensus_score = 0.0
        
        # 방향성 컨센서스 계산
        long_count = sum(1 for a in category_actions.values() if a == "LONG")
        short_count = sum(1 for a in category_actions.values() if a == "SHORT")
        hold_count = sum(1 for a in category_actions.values() if a == "HOLD")
        
        # 최종 action 결정
        if abs(consensus_score) < 0.1 or (long_count == 0 and short_count == 0):
            final_action = "HOLD"
        elif long_count > short_count:
            final_action = "LONG"
        elif short_count > long_count:
            final_action = "SHORT"
        else:
            # 동점인 경우 점수로 결정
            final_action = "LONG" if consensus_score > 0 else "SHORT" if consensus_score < 0 else "HOLD"
        
        # 최종 신뢰도 계산
        active_categories = long_count + short_count
        if active_categories == 0:
            final_confidence = "LOW"
        elif active_categories >= 2 and conflicts.get("conflict_severity", 0.0) < 0.3:
            # 여러 카테고리가 일치하고 충돌이 적으면 높은 신뢰도
            final_confidence = "HIGH"
        elif active_categories >= 2:
            final_confidence = "MEDIUM"
        else:
            final_confidence = "LOW"
        
        # 진입가/손절가 결정 (가장 신뢰도 높은 카테고리 사용)
        entry_price = None
        stop_price = None
        leverage = 1
        
        for cat_key in ["short_term", "medium_term", "long_term"]:
            decision = category_decisions.get(cat_key, {})
            if decision.get("action") == final_action:
                sizing = decision.get("sizing", {})
                if sizing.get("entry_used") and sizing.get("stop_used"):
                    entry_price = sizing.get("entry_used")
                    stop_price = sizing.get("stop_used")
                    leverage = decision.get("leverage", 1)
                    break
        
        # 포지션 크기 계산
        sizing = {}
        if final_action != "HOLD" and entry_price and stop_price:
            # 가장 보수적인 리스크 사용
            risk_multiplier = min([
                self.STRATEGY_CATEGORIES[cat.upper()].get("risk_multiplier", 1.0)
                for cat, dec in category_decisions.items()
                if dec.get("action") == final_action
            ] or [1.0])
            
            adjusted_risk_pct = base_risk_pct * risk_multiplier
            risk_usd = account_balance * adjusted_risk_pct
            
            distance = abs(entry_price - stop_price)
            qty = None
            if distance > 0:
                qty = risk_usd / distance * leverage
            
            sizing = {
                "qty": qty,
                "risk_usd": risk_usd,
                "entry_used": entry_price,
                "stop_used": stop_price,
                "leverage": leverage,
                "risk_multiplier": risk_multiplier
            }
        
        # 이유 생성
        reason_parts = []
        if long_count > 0:
            reason_parts.append(f"LONG {long_count}개 카테고리")
        if short_count > 0:
            reason_parts.append(f"SHORT {short_count}개 카테고리")
        if conflicts.get("conflict_severity", 0.0) > 0.3:
            reason_parts.append(f"충돌 심각도: {conflicts.get('conflict_severity', 0):.2f}")
        reason = ", ".join(reason_parts) if reason_parts else "신호 없음"
        
        return {
            "action": final_action,
            "net_score": round(consensus_score, 4),
            "reason": reason,
            "confidence": final_confidence,
            "consensus_meta": {
                "long_categories": long_count,
                "short_categories": short_count,
                "hold_categories": hold_count,
                "total_weight": round(total_weight, 4),
                "category_scores": {k: round(v, 4) for k, v in category_scores.items()},
                "category_actions": category_actions,
                "category_confidences": category_confidences
            },
            "sizing": sizing,
            "meta": {
                "timestamp_utc": self.time_manager.get_current_time().isoformat(),
                "timeframe": "MULTI",
                "architecture": "meta_guided_consensus"
            }
        }
    
    def _apply_meta_guided_consensus(
        self,
        final_decision: Dict[str, Any],
        category_decisions: Dict[str, Dict[str, Any]],
        conflicts: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Meta-Guided Consensus: 메타 라벨링으로 최종 결정 검증
        
        모든 카테고리 정보와 충돌 정보를 종합하여
        최종 실행 여부를 결정합니다.
        """
        if not self.meta_labeling_engine:
            return final_decision
        
        # HOLD는 메타 라벨링 불필요
        if final_decision.get("action") == "HOLD":
            return final_decision
        
        # 시장 데이터 수집
        indicators = get_all_indicators()
        market_data = {
            "atr": indicators.get("atr", 0.0),
            "volume": 0.0,  # TODO: 실제 볼륨 데이터 추가
            "volatility": indicators.get("vwap_std", 0.0) if indicators.get("vwap") else 0.0
        }
        
        # 메타 라벨링을 위한 종합 결정 생성
        # 시그널 특성만 사용 (충돌/시너지 특성 제외)
        meta_decision = final_decision.copy()
        meta_decision["meta"] = meta_decision.get("meta", {})
        meta_decision["meta"]["synergy_meta"] = {
            "confidence": final_decision.get("confidence", "LOW"),
            # 충돌 관련 특성 제거: conflict_severity, directional_consensus, active_categories
            "buy_score": final_decision.get("net_score", 0.0) if final_decision.get("action") == "LONG" else 0.0,
            "sell_score": abs(final_decision.get("net_score", 0.0)) if final_decision.get("action") == "SHORT" else 0.0,
            "signals_used": sum(len(dec.get("strategies_used", [])) for dec in category_decisions.values())
        }
        
        # strategies_used 추가 (extract_features에서 사용)
        meta_decision["strategies_used"] = []
        for dec in category_decisions.values():
            meta_decision["strategies_used"].extend(dec.get("strategies_used", []))
        
        # 메타 라벨링 예측
        meta_result = self.meta_labeling_engine.predict(meta_decision, market_data)
        
        # 메타 라벨링 결과를 결정에 추가
        final_decision["meta"]["meta_labeling"] = {
            "should_execute": meta_result.get("should_execute", True),
            "prediction": meta_result.get("prediction", 1),
            "probability": meta_result.get("probability", 0.5),
            "confidence": meta_result.get("confidence", "MEDIUM")
        }
        
        # 메타 라벨링이 거래 실행을 권장하지 않으면 HOLD로 변경
        if not meta_result.get("should_execute", True):
            original_action = final_decision.get("action")
            probability = meta_result.get("probability", 0.0)
            
            # HOLD의 의미: "새로운 포지션을 열지 말라" (기존 포지션 유지)
            # 주의: 이미 열려있는 포지션은 별도로 관리해야 함
            final_decision["action"] = "HOLD"
            final_decision["reason"] = f"메타 라벨링: {original_action} → HOLD (확률: {probability:.1%}, 임계값: {self.meta_labeling_engine.confidence_threshold:.1%} 미달) [새 포지션 미개설]"
            final_decision["net_score"] = 0.0
            
            # 원본 정보 저장 (기존 포지션 관리에 사용 가능)
            final_decision["meta"]["_original_action"] = original_action
            final_decision["meta"]["_original_score"] = final_decision.get("net_score", 0.0)
            final_decision["meta"]["_meta_blocked"] = True  # 메타 라벨링에 의해 차단됨 표시
            
            # 포지션 크기 초기화 (새 포지션을 열지 않음)
            final_decision["sizing"] = {
                "qty": None,
                "risk_usd": 0.0,
                "entry_used": None,
                "stop_used": None,
                "leverage": 1,
                "risk_multiplier": 1.0
            }
        else:
            # 거래 실행 권장 시 이유 업데이트
            probability = meta_result.get("probability", 0.0)
            original_reason = final_decision.get("reason", "")
            final_decision["reason"] = f"{original_reason} [메타 라벨링: {probability:.1%}]"
        
        return final_decision
    
    def _apply_meta_labeling(self, decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        메타 라벨링을 적용하여 거래 실행 여부 결정
        
        Args:
            decisions: 원본 결정 딕셔너리
            
        Returns:
            메타 라벨링이 적용된 결정 딕셔너리
        """
        if not self.meta_labeling_engine:
            return decisions
        
        # 시장 데이터 수집
        indicators = get_all_indicators()
        market_data = {
            "atr": indicators.get("atr", 0.0),
            "volume": 0.0,  # TODO: 실제 볼륨 데이터 추가
            "volatility": indicators.get("vwap_std", 0.0) if indicators.get("vwap") else 0.0
        }
        
        # 각 결정에 메타 라벨링 적용
        for category_name, decision in decisions.items():
            # HOLD는 메타 라벨링 불필요
            if decision.get("action") == "HOLD":
                continue
            
            # 메타 라벨링 예측
            meta_result = self.meta_labeling_engine.predict(decision, market_data)
            
            # 메타 라벨링 결과를 결정에 추가
            if "meta" not in decision:
                decision["meta"] = {}
            
            if "meta_labeling" not in decision["meta"]:
                decision["meta"]["meta_labeling"] = {}
            
            decision["meta"]["meta_labeling"] = {
                "should_execute": meta_result.get("should_execute", True),
                "prediction": meta_result.get("prediction", 1),
                "probability": meta_result.get("probability", 0.5),
                "confidence": meta_result.get("confidence", "MEDIUM")
            }
            
            # 메타 라벨링이 거래 실행을 권장하지 않으면 HOLD로 변경
            if not meta_result.get("should_execute", True):
                original_action = decision.get("action")
                original_score = decision.get("net_score", 0.0)
                probability = meta_result.get("probability", 0.0)
                
                decision["action"] = "HOLD"
                decision["reason"] = f"메타 라벨링: {original_action} → HOLD (확률: {probability:.1%}, 임계값: {self.meta_labeling_engine.confidence_threshold:.1%} 미달)"
                decision["net_score"] = 0.0
                
                # 원본 정보 저장 (디버깅용)
                if "meta" not in decision:
                    decision["meta"] = {}
                decision["meta"]["_original_action"] = original_action
                decision["meta"]["_original_score"] = original_score
                
                # 포지션 크기 초기화
                decision["sizing"] = {
                    "qty": None,
                    "risk_usd": 0.0,
                    "entry_used": None,
                    "stop_used": None,
                    "leverage": decision.get("leverage", 1),
                    "risk_multiplier": 1.0
                }
            else:
                # 거래 실행 권장 시 이유 업데이트
                probability = meta_result.get("probability", 0.0)
                original_reason = decision.get("reason", "")
                decision["reason"] = f"{original_reason} [메타 라벨링: {probability:.1%}]"
        
        return decisions