import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class TimingConfig:
    """타이밍 전략 설정"""
    # 진입 설정
    entry_confidence_min: float = 0.25  # 최소 진입 신뢰도
    entry_rr_min: float = 0.2  # 최소 리스크/보상 비율
    entry_score_threshold: float = 0.4  # 추가
    
    # 청산 설정
    take_profit_levels: List[float] = None  # 익절 레벨 (기본값: [1.5, 2.5])
    stop_loss_atr_mult: float = 1.0  # 손절 ATR 배수
    
    # 시간 기반 청산
    max_hold_time_hours: int = 24  # 최대 보유 시간 (시간)
    partial_exit_hours: List[int] = field(default_factory=lambda: [4, 8, 12])
    
    # 동적 청산
    trailing_stop: bool = True  # 트레일링 스탑 사용
    trailing_stop_atr: float = 2.0  # 트레일링 스탑 ATR 배수
    
    # 리스크 관리
    max_position_size: float = 0.1  # 최대 포지션 크기 (계좌의 10%)
    max_daily_loss: float = 0.05  # 최대 일일 손실 (계좌의 5%)
    atr_multiplier: float = 2.0
    
    def __post_init__(self):
        if self.take_profit_levels is None:
            self.take_profit_levels = [1.5, 2.5]
        if self.partial_exit_hours is None:
            self.partial_exit_hours = [4, 12]

class TimingStrategy:
    """타이밍 전략 클래스"""
    
    def __init__(self, config: TimingConfig):
        self.config = config
        self.active_positions = {}  # 활성 포지션 추적
        self.daily_pnl = 0.0  # 일일 손익
        self.last_reset_date = datetime.now().date()
    
    def analyze_entry_timing(self, plan: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """진입 타이밍 분석"""
        if not plan or plan.get('bias') == 'NEUTRAL':
            return {"action": "WAIT", "reason": "명확한 신호 없음"}
        
        bias = plan.get('bias')
        confidence = plan.get('confidence', 0)
        rr_ratio = plan.get('risk_reward_ratio', 0)
        
        # 기본 조건 체크
        if confidence < self.config.entry_confidence_min:
            return {"action": "WAIT", "reason": f"신뢰도 부족 ({confidence:.2f} < {self.config.entry_confidence_min})"}
        
        if rr_ratio < self.config.entry_rr_min:
            return {"action": "WAIT", "reason": f"리스크/보상 비율 부족 ({rr_ratio:.2f} < {self.config.entry_rr_min})"}
        
        # 추가 진입 조건 분석
        entry_score = self._calculate_entry_score(plan, current_price)
        
        if entry_score >= 0.7:
            action = "STRONG_BUY" if bias == "LONG" else "STRONG_SELL"
        elif entry_score >= 0.4:  # 0.5 → 0.4로 낮춤
            action = "BUY" if bias == "LONG" else "SELL"
        else:
            action = "WAIT"
            reason = "진입 점수 부족"
        
        return {
            "action": action,
            "bias": bias,
            "confidence": confidence,
            "rr_ratio": rr_ratio,
            "entry_score": entry_score,
            "entry_price": current_price,
            "stop_loss": plan.get('stop'),
            "take_profit1": plan.get('tp1'),
            "take_profit2": plan.get('tp2'),
            "timestamp": datetime.now(),
            "reason": reason if action == "WAIT" else "진입 조건 충족"
        }
    
    def _calculate_entry_score(self, plan: Dict[str, Any], current_price: float) -> float:
        """진입 점수 계산"""
        score = 0.0
        
        # 1. 신뢰도 점수 (40%)
        confidence = plan.get('confidence', 0)
        score += confidence * 0.4
        
        # 2. 리스크/보상 점수 (30%)
        rr_ratio = plan.get('risk_reward_ratio', 0)
        if rr_ratio >= 2.0:
            score += 0.3
        elif rr_ratio >= 1.5:
            score += 0.25
        elif rr_ratio >= 1.0:
            score += 0.2
        else:
            score += 0.1
        
        # 3. 시장 상황 점수 (20%)
        hybrid_info = plan.get("hybrid_info", {})
        trend_strength = hybrid_info.get("trend_strength", 0)
        entry_strength = hybrid_info.get("entry_strength", 0)
        
        market_score = (trend_strength + entry_strength) / 2
        score += market_score * 0.2
        
        # 4. 변동성 점수 (10%)
        atr = plan.get('atr', 0)
        if atr > 0:
            volatility_score = min(1.0, atr / current_price * 100)  # ATR 비율
            score += volatility_score * 0.1
        
        return min(1.0, score)
    
    def analyze_exit_timing(self, position_id: str, current_price: float, current_time: datetime) -> Dict[str, Any]:
        """포지션 종료 타이밍 분석"""
        if position_id not in self.active_positions:
            return {"action": "WAIT", "reason": "포지션을 찾을 수 없음"}
        
        position = self.active_positions[position_id]
        bias = position['bias']
        entry_price = position['entry_price']
        entry_time = position['timestamp']
        
        # 보유 시간 계산
        hold_time = current_time - entry_time
        hold_hours = hold_time.total_seconds() / 3600
        position['hold_time'] = f"{hold_hours:.1f}시간"  # 보유 시간을 position에 저장
        
        # 디버깅: position 전체 내용 출력
        print(f"DEBUG: position 전체 내용: {position}")
        
        # 1. 손절 체크 (안전한 처리)
        stop_loss = position.get('stop_loss')
        print(f"DEBUG: position_id={position_id}, stop_loss={stop_loss}, type={type(stop_loss)}")
        
        # stop_loss가 None이거나 유효하지 않은 경우 기본값 설정
        if stop_loss is None or not isinstance(stop_loss, (int, float)):
            print(f"DEBUG: stop_loss가 유효하지 않음, 기본값 설정")
            if bias == "LONG":
                stop_loss = entry_price * 0.99  # 1% 손절
            else:
                stop_loss = entry_price * 1.01  # 1% 손절
            position['stop_loss'] = stop_loss  # position 업데이트
        
        # stop_loss가 여전히 None인 경우 안전하게 처리
        if stop_loss is None:
            print(f"DEBUG: stop_loss가 여전히 None, 기본값 사용")
            stop_loss = entry_price * 0.99 if bias == "LONG" else entry_price * 1.01
        
        if (bias == "LONG" and current_price <= stop_loss) or \
           (bias == "SHORT" and current_price >= stop_loss):
            return {
                "action": "STOP_LOSS",
                "reason": f"손절가 도달 (가격: {current_price:.2f}, 손절: {stop_loss:.2f})",
                "pnl": self._calculate_pnl(bias, entry_price, current_price, position.get('size', 1.0))
            }
        
        # 2. 익절 체크 (안전한 처리)
        take_profits = position.get('take_profits', [])
        print(f"DEBUG: take_profits={take_profits}")
        
        # take_profits가 None이거나 유효하지 않은 경우 기본값 설정
        if not take_profits or not isinstance(take_profits, list):
            print(f"DEBUG: take_profits가 유효하지 않음, 기본값 설정")
            if bias == "LONG":
                take_profits = [entry_price * 1.01, entry_price * 1.02]  # 1%, 2% 익절
            else:
                take_profits = [entry_price * 0.99, entry_price * 0.98]  # 1%, 2% 익절
            position['take_profits'] = take_profits  # position 업데이트
        
        # take_profits가 여전히 유효하지 않은 경우 안전하게 처리
        if not take_profits or len(take_profits) < 2:
            print(f"DEBUG: take_profits가 여전히 유효하지 않음, 기본값 사용")
            if bias == "LONG":
                take_profits = [entry_price * 1.01, entry_price * 1.02]
            else:
                take_profits = [entry_price * 0.99, entry_price * 0.98]
        
        for i, tp in enumerate(take_profits):
            if tp is not None and isinstance(tp, (int, float)):  # 타입 체크 추가
                if (bias == "LONG" and current_price >= tp) or \
                   (bias == "SHORT" and current_price <= tp):
                    return {
                        "action": f"TAKE_PROFIT_{i+1}",
                        "reason": f"익절{i+1} 도달 (가격: {current_price:.2f}, 익절: {tp:.2f})",
                        "pnl": self._calculate_pnl(bias, entry_price, current_price, position.get('size', 1.0))
                    }
        
        # 3. 트레일링 스탑 체크
        if bias == "LONG":
            # LONG 포지션: 고점 대비 하락 체크
            if current_price > position['high_price']:
                position['high_price'] = current_price
            
            trailing_stop = position['high_price'] - (self.config.trailing_stop_atr * position.get('atr', entry_price * 0.02))
            if current_price <= trailing_stop:
                return {
                    "action": "TRAILING_STOP",
                    "reason": f"트레일링 스탑 도달 (가격: {current_price:.2f}, 트레일링: {trailing_stop:.2f})",
                    "pnl": self._calculate_pnl(bias, entry_price, current_price, position.get('size', 1.0))
                }
        else:
            # SHORT 포지션: 저점 대비 상승 체크
            if current_price < position['low_price']:
                position['low_price'] = current_price
            
            trailing_stop = position['low_price'] + (self.config.trailing_stop_atr * position.get('atr', entry_price * 0.02))
            if current_price >= trailing_stop:
                return {
                    "action": "TRAILING_STOP",
                    "reason": f"트레일링 스탑 도달 (가격: {current_price:.2f}, 트레일링: {trailing_stop:.2f})",
                    "pnl": self._calculate_pnl(bias, entry_price, current_price, position.get('size', 1.0))
                }
        
        # 4. 시간 기반 청산
        hold_minutes = hold_time.total_seconds() / 60
        if hold_minutes >= self.config.max_hold_time_hours * 60: # 시간 단위로 변경
            return {
                "action": "TIME_EXIT",
                "reason": f"최대 보유 시간 도달 ({hold_minutes:.1f}분)",
                "pnl": self._calculate_pnl(bias, entry_price, current_price, position.get('size', 1.0))
            }
        
        # 5. 시장 변화 감지
        market_change = self._detect_market_change(current_price, entry_price, bias)
        if market_change:
            return {
                "action": "MARKET_CHANGE",
                "reason": f"시장 변화 감지: {market_change}",
                "pnl": self._calculate_pnl(bias, entry_price, current_price, position.get('size', 1.0))
            }
        
        # 모든 조건을 만족하지 않으면 HOLD
        return {
            "action": "HOLD",
            "reason": "청산 조건 미충족",
            "pnl": self._calculate_pnl(bias, entry_price, current_price, position.get('size', 1.0))
        }
    
    def _calculate_trailing_stop(self, position: Dict, current_price: float) -> Optional[float]:
        """트레일링 스탑 계산"""
        bias = position['bias']
        entry_price = position['entry_price']
        atr = position.get('atr', 0)
        
        if not atr:
            return None
        
        if bias == "LONG":
            # 롱 포지션: 고점에서 ATR 배수만큼 아래
            high_price = position.get('high_price', entry_price)
            if current_price > high_price:
                position['high_price'] = current_price
                high_price = current_price
            
            trailing_stop = high_price - (atr * self.config.trailing_stop_atr)
            return max(trailing_stop, entry_price)  # 진입가 이하로는 내려가지 않음
            
        else:  # SHORT
            # 숏 포지션: 저점에서 ATR 배수만큼 위
            low_price = position.get('low_price', entry_price)
            if current_price < low_price:
                position['low_price'] = current_price
                low_price = current_price
            
            trailing_stop = low_price + (atr * self.config.trailing_stop_atr)
            return min(trailing_stop, entry_price)  # 진입가 이상으로는 올라가지 않음
    
    def _detect_market_change(self, current_price: float, entry_price: float, bias: str) -> Optional[str]:
        """시장 변화 감지"""
        price_change = (current_price - entry_price) / entry_price
        
        if bias == "LONG":
            if price_change < -0.05:  # 5% 이상 하락
                return "강한 하락 모멘텀"
            elif price_change < -0.02:  # 2% 이상 하락
                return "약한 하락 모멘텀"
        else:  # SHORT
            if price_change > 0.05:  # 5% 이상 상승
                return "강한 상승 모멘텀"
            elif price_change > 0.02:  # 2% 이상 상승
                return "약한 상승 모멘텀"
        
        return None
    
    def _calculate_pnl(self, bias: str, entry_price: float, current_price: float, size: float) -> float:
        """손익 계산"""
        if bias == "LONG":
            return (current_price - entry_price) / entry_price * size
        else:  # SHORT
            return (entry_price - current_price) / entry_price * size
    
    def open_position(self, entry_signal: Dict[str, Any], size: float = 1.0) -> str:
        """포지션 오픈"""
        position_id = f"pos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 유효한 값인지 확인
        stop_loss = entry_signal.get('stop_loss')
        take_profit1 = entry_signal.get('take_profit1')
        take_profit2 = entry_signal.get('take_profit2')
        
        # None이거나 숫자가 아닌 경우 기본값 설정
        if stop_loss is None or not isinstance(stop_loss, (int, float)):
            stop_loss = entry_signal.get('entry_price', 0) * 0.99  # 기본 1% 손절
        
        if take_profit1 is None or not isinstance(take_profit1, (int, float)):
            take_profit1 = entry_signal.get('entry_price', 0) * 1.01  # 기본 1% 익절
        
        if take_profit2 is None or not isinstance(take_profit2, (int, float)):
            take_profit2 = entry_signal.get('entry_price', 0) * 1.02  # 기본 2% 익절
        
        # 디버깅 로그
        print(f"DEBUG: stop_loss={stop_loss}, take_profit1={take_profit1}, take_profit2={take_profit2}")
        
        self.active_positions[position_id] = {
            'bias': entry_signal['bias'],
            'entry_price': entry_signal['entry_price'],
            'stop_loss': stop_loss,
            'take_profits': [take_profit1, take_profit2],
            'size': size,
            'timestamp': entry_signal['timestamp'],
            'atr': entry_signal.get('atr', 0),
            'high_price': entry_signal['entry_price'],
            'low_price': entry_signal['entry_price']
        }
        
        return position_id
    
    def close_position(self, position_id: str, reason: str, pnl: float):
        """포지션 클로즈"""
        if position_id in self.active_positions:
            position = self.active_positions.pop(position_id)
            
            # 일일 손익 업데이트
            self.daily_pnl += pnl
            
            # 날짜 변경 시 리셋
            current_date = datetime.now().date()
            if current_date != self.last_reset_date:
                self.daily_pnl = pnl
                self.last_reset_date = current_date
            
            return position
        return None
    
    def get_position_summary(self) -> Dict[str, Any]:
        """포지션 요약 정보"""
        return {
            'active_positions': len(self.active_positions),
            'daily_pnl': self.daily_pnl,
            'positions': list(self.active_positions.keys())
        }
