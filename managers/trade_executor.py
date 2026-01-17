#!/usr/bin/env python3
"""
거래 결정을 실제 주문으로 변환하고 실행하는 클래스
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
from managers.binance_trader import BinanceTrader
from utils.telegram import send_telegram_message
import json


class TradeExecutor:
    """거래 실행 클래스 - TradeDecisionEngine의 결정을 실제 주문으로 변환"""
    
    def __init__(
        self,
        binance_trader: BinanceTrader,
        symbol: str = "ETHUSDT",
        max_position_size_usdt: float = 1000.0,  # 최대 포지션 크기 (USDT)
        default_leverage: int = 10,
        use_telegram: bool = True
    ):
        """
        Args:
            binance_trader: BinanceTrader 인스턴스
            symbol: 거래 심볼
            max_position_size_usdt: 최대 포지션 크기 (USDT)
            default_leverage: 기본 레버리지
            use_telegram: 텔레그램 알림 사용 여부
        """
        self.trader = binance_trader
        self.symbol = symbol
        self.max_position_size_usdt = max_position_size_usdt
        self.default_leverage = default_leverage
        self.use_telegram = use_telegram
        
        # 레버리지 및 마진 타입 설정
        self._initialize_account_settings()
    
    def _initialize_account_settings(self):
        """계좌 설정 초기화"""
        try:
            # 레버리지 설정
            self.trader.set_leverage(self.symbol, self.default_leverage)
            print(f"✅ 레버리지 설정: {self.default_leverage}x")
            
            # 마진 타입 설정 (격리 마진)
            self.trader.set_margin_type(self.symbol, "ISOLATED")
            print("✅ 마진 타입 설정: ISOLATED")
        except Exception as e:
            print(f"⚠️ 계좌 설정 실패: {e}")
    
    def _send_notification(self, message: str, level: str = "INFO"):
        """알림 전송"""
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌"
        }.get(level, "ℹ️")
        
        full_message = f"{prefix} {message}"
        print(full_message)
        
        if self.use_telegram:
            try:
                send_telegram_message(full_message)
            except Exception as e:
                print(f"⚠️ 텔레그램 전송 실패: {e}")
    
    def _calculate_position_size(
        self,
        decision: Dict[str, Any],
        account_balance: float
    ) -> float:
        """
        결정에 따라 포지션 크기 계산
        
        Args:
            decision: TradeDecisionEngine의 결정
            account_balance: 계좌 잔액
        
        Returns:
            포지션 크기 (USDT)
        """
        # sizing 정보가 있으면 사용
        sizing = decision.get("sizing", {})
        if sizing:
            usdt_amount = sizing.get("usdt_amount", 0.0)
            if usdt_amount > 0:
                return min(usdt_amount, self.max_position_size_usdt)
        
        # net_score 기반 계산
        net_score = abs(decision.get("net_score", 0.0))
        confidence = decision.get("confidence", "LOW")
        
        # 신뢰도에 따른 가중치
        confidence_multiplier = {
            "VERY_HIGH": 1.0,
            "HIGH": 0.8,
            "MEDIUM": 0.6,
            "LOW": 0.4
        }.get(confidence, 0.4)
        
        # net_score에 따른 비율 (0.0 ~ 1.0)
        score_ratio = min(net_score, 1.0)
        
        # 기본 리스크 비율 (계좌의 2%)
        base_risk_pct = 0.02
        
        # 최종 포지션 크기 계산
        position_size = account_balance * base_risk_pct * score_ratio * confidence_multiplier
        
        # 최대값 제한
        return min(position_size, self.max_position_size_usdt)
    
    def execute_decision(self, decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        거래 결정 실행 (Spot 거래: LONG=매수, SHORT=매도)
        
        Args:
            decision: TradeDecisionEngine의 final_decision
        
        Returns:
            주문 결과 또는 None
        """
        action = decision.get("action", "HOLD")
        
        if action == "HOLD":
            return None
        
        try:
            # 계좌 정보 조회
            account_info = self.trader.get_account_info()
            available_balance = float(account_info.get("availableBalance", 0.0))
            
            # Spot 거래: 잔액 확인
            if self.trader.use_futures:
                # Futures 거래 로직 (기존)
                if available_balance < 10.0:
                    self._send_notification(
                        f"잔액 부족: {available_balance:.2f} USDT",
                        "WARNING"
                    )
                    return None
                
                # 현재 포지션 확인
                current_position = self.trader.get_position_info(self.symbol)
                current_position_amt = 0.0
                if current_position:
                    current_position_amt = float(current_position.get("positionAmt", 0.0))
                
                # 포지션 크기 계산
                position_size_usdt = self._calculate_position_size(decision, available_balance)
                
                if position_size_usdt < 10.0:
                    return None
                
                # 주문 실행
                if action == "LONG":
                    if current_position_amt < 0:
                        self._close_position(abs(current_position_amt))
                    result = self._open_long_position(position_size_usdt, decision)
                elif action == "SHORT":
                    if current_position_amt > 0:
                        self._close_position(current_position_amt)
                    result = self._open_short_position(position_size_usdt, decision)
                else:
                    return None
            else:
                # Spot 거래 로직: LONG=매수, SHORT=매도
                if action == "LONG":
                    # LONG 신호: ETH 매수 (USDT로 ETH 구매)
                    if available_balance < 10.0:
                        self._send_notification(
                            f"USDT 잔액 부족: {available_balance:.2f} USDT",
                            "WARNING"
                        )
                        return None
                    
                    # 포지션 크기 계산
                    position_size_usdt = self._calculate_position_size(decision, available_balance)
                    
                    if position_size_usdt < 10.0:
                        return None
                    
                    # ETH 매수
                    result = self._open_long_position(position_size_usdt, decision)
                    
                elif action == "SHORT":
                    # SHORT 신호: ETH 매도 (보유한 ETH를 USDT로 판매)
                    # Spot 거래에서는 보유한 ETH 잔액 확인 필요
                    balances = account_info.get("balances", [])
                    eth_balance = 0.0
                    eth_symbol = self.symbol.replace("USDT", "")  # "ETHUSDT" -> "ETH"
                    
                    for balance in balances:
                        if balance.get("asset") == eth_symbol:
                            eth_balance = float(balance.get("free", 0.0))
                            break
                    
                    if eth_balance < 0.001:  # 최소 거래 수량 (ETH는 보통 0.001 이상)
                        self._send_notification(
                            f"ETH 잔액 부족: {eth_balance:.8f} ETH (매도 불가, SHORT 신호 무시)",
                            "WARNING"
                        )
                        print(f"⚠️ SHORT 신호 무시: ETH 잔액이 부족합니다 ({eth_balance:.8f} ETH)")
                        return None
                    
                    # 보유한 ETH 전부 매도 또는 일부만 매도
                    # 일단 보유한 ETH의 일정 비율만 매도하도록 설정
                    sell_ratio = 0.5  # 50%만 매도
                    sell_quantity = eth_balance * sell_ratio
                    
                    # LOT_SIZE 필터에 맞게 수량 조정
                    try:
                        sell_quantity = self.trader._adjust_quantity_to_lot_size(self.symbol, sell_quantity)
                    except Exception as e:
                        self._send_notification(
                            f"수량 조정 실패: {e}",
                            "WARNING"
                        )
                        return None
                    
                    # 최소 거래 수량 확인 (조정된 수량 기준)
                    if sell_quantity < 0.001:
                        self._send_notification(
                            f"매도 수량 부족: {sell_quantity:.8f} ETH (최소 0.001 ETH 필요)",
                            "WARNING"
                        )
                        return None
                    
                    # ETH 매도
                    try:
                        result = self._open_short_position(sell_quantity, decision)
                    except Exception as e:
                        # 매도 실패 시 상세 에러 메시지
                        error_msg = str(e)
                        if "insufficient balance" in error_msg.lower() or "잔액" in error_msg:
                            self._send_notification(
                                f"ETH 매도 실패: 잔액 부족 ({eth_balance:.8f} ETH)",
                                "ERROR"
                            )
                        else:
                            self._send_notification(
                                f"ETH 매도 실패: {error_msg}",
                                "ERROR"
                            )
                        raise
                else:
                    return None
            
            # 결과 로깅
            if result:
                self._log_trade_execution(decision, result)
            
            return result
        
        except Exception as e:
            self._send_notification(f"주문 실행 실패: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return None
    
    def _open_long_position(self, usdt_amount: float, decision: Dict[str, Any]) -> Dict[str, Any]:
        """LONG 포지션 개설 (Spot: ETH 매수, Futures: 롱 포지션)"""
        try:
            current_price = self.trader.get_current_price(self.symbol)
            quantity = self.trader._calculate_quantity(self.symbol, usdt_amount, current_price)
            
            result = self.trader.place_market_order(
                symbol=self.symbol,
                side="BUY",
                quantity=quantity
            )
            
            if self.trader.use_futures:
                msg = f"LONG 포지션 개설: {quantity:.4f} {self.symbol.replace('USDT', '')} @ ${current_price:.2f}"
            else:
                msg = f"ETH 매수: {quantity:.4f} ETH @ ${current_price:.2f} (USDT {usdt_amount:.2f})"
            
            self._send_notification(msg, "SUCCESS")
            
            return result
        
        except Exception as e:
            self._send_notification(f"LONG 포지션 개설 실패: {e}", "ERROR")
            raise
    
    def _open_short_position(self, quantity: float, decision: Dict[str, Any]) -> Dict[str, Any]:
        """SHORT 포지션 개설 (Spot: ETH 매도, Futures: 숏 포지션)"""
        try:
            current_price = self.trader.get_current_price(self.symbol)
            
            # Spot 거래에서는 quantity가 이미 ETH 수량
            # Futures 거래에서는 USDT 금액을 수량으로 변환 필요
            if self.trader.use_futures:
                quantity = self.trader._calculate_quantity(self.symbol, quantity, current_price)
            
            result = self.trader.place_market_order(
                symbol=self.symbol,
                side="SELL",
                quantity=quantity
            )
            
            if self.trader.use_futures:
                msg = f"SHORT 포지션 개설: {quantity:.4f} {self.symbol.replace('USDT', '')} @ ${current_price:.2f}"
            else:
                usdt_amount = quantity * current_price
                msg = f"ETH 매도: {quantity:.4f} ETH @ ${current_price:.2f} (USDT {usdt_amount:.2f})"
            
            self._send_notification(msg, "SUCCESS")
            
            return result
        
        except Exception as e:
            self._send_notification(f"SHORT 포지션 개설 실패: {e}", "ERROR")
            raise
    
    def _close_position(self, quantity: float) -> Dict[str, Any]:
        """포지션 청산"""
        try:
            current_position = self.trader.get_position_info(self.symbol)
            if not current_position:
                return None
            
            position_amt = float(current_position.get("positionAmt", 0.0))
            if position_amt == 0:
                return None
            
            # 청산 방향 결정
            side = "SELL" if position_amt > 0 else "BUY"
            close_quantity = min(abs(position_amt), quantity)
            
            result = self.trader.place_market_order(
                symbol=self.symbol,
                side=side,
                quantity=close_quantity,
                reduce_only=True
            )
            
            self._send_notification(
                f"포지션 청산: {close_quantity:.4f} {self.symbol.replace('USDT', '')}",
                "INFO"
            )
            
            return result
        
        except Exception as e:
            self._send_notification(f"포지션 청산 실패: {e}", "ERROR")
            raise
    
    def _log_trade_execution(self, decision: Dict[str, Any], order_result: Dict[str, Any]):
        """거래 실행 로그"""
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "action": decision.get("action"),
            "net_score": decision.get("net_score"),
            "confidence": decision.get("confidence"),
            "order_id": order_result.get("orderId"),
            "quantity": order_result.get("executedQty") or order_result.get("quantity"),
            "price": order_result.get("price"),
            "simulation": order_result.get("simulation", False)
        }
        
        # 로그 파일에 저장 (선택사항)
        try:
            import os
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file = f"{log_dir}/trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            with open(log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")
        except Exception as e:
            print(f"⚠️ 로그 저장 실패: {e}")

