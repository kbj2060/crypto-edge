#!/usr/bin/env python3
"""
바이낸스 선물 API를 사용한 실제 거래 실행 클래스
"""

import hmac
import hashlib
import time
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
import os
import json


class BinanceTrader:
    """바이낸스 선물 거래 클라이언트"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        simulation_mode: bool = True
    ):
        """
        Args:
            api_key: 바이낸스 API 키 (환경 변수 BINANCE_API_KEY에서도 읽을 수 있음)
            api_secret: 바이낸스 API 시크릿 (환경 변수 BINANCE_API_SECRET에서도 읽을 수 있음)
            testnet: 테스트넷 사용 여부
            simulation_mode: 시뮬레이션 모드 (실제 주문 실행 안 함)
        """
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
        
        self.simulation_mode = simulation_mode
        
        if not self.simulation_mode and (not self.api_key or not self.api_secret):
            raise ValueError("실제 거래 모드에서는 API 키와 시크릿이 필요합니다.")
        
        if self.simulation_mode:
            print("⚠️ 시뮬레이션 모드로 실행 중 (실제 주문은 실행되지 않습니다)")
        else:
            print(f"✅ 실제 거래 모드 ({'테스트넷' if testnet else '메인넷'})")
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """API 요청 서명 생성"""
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _get_headers(self) -> Dict[str, str]:
        """API 요청 헤더 생성"""
        return {
            "X-MBX-APIKEY": self.api_key,
            "Content-Type": "application/json"
        }
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """API 요청 실행"""
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=self._get_headers(), timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, headers=self._get_headers(), timeout=10)
            elif method.upper() == "DELETE":
                response = requests.delete(url, params=params, headers=self._get_headers(), timeout=10)
            else:
                raise ValueError(f"지원하지 않는 HTTP 메서드: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"❌ API 요청 실패: {e}")
            if hasattr(e.response, 'text'):
                print(f"   응답: {e.response.text}")
            raise
    
    def get_account_info(self) -> Dict[str, Any]:
        """계좌 정보 조회"""
        if self.simulation_mode:
            return {
                "totalWalletBalance": 10000.0,
                "availableBalance": 10000.0,
                "totalUnrealizedProfit": 0.0,
                "assets": []
            }
        
        return self._request("GET", "/fapi/v2/account", signed=True)
    
    def get_position_info(self, symbol: str = "ETHUSDT") -> Optional[Dict[str, Any]]:
        """포지션 정보 조회"""
        if self.simulation_mode:
            return None
        
        params = {"symbol": symbol}
        positions = self._request("GET", "/fapi/v2/positionRisk", params=params, signed=True)
        
        for pos in positions:
            if pos['symbol'] == symbol and float(pos['positionAmt']) != 0:
                return pos
        
        return None
    
    def get_open_orders(self, symbol: str = "ETHUSDT") -> List[Dict[str, Any]]:
        """미체결 주문 조회"""
        if self.simulation_mode:
            return []
        
        params = {"symbol": symbol}
        return self._request("GET", "/fapi/v1/openOrders", params=params, signed=True)
    
    def get_current_price(self, symbol: str = "ETHUSDT") -> float:
        """현재 가격 조회"""
        params = {"symbol": symbol}
        ticker = self._request("GET", "/fapi/v1/ticker/price", params=params)
        return float(ticker['price'])
    
    def _calculate_quantity(
        self,
        symbol: str,
        usdt_amount: float,
        price: Optional[float] = None
    ) -> float:
        """USDT 금액을 수량으로 변환"""
        if price is None:
            price = self.get_current_price(symbol)
        
        # 심볼별 계약 크기 (ETHUSDT는 1)
        contract_size = 1.0
        
        # 수량 계산 (소수점 처리)
        quantity = usdt_amount / price / contract_size
        
        # 바이낸스 선물의 최소 수량 단위에 맞춤 (ETHUSDT는 보통 0.001)
        quantity = Decimal(str(quantity)).quantize(Decimal('0.001'), rounding=ROUND_DOWN)
        
        return float(quantity)
    
    def place_market_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity: Optional[float] = None,
        usdt_amount: Optional[float] = None,
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """
        시장가 주문 실행
        
        Args:
            symbol: 거래 심볼 (예: "ETHUSDT")
            side: 주문 방향 ("BUY" or "SELL")
            quantity: 수량 (quantity 또는 usdt_amount 중 하나 필수)
            usdt_amount: USDT 금액 (quantity 또는 usdt_amount 중 하나 필수)
            reduce_only: 포지션 감소만 허용 (True인 경우 새 포지션 개설 불가)
        
        Returns:
            주문 결과 딕셔너리
        """
        if quantity is None and usdt_amount is None:
            raise ValueError("quantity 또는 usdt_amount 중 하나는 필수입니다.")
        
        if self.simulation_mode:
            price = self.get_current_price(symbol)
            if quantity is None:
                quantity = self._calculate_quantity(symbol, usdt_amount, price)
            
            return {
                "orderId": int(time.time() * 1000),
                "symbol": symbol,
                "status": "FILLED",
                "side": side,
                "type": "MARKET",
                "quantity": quantity,
                "price": price,
                "executedQty": quantity,
                "cumQuote": quantity * price,
                "simulation": True
            }
        
        # 실제 주문
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "reduceOnly": "true" if reduce_only else "false"
        }
        
        if quantity is not None:
            params["quantity"] = quantity
        elif usdt_amount is not None:
            price = self.get_current_price(symbol)
            quantity = self._calculate_quantity(symbol, usdt_amount, price)
            params["quantity"] = quantity
        
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)
    
    def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        time_in_force: str = "GTC",  # GTC, IOC, FOK
        reduce_only: bool = False
    ) -> Dict[str, Any]:
        """지정가 주문 실행"""
        if self.simulation_mode:
            return {
                "orderId": int(time.time() * 1000),
                "symbol": symbol,
                "status": "NEW",
                "side": side,
                "type": "LIMIT",
                "quantity": quantity,
                "price": price,
                "timeInForce": time_in_force,
                "simulation": True
            }
        
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "quantity": quantity,
            "price": price,
            "timeInForce": time_in_force,
            "reduceOnly": "true" if reduce_only else "false"
        }
        
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """주문 취소"""
        if self.simulation_mode:
            return {"orderId": order_id, "status": "CANCELED", "simulation": True}
        
        params = {
            "symbol": symbol,
            "orderId": order_id
        }
        return self._request("DELETE", "/fapi/v1/order", params=params, signed=True)
    
    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """모든 주문 취소"""
        if self.simulation_mode:
            return {"symbol": symbol, "status": "CANCELED", "simulation": True}
        
        params = {"symbol": symbol}
        return self._request("DELETE", "/fapi/v1/allOpenOrders", params=params, signed=True)
    
    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """레버리지 설정"""
        if self.simulation_mode:
            return {"leverage": leverage, "symbol": symbol, "simulation": True}
        
        params = {
            "symbol": symbol,
            "leverage": leverage
        }
        return self._request("POST", "/fapi/v1/leverage", params=params, signed=True)
    
    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Dict[str, Any]:
        """마진 타입 설정 (ISOLATED 또는 CROSSED)"""
        if self.simulation_mode:
            return {"marginType": margin_type, "symbol": symbol, "simulation": True}
        
        params = {
            "symbol": symbol,
            "marginType": margin_type
        }
        return self._request("POST", "/fapi/v1/marginType", params=params, signed=True)

