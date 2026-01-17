#!/usr/bin/env python3
"""
바이낸스 선물 API를 사용한 실제 거래 실행 클래스
python-binance 라이브러리 사용
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
import os
from pathlib import Path

# .env 파일 로드 (가장 먼저)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv가 없으면 환경 변수만 사용
except Exception:
    pass  # .env 파일 로드 실패해도 계속 진행

# python-binance 라이브러리 import
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("⚠️ python-binance 라이브러리가 설치되지 않았습니다.")
    print("   pip install python-binance로 설치하세요.")


class BinanceTrader:
    """바이낸스 거래 클라이언트 (python-binance 라이브러리 사용)"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        demo: bool = False,
        simulation_mode: bool = True,
        use_futures: bool = True  # Futures 거래 사용 여부
    ):
        """
        Args:
            api_key: 바이낸스 API 키 (환경 변수 BINANCE_API_KEY에서도 읽을 수 있음)
            api_secret: 바이낸스 API 시크릿 (환경 변수 BINANCE_API_SECRET에서도 읽을 수 있음)
            demo: Demo Trading 사용 여부
            simulation_mode: 시뮬레이션 모드 (실제 주문 실행 안 함)
            use_futures: Futures 거래 사용 여부 (True: Futures, False: Spot)
        """
        # simulation_mode와 demo를 먼저 설정 (다른 코드에서 사용하기 전에)
        self.simulation_mode = simulation_mode
        self.demo = demo
        self.use_futures = use_futures
        
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET", "")
        
        # 디버깅: 환경 변수 로드 확인
        if not self.simulation_mode:
            print("🔍 API 키 로드 상태 확인:")
            print(f"   현재 디렉토리: {os.getcwd()}")
            print(f"   .env 파일 경로: {Path(__file__).parent.parent / '.env'}")
            print(f"   .env 파일 존재 여부: {(Path(__file__).parent.parent / '.env').exists()}")
            
            # 환경 변수 직접 확인
            env_api_key = os.getenv("BINANCE_API_KEY", "")
            env_api_secret = os.getenv("BINANCE_API_SECRET", "")
            
            print(f"   환경 변수 BINANCE_API_KEY: {'있음' if env_api_key else '없음'} (길이: {len(env_api_key)})")
            print(f"   환경 변수 BINANCE_API_SECRET: {'있음' if env_api_secret else '없음'} (길이: {len(env_api_secret)})")
            
            if not self.api_key:
                print("⚠️ BINANCE_API_KEY 환경 변수를 찾을 수 없습니다.")
                print("   .env 파일에 BINANCE_API_KEY=your_key 형식으로 설정하세요.")
            else:
                print(f"✅ BINANCE_API_KEY 로드 완료 (길이: {len(self.api_key)})")
                print(f"   API Key 처음 10자: {self.api_key[:10]}...")
                print(f"   API Key 마지막 10자: ...{self.api_key[-10:]}")
                
            if not self.api_secret:
                print("⚠️ BINANCE_API_SECRET 환경 변수를 찾을 수 없습니다.")
                print("   .env 파일에 BINANCE_API_SECRET=your_secret 형식으로 설정하세요.")
            else:
                print(f"✅ BINANCE_API_SECRET 로드 완료 (길이: {len(self.api_secret)})")
                print(f"   API Secret 처음 10자: {self.api_secret[:10]}...")
                print(f"   API Secret 마지막 10자: ...{self.api_secret[-10:]}")
        
        if not self.simulation_mode and (not self.api_key or not self.api_secret):
            raise ValueError("실제 거래 모드에서는 API 키와 시크릿이 필요합니다.")
        
        # python-binance Client 초기화
        self.client = None
        if not self.simulation_mode and BINANCE_AVAILABLE:
            try:
                if self.demo:
                    # Demo Trading: testnet=True 사용
                    self.client = Client(
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        testnet=True
                    )
                else:
                    # Mainnet
                    self.client = Client(
                        api_key=self.api_key,
                        api_secret=self.api_secret,
                        testnet=False
                    )
            except Exception as e:
                print(f"⚠️ python-binance Client 초기화 실패: {e}")
                self.client = None
        
        if self.simulation_mode:
            print("⚠️ 시뮬레이션 모드로 실행 중 (실제 주문은 실행되지 않습니다)")
        else:
            if self.demo:
                print(f"✅ 실제 거래 모드 (Demo Trading)")
            else:
                print(f"✅ 실제 거래 모드 (메인넷)")
            
            # API 연결 테스트 (계좌 정보 조회)
            if self.client:
                try:
                    print("🔍 API 연결 테스트 중...")
                    account_info = self.get_account_info()
                    if account_info:
                        # Demo는 Spot API, Mainnet은 Futures API
                        if self.demo:
                            balances = account_info.get("balances", [])
                            total_balance = sum(float(b.get("free", 0)) + float(b.get("locked", 0)) for b in balances)
                            print(f"✅ API 연결 성공! 계좌 잔액: {total_balance:.2f} USDT (Demo Trading)")
                        else:
                            # Futures API 응답 처리
                            total_balance = float(account_info.get("totalWalletBalance", 0))
                            print(f"✅ API 연결 성공! 계좌 잔액: {total_balance:.2f} USDT (Futures)")
                except Exception as e:
                    print(f"❌ API 연결 실패: {e}")
                    print("   API 키와 시크릿이 올바른지 확인하세요.")
                    import traceback
                    traceback.print_exc()
    
    def get_account_info(self) -> Dict[str, Any]:
        """계좌 정보 조회"""
        if self.simulation_mode:
            return {
                "totalWalletBalance": 10000.0,
                "availableBalance": 10000.0,
                "totalUnrealizedProfit": 0.0,
                "assets": []
            }
        
        if not self.client:
            raise ValueError("python-binance Client가 초기화되지 않았습니다.")
        
        # Futures 거래 사용 시
        if self.use_futures:
            try:
                # Futures 계좌 정보 조회
                account = self.client.futures_account()
                return {
                    "totalWalletBalance": float(account.get("totalWalletBalance", 0)),
                    "availableBalance": float(account.get("availableBalance", 0)),
                    "totalUnrealizedProfit": float(account.get("totalUnrealizedProfit", 0)),
                    "assets": account.get("assets", []),
                    "positions": account.get("positions", [])
                }
            except Exception as e:
                print(f"⚠️ Futures 계좌 정보 조회 실패: {e}")
                # Fallback: Spot 계좌 정보 사용
                account = self.client.get_account()
                balances = account.get("balances", [])
                usdt_balance = None
                for balance in balances:
                    if balance.get("asset") == "USDT":
                        usdt_balance = float(balance.get("free", 0.0))
                        break
                return {
                    "totalWalletBalance": usdt_balance if usdt_balance is not None else 0.0,
                    "availableBalance": usdt_balance if usdt_balance is not None else 0.0,
                    "totalUnrealizedProfit": 0.0,
                    "assets": [],
                    "balances": balances
                }
        else:
            # Spot 거래 사용 시
            account = self.client.get_account()
            balances = account.get("balances", [])
            usdt_balance = None
            for balance in balances:
                if balance.get("asset") == "USDT":
                    usdt_balance = float(balance.get("free", 0.0))
                    break
            
            return {
                "totalWalletBalance": usdt_balance if usdt_balance is not None else 0.0,
                "availableBalance": usdt_balance if usdt_balance is not None else 0.0,
                "totalUnrealizedProfit": 0.0,
                "assets": [],
                "balances": balances
            }
    
    def get_position_info(self, symbol: str = "ETHUSDT") -> Optional[Dict[str, Any]]:
        """포지션 정보 조회 (Futures 전용, Spot에서는 None 반환)"""
        if self.simulation_mode:
            return None
        
        if not self.use_futures:
            return None  # Spot 거래는 포지션 개념 없음
        
        if not self.client:
            return None
        
        try:
            # Futures 포지션 정보 조회
            positions = self.client.futures_position_information(symbol=symbol)
            for pos in positions:
                position_amt = float(pos.get("positionAmt", 0))
                if position_amt != 0:
                    return pos
            return None
        except Exception as e:
            print(f"⚠️ 포지션 정보 조회 실패: {e}")
            return None
    
    def get_open_orders(self, symbol: str = "ETHUSDT") -> List[Dict[str, Any]]:
        """미체결 주문 조회"""
        if self.simulation_mode:
            return []
        
        if not self.client:
            return []
        
        if self.use_futures:
            # Futures 미체결 주문 조회
            return self.client.futures_get_open_orders(symbol=symbol)
        else:
            # Spot 미체결 주문 조회
            return self.client.get_open_orders(symbol=symbol)
    
    def get_current_price(self, symbol: str = "ETHUSDT") -> float:
        """현재 가격 조회"""
        if not self.client:
            # 시뮬레이션 모드나 클라이언트가 없을 때는 기본값 반환
            if self.simulation_mode:
                return 3000.0  # 기본 ETH 가격
            raise ValueError("python-binance Client가 초기화되지 않았습니다.")
        
        if self.use_futures:
            # Futures 가격 조회
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        else:
            # Spot 가격 조회
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
    
    def _get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """심볼 정보 조회 (LOT_SIZE 필터 확인용)"""
        if not self.client:
            return None
        
        try:
            if self.use_futures:
                # Futures 심볼 정보
                exchange_info = self.client.futures_exchange_info()
            else:
                # Spot 심볼 정보
                exchange_info = self.client.get_exchange_info()
            
            for s in exchange_info.get("symbols", []):
                if s.get("symbol") == symbol:
                    return s
            return None
        except Exception as e:
            print(f"⚠️ 심볼 정보 조회 실패: {e}")
            return None
    
    def _adjust_quantity_to_lot_size(
        self,
        symbol: str,
        quantity: float
    ) -> float:
        """수량을 LOT_SIZE 필터에 맞게 조정"""
        symbol_info = self._get_symbol_info(symbol)
        if not symbol_info:
            # 심볼 정보를 가져올 수 없으면 기본값 사용
            quantity = Decimal(str(quantity)).quantize(Decimal('0.001'), rounding=ROUND_DOWN)
            return float(quantity)
        
        # LOT_SIZE 필터 찾기
        lot_size_filter = None
        for f in symbol_info.get("filters", []):
            if f.get("filterType") == "LOT_SIZE":
                lot_size_filter = f
                break
        
        if not lot_size_filter:
            # LOT_SIZE 필터가 없으면 기본값 사용
            quantity = Decimal(str(quantity)).quantize(Decimal('0.001'), rounding=ROUND_DOWN)
            return float(quantity)
        
        # 최소 수량
        min_qty = float(lot_size_filter.get("minQty", "0.001"))
        # 최대 수량
        max_qty = float(lot_size_filter.get("maxQty", "1000000"))
        # 수량 단위 (stepSize)
        step_size = float(lot_size_filter.get("stepSize", "0.001"))
        
        # stepSize에 맞게 반올림
        if step_size > 0:
            # stepSize의 소수점 자릿수 계산
            step_precision = len(str(step_size).rstrip('0').split('.')[-1]) if '.' in str(step_size) else 0
            # stepSize의 배수로 조정
            quantity = (quantity // step_size) * step_size
            # 소수점 자릿수 맞춤
            quantity = Decimal(str(quantity)).quantize(Decimal(str(step_size)), rounding=ROUND_DOWN)
        else:
            quantity = Decimal(str(quantity)).quantize(Decimal('0.001'), rounding=ROUND_DOWN)
        
        quantity = float(quantity)
        
        # 최소/최대 수량 확인
        if quantity < min_qty:
            raise ValueError(f"수량이 최소값보다 작습니다: {quantity} < {min_qty}")
        if quantity > max_qty:
            raise ValueError(f"수량이 최대값보다 큽니다: {quantity} > {max_qty}")
        
        return quantity
    
    def _calculate_quantity(
        self,
        symbol: str,
        usdt_amount: float,
        price: Optional[float] = None
    ) -> float:
        """USDT 금액을 수량으로 변환 (LOT_SIZE 필터 적용)"""
        if price is None:
            price = self.get_current_price(symbol)
        
        # 심볼별 계약 크기 (ETHUSDT는 1)
        contract_size = 1.0
        
        # 수량 계산 (소수점 처리)
        quantity = usdt_amount / price / contract_size
        
        # LOT_SIZE 필터에 맞게 조정
        quantity = self._adjust_quantity_to_lot_size(symbol, quantity)
        
        return quantity
    
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
        
        if not self.client:
            raise ValueError("python-binance Client가 초기화되지 않았습니다.")
        
        # 수량 계산
        if quantity is None:
            price = self.get_current_price(symbol)
            quantity = self._calculate_quantity(symbol, usdt_amount, price)
        
        # LOT_SIZE 필터에 맞게 수량 조정 (주문 전 최종 검증)
        try:
            quantity = self._adjust_quantity_to_lot_size(symbol, quantity)
        except Exception as e:
            raise ValueError(f"수량 조정 실패: {e}")
        
        # Futures 거래 사용 시
        if self.use_futures:
            # Futures 주문 실행
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=quantity,
                reduceOnly=reduce_only
            )
        else:
            # Spot 주문 실행 (reduceOnly 파라미터 없음)
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_MARKET,
                quantity=quantity
            )
        
        return order
    
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
        
        if not self.client:
            raise ValueError("python-binance Client가 초기화되지 않았습니다.")
        
        # Futures 거래 사용 시
        if self.use_futures:
            # Futures 지정가 주문
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="LIMIT",
                timeInForce=time_in_force,
                quantity=quantity,
                price=price,
                reduceOnly=reduce_only
            )
        else:
            # Spot 지정가 주문 (reduceOnly 파라미터 없음)
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=Client.ORDER_TYPE_LIMIT,
                timeInForce=time_in_force,
                quantity=quantity,
                price=price
            )
        
        return order
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """주문 취소"""
        if self.simulation_mode:
            return {"orderId": order_id, "status": "CANCELED", "simulation": True}
        
        if not self.client:
            raise ValueError("python-binance Client가 초기화되지 않았습니다.")
        
        if self.use_futures:
            # Futures 주문 취소
            return self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
        else:
            # Spot 주문 취소
            return self.client.cancel_order(symbol=symbol, orderId=order_id)
    
    def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """모든 주문 취소"""
        if self.simulation_mode:
            return {"symbol": symbol, "status": "CANCELED", "simulation": True}
        
        if not self.client:
            raise ValueError("python-binance Client가 초기화되지 않았습니다.")
        
        # python-binance는 cancel_all_orders를 지원하지 않을 수 있으므로
        # open_orders를 가져와서 각각 취소
        open_orders = self.get_open_orders(symbol)
        results = []
        for order in open_orders:
            try:
                result = self.cancel_order(symbol, order['orderId'])
                results.append(result)
            except Exception as e:
                print(f"⚠️ 주문 취소 실패 (orderId: {order['orderId']}): {e}")
        
        return {"symbol": symbol, "cancelled": len(results), "results": results}
    
    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """레버리지 설정 (Futures 전용, Spot에서는 무시)"""
        if self.simulation_mode:
            return {"leverage": leverage, "symbol": symbol, "simulation": True}
        
        if not self.use_futures:
            print("⚠️ Spot API는 레버리지 설정을 지원하지 않습니다.")
            return {"leverage": leverage, "symbol": symbol, "note": "Spot API does not support leverage"}
        
        if not self.client:
            raise ValueError("python-binance Client가 초기화되지 않았습니다.")
        
        try:
            # Futures 레버리지 설정
            result = self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            return result
        except Exception as e:
            print(f"⚠️ 레버리지 설정 실패: {e}")
            return {"leverage": leverage, "symbol": symbol, "error": str(e)}
    
    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Dict[str, Any]:
        """마진 타입 설정 (ISOLATED 또는 CROSSED, Futures 전용)"""
        if self.simulation_mode:
            return {"marginType": margin_type, "symbol": symbol, "simulation": True}
        
        if not self.use_futures:
            print("⚠️ Spot API는 마진 타입 설정을 지원하지 않습니다.")
            return {"marginType": margin_type, "symbol": symbol, "note": "Spot API does not support margin type"}
        
        if not self.client:
            raise ValueError("python-binance Client가 초기화되지 않았습니다.")
        
        try:
            # Futures 마진 타입 설정
            result = self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
            return result
        except Exception as e:
            print(f"⚠️ 마진 타입 설정 실패: {e}")
            return {"marginType": margin_type, "symbol": symbol, "error": str(e)}
