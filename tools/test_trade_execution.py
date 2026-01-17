#!/usr/bin/env python3
"""
거래 실행 기능 테스트 스크립트

테스트 항목:
1. API 연결 및 계좌 정보 조회
2. 잔액 확인 (Demo Trading)
3. 거래 결정 생성 및 실행 테스트
4. 주문 실행 테스트 (Demo Trading)
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import os
from dotenv import load_dotenv
from managers.binance_trader import BinanceTrader
from managers.trade_executor import TradeExecutor

# .env 파일 로드
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)


def test_api_connection():
    """1. API 연결 및 계좌 정보 조회 테스트"""
    print("=" * 60)
    print("1. API 연결 및 계좌 정보 조회 테스트")
    print("=" * 60)
    
    try:
        trader = BinanceTrader(
            demo=True,
            simulation_mode=False,
            use_futures=False  # Spot 거래 사용
        )
        
        print("✅ BinanceTrader 초기화 성공")
        
        # 계좌 정보 조회
        account_info = trader.get_account_info()
        print(f"\n📊 계좌 정보:")
        print(f"   totalWalletBalance: {account_info.get('totalWalletBalance', 0):.2f} USDT")
        print(f"   availableBalance: {account_info.get('availableBalance', 0):.2f} USDT")
        
        if 'balances' in account_info:
            print(f"\n💰 잔액 상세:")
            for balance in account_info['balances']:
                asset = balance.get('asset', '')
                free = float(balance.get('free', 0))
                locked = float(balance.get('locked', 0))
                if free > 0 or locked > 0:
                    print(f"   {asset}: {free:.8f} (사용 가능: {free:.8f}, 잠김: {locked:.8f})")
        
        # 현재 가격 조회
        current_price = trader.get_current_price("ETHUSDT")
        print(f"\n💹 현재 ETHUSDT 가격: {current_price:.2f} USDT")
        
        return trader
        
    except Exception as e:
        print(f"❌ API 연결 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_symbol_info(trader: BinanceTrader):
    """1-1. 심볼 정보 및 LOT_SIZE 필터 확인"""
    print("\n" + "-" * 60)
    print("1-1. 심볼 정보 및 LOT_SIZE 필터 확인")
    print("-" * 60)
    
    try:
        symbol_info = trader._get_symbol_info("ETHUSDT")
        if symbol_info:
            print(f"✅ 심볼 정보 조회 성공: {symbol_info.get('symbol')}")
            
            # LOT_SIZE 필터 찾기
            lot_size_filter = None
            for f in symbol_info.get("filters", []):
                if f.get("filterType") == "LOT_SIZE":
                    lot_size_filter = f
                    break
            
            if lot_size_filter:
                print(f"\n📊 LOT_SIZE 필터:")
                print(f"   최소 수량 (minQty): {lot_size_filter.get('minQty')}")
                print(f"   최대 수량 (maxQty): {lot_size_filter.get('maxQty')}")
                print(f"   수량 단위 (stepSize): {lot_size_filter.get('stepSize')}")
                
                # 수량 조정 테스트
                test_quantities = [0.0005, 0.001, 0.0015, 0.0023, 0.01]
                print(f"\n🔧 수량 조정 테스트:")
                for qty in test_quantities:
                    try:
                        adjusted = trader._adjust_quantity_to_lot_size("ETHUSDT", qty)
                        print(f"   {qty:.6f} → {adjusted:.6f}")
                    except Exception as e:
                        print(f"   {qty:.6f} → ❌ {e}")
            else:
                print("⚠️ LOT_SIZE 필터를 찾을 수 없습니다.")
        else:
            print("⚠️ 심볼 정보를 가져올 수 없습니다.")
    except Exception as e:
        print(f"❌ 심볼 정보 조회 실패: {e}")
        import traceback
        traceback.print_exc()


def test_trade_executor(trader: BinanceTrader):
    """2. TradeExecutor 초기화 테스트"""
    print("\n" + "=" * 60)
    print("2. TradeExecutor 초기화 테스트")
    print("=" * 60)
    
    try:
        executor = TradeExecutor(
            binance_trader=trader,
            symbol="ETHUSDT",
            max_position_size_usdt=100.0,  # 테스트용 작은 금액
            default_leverage=10,
            use_telegram=False  # 테스트 중에는 텔레그램 알림 비활성화
        )
        
        print("✅ TradeExecutor 초기화 성공")
        return executor
        
    except Exception as e:
        print(f"❌ TradeExecutor 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_decision_execution(executor: TradeExecutor):
    """3. 거래 결정 실행 테스트"""
    print("\n" + "=" * 60)
    print("3. 거래 결정 실행 테스트")
    print("=" * 60)
    
    # 테스트용 거래 결정 생성 (LONG)
    test_decision_long = {
        "action": "LONG",
        "net_score": 0.5,
        "confidence": "MEDIUM",
        "reason": "테스트용 LONG 주문",
        "meta": {
            "meta_labeling": {
                "should_execute": True,
                "probability": 0.6,
                "prediction": 1,
                "confidence": "MEDIUM"
            }
        }
    }
    
    # 테스트용 거래 결정 생성 (SHORT)
    test_decision_short = {
        "action": "SHORT",
        "net_score": -0.5,
        "confidence": "MEDIUM",
        "reason": "테스트용 SHORT 주문",
        "meta": {
            "meta_labeling": {
                "should_execute": True,
                "probability": 0.6,
                "prediction": 1,
                "confidence": "MEDIUM"
            }
        }
    }
    
    print("\n📝 테스트 1: LONG 주문 실행 테스트")
    print(f"   결정: {test_decision_long}")
    
    try:
        result = executor.execute_decision(test_decision_long)
        if result:
            print(f"✅ LONG 주문 실행 성공!")
            print(f"   결과: {result}")
        else:
            print("⚠️ LONG 주문 실행 결과 없음 (잔액 부족 또는 기타 이유)")
    except Exception as e:
        print(f"❌ LONG 주문 실행 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 잠시 대기 (주문 처리 시간)
    import time
    print("\n⏳ 3초 대기 중...")
    time.sleep(3)
    
    print("\n📝 테스트 2: SHORT 주문 실행 테스트")
    print(f"   결정: {test_decision_short}")
    
    try:
        result = executor.execute_decision(test_decision_short)
        if result:
            print(f"✅ SHORT 주문 실행 성공!")
            print(f"   결과: {result}")
        else:
            print("⚠️ SHORT 주문 실행 결과 없음 (잔액 부족 또는 기타 이유)")
    except Exception as e:
        print(f"❌ SHORT 주문 실행 실패: {e}")
        import traceback
        traceback.print_exc()


def test_position_info(trader: BinanceTrader):
    """4. 포지션 정보 조회 테스트"""
    print("\n" + "=" * 60)
    print("4. 포지션 정보 조회 테스트")
    print("=" * 60)
    
    try:
        # 현재 포지션 조회 (Futures 전용, Spot에서는 None)
        position = trader.get_position_info("ETHUSDT")
        if position:
            print(f"📊 현재 포지션:")
            print(f"   {position}")
        else:
            print("ℹ️ 현재 포지션 없음 (Spot API는 포지션 개념이 없습니다)")
        
        # 미체결 주문 조회
        open_orders = trader.get_open_orders("ETHUSDT")
        if open_orders:
            print(f"\n📋 미체결 주문 ({len(open_orders)}개):")
            for order in open_orders:
                print(f"   주문 ID: {order.get('orderId')}, "
                      f"심볼: {order.get('symbol')}, "
                      f"방향: {order.get('side')}, "
                      f"수량: {order.get('origQty')}, "
                      f"가격: {order.get('price')}, "
                      f"상태: {order.get('status')}")
        else:
            print("\nℹ️ 미체결 주문 없음")
        
        # Spot 거래에서는 잔액으로 확인
        print("\n💰 현재 잔액 확인 (Spot 거래):")
        account_info = trader.get_account_info()
        if 'balances' in account_info:
            # ETH와 USDT 잔액 확인
            for balance in account_info['balances']:
                asset = balance.get('asset', '')
                if asset in ['ETH', 'USDT']:
                    free = float(balance.get('free', 0))
                    locked = float(balance.get('locked', 0))
                    total = free + locked
                    if total > 0:
                        print(f"   {asset}: {total:.8f} (사용 가능: {free:.8f}, 잠김: {locked:.8f})")
            
    except Exception as e:
        print(f"❌ 포지션 정보 조회 실패: {e}")
        import traceback
        traceback.print_exc()


def test_small_order(trader: BinanceTrader):
    """5. 소액 주문 테스트 (최소 주문 금액 확인)"""
    print("\n" + "=" * 60)
    print("5. 소액 주문 테스트")
    print("=" * 60)
    
    try:
        current_price = trader.get_current_price("ETHUSDT")
        print(f"💹 현재 가격: {current_price:.2f} USDT")
        
        # 최소 주문 금액 테스트 (10 USDT)
        test_amount = 10.0
        print(f"\n📝 테스트: {test_amount} USDT로 시장가 주문 (ETH 매수)")
        
        # 수량 계산 테스트
        calculated_qty = trader._calculate_quantity("ETHUSDT", test_amount, current_price)
        print(f"   계산된 수량: {calculated_qty:.8f} ETH")
        
        result = trader.place_market_order(
            symbol="ETHUSDT",
            side="BUY",
            usdt_amount=test_amount
        )
        
        if result:
            print(f"✅ 주문 성공!")
            print(f"   주문 ID: {result.get('orderId')}")
            print(f"   상태: {result.get('status')}")
            executed_qty = result.get('executedQty', result.get('quantity', 'N/A'))
            print(f"   체결 수량: {executed_qty}")
            if isinstance(executed_qty, (int, float)):
                print(f"   체결 금액: {float(executed_qty) * current_price:.2f} USDT")
            print(f"   가격: {result.get('price', 'N/A')}")
        else:
            print("⚠️ 주문 실패 (결과 없음)")
            
    except Exception as e:
        print(f"❌ 소액 주문 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("거래 실행 기능 테스트 시작")
    print("=" * 60)
    print(f"현재 디렉토리: {os.getcwd()}")
    print(f".env 파일 경로: {env_path}")
    print(f".env 파일 존재: {env_path.exists()}")
    
    # API 키 확인
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    print(f"\nAPI 키 상태:")
    print(f"   BINANCE_API_KEY: {'있음' if api_key else '없음'} (길이: {len(api_key)})")
    print(f"   BINANCE_API_SECRET: {'있음' if api_secret else '없음'} (길이: {len(api_secret)})")
    
    if not api_key or not api_secret:
        print("\n❌ API 키가 설정되지 않았습니다.")
        print("   .env 파일에 BINANCE_API_KEY와 BINANCE_API_SECRET을 설정하세요.")
        return
    
    # 테스트 실행
    trader = test_api_connection()
    if not trader:
        print("\n❌ API 연결 실패로 테스트 중단")
        return
    
    # 심볼 정보 및 LOT_SIZE 필터 확인
    test_symbol_info(trader)
    
    executor = test_trade_executor(trader)
    if not executor:
        print("\n❌ TradeExecutor 초기화 실패로 테스트 중단")
        return
    
    # 포지션 정보 조회
    test_position_info(trader)
    
    # 사용자 확인
    print("\n" + "=" * 60)
    print("⚠️ 주의: 다음 테스트는 실제 주문을 실행합니다 (Demo Trading)")
    print("=" * 60)
    response = input("\n주문 실행 테스트를 진행하시겠습니까? (y/N): ")
    
    if response.lower() == 'y':
        # 소액 주문 테스트
        test_small_order(trader)
        
        # 거래 결정 실행 테스트
        test_decision_execution(executor)
        
        # 최종 포지션 정보 확인
        test_position_info(trader)
    else:
        print("\n주문 실행 테스트를 건너뜁니다.")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 테스트가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
