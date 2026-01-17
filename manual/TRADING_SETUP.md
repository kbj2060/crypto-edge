# 바이낸스 실제 거래 설정 가이드

## 개요

이 시스템은 바이낸스 선물 API를 사용하여 실제 거래를 실행할 수 있습니다. 안전을 위해 기본적으로 **시뮬레이션 모드**로 실행되며, 실제 거래를 사용하려면 명시적으로 활성화해야 합니다.

## 주요 기능

1. **BinanceTrader**: 바이낸스 API와 통신하는 클라이언트
   - 계좌 정보 조회
   - 시장가/지정가 주문 실행
   - 포지션 조회 및 관리
   - 레버리지 및 마진 타입 설정

2. **TradeExecutor**: 거래 결정을 실제 주문으로 변환
   - TradeDecisionEngine의 결정을 주문으로 변환
   - 포지션 크기 계산 및 리스크 관리
   - 자동 포지션 청산 (반대 포지션 개설 시)
   - 거래 로그 저장

3. **BinanceWebSocket 통합**: 실시간 결정에 따른 자동 거래 실행

## 설정 방법

### 1. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 입력하세요:

```bash
# 바이낸스 API 키 (선물 거래 권한 필요)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# 거래 설정
ENABLE_TRADING=false          # true로 설정하면 거래 활성화
SIMULATION_MODE=true          # false로 설정하면 실제 거래
USE_TESTNET=false             # true로 설정하면 테스트넷 사용
```

### 2. 바이낸스 API 키 생성

1. 바이낸스 계정에 로그인
2. API 관리 페이지로 이동
3. 새 API 키 생성
4. **반드시 "선물 거래" 권한을 활성화**하세요
5. IP 제한 설정 (보안 강화)

⚠️ **주의**: API 키는 절대 공개 저장소에 커밋하지 마세요!

### 3. 코드에서 거래 활성화

#### 방법 1: BinanceWebSocket 직접 사용

```python
from managers.binance_websocket import BinanceWebSocket

# 시뮬레이션 모드
websocket = BinanceWebSocket(
    symbol="ETHUSDT",
    enable_trading=True,
    simulation_mode=True  # 실제 거래는 False
)

# 실제 거래 모드 (주의!)
websocket = BinanceWebSocket(
    symbol="ETHUSDT",
    enable_trading=True,
    simulation_mode=False  # 실제 자금이 사용됩니다!
)
```

#### 방법 2: 환경 변수 사용

```python
import os
from managers.binance_websocket import BinanceWebSocket

enable_trading = os.getenv("ENABLE_TRADING", "false").lower() == "true"
simulation_mode = os.getenv("SIMULATION_MODE", "true").lower() == "true"

websocket = BinanceWebSocket(
    symbol="ETHUSDT",
    enable_trading=enable_trading,
    simulation_mode=simulation_mode
)
```

## 리스크 관리

### 포지션 크기 제한

기본적으로 다음 제한이 적용됩니다:

- **최대 포지션 크기**: 1000 USDT (TradeExecutor의 `max_position_size_usdt`로 변경 가능)
- **최소 주문 금액**: 10 USDT
- **기본 레버리지**: 10x
- **마진 타입**: ISOLATED (격리 마진)

### 포지션 크기 계산 로직

포지션 크기는 다음 요소에 따라 결정됩니다:

1. **net_score**: 결정의 신호 강도 (0.0 ~ 1.0)
2. **confidence**: 신뢰도 (VERY_HIGH, HIGH, MEDIUM, LOW)
3. **계좌 잔액**: 사용 가능한 잔액
4. **기본 리스크 비율**: 계좌의 2%

최종 포지션 크기 = 계좌 잔액 × 2% × net_score × confidence_가중치

## 안전 기능

### 1. 시뮬레이션 모드

기본적으로 시뮬레이션 모드가 활성화되어 있어 실제 주문이 실행되지 않습니다. 시뮬레이션 모드에서는:

- 모든 API 호출이 시뮬레이션됨
- 실제 자금이 사용되지 않음
- 주문 결과가 시뮬레이션 데이터로 반환됨

### 2. 메타 라벨링 필터

거래 결정이 나도 메타 라벨링 모델이 승인하지 않으면 거래가 실행되지 않습니다:

```python
# 메타 라벨링 결과 확인
meta_labeling = final_decision.get("meta", {}).get("meta_labeling", {})
should_execute = meta_labeling.get("should_execute", False)

if should_execute:
    # 거래 실행
    trade_executor.execute_decision(final_decision)
else:
    # 거래 차단
    print("메타 라벨링에 의해 거래 차단")
```

### 3. 자동 포지션 청산

반대 방향의 포지션을 개설하려고 하면 기존 포지션이 자동으로 청산됩니다:

- LONG 포지션이 있을 때 SHORT 신호 → LONG 청산 후 SHORT 개설
- SHORT 포지션이 있을 때 LONG 신호 → SHORT 청산 후 LONG 개설

## 테스트넷 사용

실제 자금 없이 테스트하려면 테스트넷을 사용하세요:

1. 바이낸스 테스트넷 계정 생성
2. 테스트넷 API 키 생성
3. `BinanceTrader` 초기화 시 `testnet=True` 설정

```python
from managers.binance_trader import BinanceTrader

trader = BinanceTrader(
    api_key="testnet_api_key",
    api_secret="testnet_api_secret",
    testnet=True,
    simulation_mode=False  # 테스트넷이므로 실제 거래 가능
)
```

## 모니터링

### 거래 로그

모든 거래는 `logs/trades_YYYYMMDD.jsonl` 파일에 저장됩니다:

```json
{
  "timestamp": "2026-01-16T12:00:00+00:00",
  "symbol": "ETHUSDT",
  "action": "LONG",
  "net_score": 0.75,
  "confidence": "HIGH",
  "order_id": 123456789,
  "quantity": 0.1,
  "price": 2500.0,
  "simulation": true
}
```

### 텔레그램 알림

거래 실행 시 텔레그램으로 알림이 전송됩니다 (설정된 경우).

## 주의사항

⚠️ **중요**: 실제 거래를 사용하기 전에 반드시 다음을 확인하세요:

1. ✅ API 키가 올바르게 설정되었는지
2. ✅ 시뮬레이션 모드에서 충분히 테스트했는지
3. ✅ 리스크 관리 설정이 적절한지
4. ✅ 최대 포지션 크기가 계좌 잔액에 비해 적절한지
5. ✅ 메타 라벨링 모델이 올바르게 학습되었는지

## 문제 해결

### "API 요청 실패" 오류

- API 키와 시크릿이 올바른지 확인
- IP 제한이 설정되어 있다면 현재 IP가 허용되었는지 확인
- API 키에 선물 거래 권한이 있는지 확인

### "잔액 부족" 경고

- 계좌에 충분한 USDT가 있는지 확인
- 최소 주문 금액(10 USDT) 이상인지 확인

### 거래가 실행되지 않음

- `enable_trading=True`로 설정되었는지 확인
- 메타 라벨링이 거래를 차단했는지 확인 (`should_execute=False`)
- 시뮬레이션 모드인지 확인 (시뮬레이션 모드에서는 실제 주문이 실행되지 않음)

## 예제 코드

```python
#!/usr/bin/env python3
"""실제 거래 예제 (주의: 실제 자금이 사용됩니다!)"""

import os
from managers.binance_websocket import BinanceWebSocket

# 환경 변수에서 설정 읽기
enable_trading = os.getenv("ENABLE_TRADING", "false").lower() == "true"
simulation_mode = os.getenv("SIMULATION_MODE", "true").lower() == "true"

# 웹소켓 초기화
websocket = BinanceWebSocket(
    symbol="ETHUSDT",
    enable_trading=enable_trading,
    simulation_mode=simulation_mode
)

# 웹소켓 시작
websocket.start_background()

# 메인 루프 (Ctrl+C로 종료)
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n프로그램 종료 중...")
    websocket.stop()
```

