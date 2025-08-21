# 🚀 청산 데이터 수집 시스템

**독립적인 Binance 실시간 청산 데이터 수집 시스템**

## 📋 개요

이 시스템은 Binance WebSocket을 통해 실시간으로 청산 데이터를 수집하고 SQLite 데이터베이스에 저장하는 간단하고 효율적인 시스템입니다.

## 🏗️ 시스템 구조

```
liquidation_system/
├── 📁 data/                           # 핵심 모듈
│   ├── 🔧 liquidation_database.py      # 데이터베이스 관리자
│   ├── 🌐 liquidation_collector.py     # 실시간 데이터 수집기
│   ├── 🔗 binance_client.py           # Binance API 클라이언트
│   └── __init__.py                    # 패키지 초기화
├── 🚀 run_liquidation_collector.py     # 메인 실행 스크립트
├── 🎯 run.py                          # 독립 실행 스크립트
├── 📋 requirements.txt                 # 의존성 패키지
├── 📖 README.md                        # 이 파일
└── 🗄️ data/liquidations.db            # SQLite 데이터베이스
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
cd liquidation_system
pip install -r requirements.txt
```

### 2. 시스템 테스트

```bash
# 모의 데이터로 테스트
python run.py --mock
```

### 3. 실제 데이터 수집

```bash
# 실제 Binance 청산 데이터 수집
python run.py --collect
```

### 4. 상태 확인

```bash
# 데이터베이스 상태 확인
python run.py --status
```

## 📊 주요 기능

### 🔄 **실시간 데이터 수집**
- Binance WebSocket 실시간 청산 이벤트 스트림
- ETHUSDT 전용 청산 데이터 수집
- 자동 재연결 및 버퍼링 시스템

### 🗄️ **데이터베이스 관리**
- SQLite 기반 효율적인 로컬 저장소
- 인덱싱된 빠른 검색
- 자동 데이터 정리

## 🛠️ 사용법

### 기본 명령어

```bash
# 상태 확인
python run.py --status

# 모의 데이터 수집
python run.py --mock

# 실제 데이터 수집
python run.py --collect
```

### 고급 옵션

```bash
# 특정 심볼만 수집 (기본값: ETHUSDT)
python run.py --collect --symbols ETHUSDT

# 커스텀 데이터베이스 경로
python run.py --collect --db-path /custom/path/liquidations.db
```

## 🔧 설정

### 환경 변수 (선택사항)

```bash
# Binance API 키 설정 (선택사항)
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

## 📊 데이터베이스 스키마

### liquidation_events (청산 이벤트)
- `symbol`: 심볼 (예: ETHUSDT)
- `side`: 사이드 (BUY: 숏 청산, SELL: 롱 청산)
- `size`: 청산 수량
- `price`: 청산 가격
- `lpi`: Liquidation Price Index
- `timestamp`: 청산 발생 시간

## 🚨 주의사항

1. **API 제한**: Binance API 사용량 제한을 준수하세요
2. **데이터 저장**: 정기적인 백업을 권장합니다
3. **메모리 사용**: 대용량 데이터 처리 시 메모리 사용량을 모니터링하세요

## 🐛 문제 해결

### 일반적인 문제들

```bash
# 데이터베이스 상태 확인
python run.py --status

# 연결 테스트
python -c "from data.binance_client import BinanceClient; BinanceClient().test_connection()"
```

## 📄 라이선스

MIT 라이선스 하에 배포됩니다.

## ⚠️ 투자 경고

이 도구는 교육 및 연구 목적으로만 사용되어야 합니다. 실제 거래에 사용하기 전에 충분한 테스트와 검증을 거쳐야 합니다.

---

**🚀 간단하고 효율적인 청산 데이터 수집 시스템이 준비되었습니다!**
