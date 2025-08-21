# 🚀 실시간 청산 데이터 수집 시스템

Binance WebSocket을 통해 실시간으로 청산 데이터를 수집하고, SQLite 데이터베이스에 저장하며, 다양한 분석 도구를 제공하는 시스템입니다.

## 📋 주요 기능

### 🔄 실시간 데이터 수집
- **Binance WebSocket**: 실시간 청산 이벤트 스트림 연결
- **다중 심볼 지원**: BTCUSDT, ETHUSDT, BNBUSDT 등 주요 코인
- **자동 재연결**: 연결 끊김 시 자동 재연결
- **버퍼링 시스템**: 효율적인 배치 데이터베이스 저장

### 🗄️ 데이터베이스 관리
- **SQLite 기반**: 가벼우면서도 강력한 로컬 데이터베이스
- **인덱싱**: 빠른 검색을 위한 최적화된 인덱스
- **자동 정리**: 오래된 데이터 자동 삭제
- **백업 지원**: 데이터베이스 백업 및 복원

### 📊 데이터 분석 및 시각화
- **트렌드 분석**: 시간별 청산량, 이벤트 수, 가격 변화
- **강도 분석**: 사이드별 청산 강도 및 통계
- **분포 분석**: 청산량, 가격, LPI 분포 히스토그램
- **종합 리포트**: JSON 형태의 상세 분석 리포트

## 🏗️ 시스템 구조

```
liquidation_system/
├── data/
│   ├── liquidation_database.py      # 데이터베이스 관리자
│   ├── liquidation_collector.py     # 실시간 데이터 수집기
│   ├── liquidation_analyzer.py      # 데이터 분석 및 시각화
│   └── binance_client.py           # Binance API 클라이언트
├── run_liquidation_collector.py     # 메인 실행 스크립트
├── logs/                           # 로그 파일
├── charts/                         # 생성된 차트
└── data/liquidations.db            # SQLite 데이터베이스
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
# 필요한 패키지 설치
pip install websockets pandas numpy matplotlib seaborn sqlite3
```

### 2. 모의 데이터로 테스트

```bash
# 30초 동안 모의 데이터 수집 후 분석
python run_liquidation_collector.py --mock --test-duration 30
```

### 3. 실제 데이터 수집 시작

```bash
# 실제 Binance 청산 데이터 수집
python run_liquidation_collector.py --collect
```

### 4. 데이터 분석

```bash
# 기본 리포트 생성
python run_liquidation_collector.py --analyze

# 모든 분석 실행 (차트 + 리포트)
python run_liquidation_collector.py --analyze --all

# 특정 심볼 분석
python run_liquidation_collector.py --analyze --symbol BTCUSDT --hours 48
```

## 📖 상세 사용법

### 🔍 데이터베이스 상태 확인

```bash
python run_liquidation_collector.py --status
```

출력 예시:
```
============================================================
청산 데이터베이스 상태
============================================================
총 이벤트 수: 1,234
최근 24시간: 567
최근 1시간: 23
데이터베이스 크기: 2.45 MB

🔸 심볼별 이벤트 수:
  BTCUSDT: 456
  ETHUSDT: 234
  BNBUSDT: 123

🔸 사이드별 이벤트 수:
  BUY (숏 청산): 678
  SELL (롱 청산): 556
============================================================
```

### 📊 분석 옵션

#### 트렌드 분석
```bash
python run_liquidation_collector.py --analyze --trends --symbol BTCUSDT --hours 24
```

#### 강도 분석
```bash
python run_liquidation_collector.py --analyze --intensity --symbol ETHUSDT --hours 48
```

#### 분포 분석
```bash
python run_liquidation_collector.py --analyze --distribution --symbol BNBUSDT --hours 12
```

#### 종합 리포트
```bash
python run_liquidation_collector.py --analyze --report --symbol BTCUSDT --hours 24
```

### ⚙️ 고급 설정

#### 사용자 정의 심볼
```bash
python run_liquidation_collector.py --collect --symbols BTCUSDT ETHUSDT ADAUSDT
```

#### 사용자 정의 데이터베이스 경로
```bash
python run_liquidation_collector.py --collect --db-path /custom/path/liquidations.db
```

#### 모의 데이터 수집 (개발/테스트용)
```bash
python run_liquidation_collector.py --mock --test-duration 60
```

## 🗄️ 데이터베이스 스키마

### liquidation_events (청산 이벤트)
```sql
CREATE TABLE liquidation_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,           -- 심볼 (예: BTCUSDT)
    side TEXT NOT NULL,             -- 사이드 (BUY: 숏 청산, SELL: 롱 청산)
    size REAL NOT NULL,             -- 청산 수량
    price REAL NOT NULL,            -- 청산 가격
    lpi REAL,                       -- Liquidation Price Index
    timestamp DATETIME NOT NULL,    -- 청산 발생 시간
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### liquidation_aggregates (1분 단위 집계)
```sql
CREATE TABLE liquidation_aggregates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    total_size REAL NOT NULL,       -- 총 청산량
    avg_price REAL NOT NULL,        -- 평균 가격
    avg_lpi REAL,                   -- 평균 LPI
    event_count INTEGER NOT NULL,   -- 이벤트 수
    period_start DATETIME NOT NULL, -- 집계 시작 시간
    period_end DATETIME NOT NULL    -- 집계 종료 시간
);
```

### liquidation_intensity (1시간 단위 강도)
```sql
CREATE TABLE liquidation_intensity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    total_size REAL NOT NULL,       -- 총 청산량
    avg_lpi REAL,                   -- 평균 LPI
    event_count INTEGER NOT NULL,   -- 이벤트 수
    period_start DATETIME NOT NULL, -- 집계 시작 시간
    period_end DATETIME NOT NULL    -- 집계 종료 시간
);
```

## 📈 분석 결과 예시

### 트렌드 차트
- **시간별 청산량**: 각 시간대별 롱/숏 청산량 변화
- **시간별 이벤트 수**: 청산 이벤트 발생 빈도
- **시간별 평균 가격**: 청산 가격의 시간대별 변화
- **시간별 평균 LPI**: 청산 강도의 시간대별 변화

### 강도 차트
- **사이드별 총 청산량**: 롱/숏 청산량 비교
- **사이드별 이벤트 수**: 롱/숏 청산 이벤트 수 비교
- **사이드별 평균 가격**: 롱/숏 청산 가격 비교
- **사이드별 평균 LPI**: 롱/숏 청산 강도 비교

### 분포 차트
- **청산량 분포**: 히스토그램으로 청산량 분포 시각화
- **가격 분포**: 청산 가격의 분포 패턴
- **LPI 분포**: 청산 강도의 분포 패턴
- **시간대별 분포**: 박스플롯으로 시간대별 청산량 분포

## 🔧 고급 기능

### 자동 데이터 정리
```python
from data.liquidation_database import LiquidationDatabase

db = LiquidationDatabase()
# 30일 이상 된 데이터 자동 삭제
deleted_count = db.cleanup_old_data(days=30)
```

### 사용자 정의 분석
```python
from data.liquidation_analyzer import LiquidationAnalyzer

analyzer = LiquidationAnalyzer()

# 특정 기간 데이터 조회
recent_data = analyzer.db.get_recent_liquidations(
    symbol='BTCUSDT', 
    hours=48, 
    limit=1000
)

# 사용자 정의 차트 생성
# (코드 예시...)
```

### 실시간 모니터링
```python
from data.liquidation_collector import LiquidationCollector

collector = LiquidationCollector(['BTCUSDT', 'ETHUSDT'])

# 상태 모니터링
status = collector.get_status()
print(f"수집기 상태: {status}")
```

## 🚨 주의사항

### 1. API 제한
- Binance WebSocket 연결은 안정적이지만, 과도한 요청 시 제한될 수 있습니다
- 네트워크 불안정 시 자동 재연결을 시도합니다

### 2. 데이터 저장
- SQLite 데이터베이스는 로컬에 저장됩니다
- 정기적인 백업을 권장합니다
- 오래된 데이터는 자동으로 정리됩니다

### 3. 메모리 사용
- 대용량 데이터 처리 시 메모리 사용량을 모니터링하세요
- 필요시 데이터베이스 파티셔닝을 고려하세요

## 🐛 문제 해결

### WebSocket 연결 실패
```bash
# 로그 확인
tail -f logs/liquidation_collector.log

# 방화벽 설정 확인
# 네트워크 연결 상태 확인
```

### 데이터베이스 오류
```bash
# 데이터베이스 파일 권한 확인
ls -la data/liquidations.db

# 데이터베이스 무결성 검사
sqlite3 data/liquidations.db "PRAGMA integrity_check;"
```

### 차트 생성 실패
```bash
# matplotlib 백엔드 확인
python -c "import matplotlib; print(matplotlib.get_backend())"

# 필요한 패키지 설치 확인
pip install matplotlib seaborn
```

## 📞 지원 및 기여

### 이슈 리포트
- GitHub Issues를 통해 버그 리포트
- 기능 요청 및 개선 제안

### 기여 방법
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🙏 감사의 말

- Binance API 팀
- SQLite 개발팀
- Python 커뮤니티

---

**⚠️ 투자 경고**: 이 도구는 교육 및 연구 목적으로만 사용되어야 합니다. 실제 거래에 사용하기 전에 충분한 테스트와 검증을 거쳐야 합니다.
