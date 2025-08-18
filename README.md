# 🚀 Crypto Edge - 통합 스마트 자동 트레이더

**ETHUSDT 전용 고성능 자동 트레이딩 시스템**

## 🌟 주요 특징

### 🔥 **통합 전략 시스템**
- **하이브리드 전략**: 15분봉 + 5분봉 다중 시간대 분석
- **실시간 청산 전략**: 웹소켓 기반 즉시 시장 변동성 감지
- **시너지 신호**: 두 전략이 일치할 때 신뢰도 극대화

### 📊 **실시간 데이터 수집**
- **바이낸스 웹소켓**: 실시간 청산, 거래량, 가격 데이터
- **즉시 반응**: 시장 변동 시 1초 내 신호 생성
- **연속 모니터링**: 24/7 시장 상황 추적

### 🎯 **고급 신호 생성**
- **다중 조건**: 기술적 지표 + 시장 심리 + 청산 패턴
- **신뢰도 기반**: 0.2~1.0 범위의 정확한 신뢰도 계산
- **리스크 관리**: ATR 기반 손절/익절 자동 설정

## 🚀 실행 방법

### 1. **기본 하이브리드 전략** (5분봉 기반)
```bash
uv run ./smart_auto_trader.py
```

### 2. **실시간 청산 전략** (웹소켓 기반)
```bash
uv run ./realtime_liquidation_trader.py
```

### 3. **🔥🔥🔥 통합 시너지 전략** (권장)
```bash
uv run ./integrated_smart_trader.py
```

## 📁 프로젝트 구조

```
crypto-edge/
├── 📊 data/                    # 데이터 처리
│   ├── binance_client.py      # 바이낸스 API 클라이언트
│   ├── binance_websocket.py   # 실시간 웹소켓 클라이언트
│   └── loader.py              # 데이터 로더
├── 📈 indicators/              # 기술적 지표
│   └── vpvr.py                # VPVR (Volume Profile)
├── 🎯 signals/                 # 전략 로직
│   ├── hybrid_strategy.py     # 하이브리드 전략
│   ├── liquidation_strategy.py # 청산 기반 전략
│   ├── integrated_strategy.py # 통합 전략
│   └── timing_strategy.py     # 타이밍 전략
├── ⚙️ config/                  # 설정 파일
│   └── integrated_config.py   # 통합 설정
├── 🚀 smart_auto_trader.py    # 기본 하이브리드 트레이더
├── 🔥 realtime_liquidation_trader.py # 실시간 청산 트레이더
├── 🎯 integrated_smart_trader.py     # 통합 시너지 트레이더
└── 📚 README.md               # 프로젝트 문서
```

## 🔥 시너지 효과

### **1. 하이브리드 전략 (5분봉)**
- **15분봉**: 장기 트렌드 방향성 파악
- **5분봉**: 단기 진입 타이밍 최적화
- **VPVR**: 주요 지지/저항 레벨 분석

### **2. 실시간 청산 전략**
- **청산 감지**: 대량 청산 발생 시 즉시 알림
- **거래량 급증**: 평균 대비 2배 이상 증가 감지
- **가격 변동성**: 0.5% 이상 변동 시 모니터링

### **3. 🎯 시너지 신호**
- **방향 일치**: 두 전략이 같은 방향 신호 생성
- **신뢰도 증가**: 평균 + 50% 보너스
- **R/R 향상**: 리스크/보상 비율 20% 증가

## 📊 신호 등급

### **🔥🔥🔥 SYNERGY 신호** (최고 등급)
- 하이브리드 + 청산 전략 일치
- 신뢰도 40% 이상
- 즉시 진입 권장

### **📈📉 개별 신호**
- **하이브리드**: 5분봉 기반 정기 신호
- **청산**: 실시간 시장 변동성 기반 신호

## ⚙️ 설정 옵션

### **전략 활성화/비활성화**
```python
config = IntegratedConfig()
config.enable_hybrid_strategy = True      # 하이브리드 전략
config.enable_liquidation_strategy = True # 청산 전략
config.enable_synergy_signals = True      # 시너지 신호
```

### **임계값 조정**
```python
# 청산 임계값
config.liquidation_min_count = 3         # 최소 청산 수
config.liquidation_min_value = 100000    # 최소 청산 가치 (USDT)
config.liquidation_volume_threshold = 2.0 # 거래량 급증 임계값

# 신호 임계값
config.hybrid_min_confidence = 0.20      # 하이브리드 최소 신뢰도
config.timing_entry_confidence_min = 0.20 # 타이밍 최소 신뢰도
```

## 💡 사용 시나리오

### **📈 강세장 (Bull Market)**
- 하이브리드: BULLISH 트렌드
- 청산: BUY 청산 급증
- **시너지**: BUY 신호 신뢰도 극대화

### **📉 약세장 (Bear Market)**
- 하이브리드: BEARISH 트렌드
- 청산: SELL 청산 급증
- **시너지**: SELL 신호 신뢰도 극대화

### **🔄 횡보장 (Sideways Market)**
- 하이브리드: NEUTRAL 트렌드
- 청산: 개별 청산 이벤트
- **개별 신호**: 각 전략의 독립적 신호

## 🚨 주의사항

1. **실시간 데이터**: 웹소켓 연결 상태 모니터링 필요
2. **신호 과다**: 쿨다운 시스템으로 과도한 신호 방지
3. **리스크 관리**: 항상 손절가 설정 필수
4. **백테스팅**: 실제 거래 전 충분한 테스트 권장

## 🔧 문제 해결

### **웹소켓 연결 오류**
```bash
# 패키지 재설치
uv add websocket-client websockets
```

### **신호 생성 안됨**
- 임계값 조정 필요
- 시장 상황에 맞는 파라미터 최적화

### **과도한 신호**
- 쿨다운 시간 증가
- 신뢰도 임계값 상향 조정

## 📈 성능 최적화

1. **멀티스레딩**: 하이브리드 분석과 웹소켓 동시 실행
2. **데이터 정리**: 오래된 데이터 자동 정리
3. **메모리 관리**: 최대 데이터 저장량 제한
4. **에러 처리**: 안정적인 재연결 및 복구

---

**🎯 이제 하이브리드 전략과 실시간 청산 전략의 시너지 효과를 활용하여 더욱 정확하고 빠른 거래 신호를 생성할 수 있습니다!** 🚀