# 🚀 Crypto Edge - 통합 스마트 자동 트레이더

**암호화폐 시장의 청산 데이터와 기술적 분석을 통합한 고성능 자동 트레이딩 시스템**

## ✨ **주요 특징**

### 🔥 **청산 기반 분석**
- **실시간 청산 데이터** 수집 및 분석
- **청산 밀도** 기반 가격 예측
- **연쇄 청산** 이벤트 감지
- **시간대별 청산 패턴** 분석

### 📊 **하이브리드 전략**
- **5분봉 + 15분봉** 기술적 분석
- **VPVR (Volume Profile Visible Range)** 레벨 활용
- **EMA, RSI, 볼린저 밴드** 등 다중 지표 통합
- **스캘핑 최적화** (0.1% 가격 변동 감지)

### 🎯 **시너지 신호**
- **여러 전략의 신호를 우선순위로 통합**
- **신뢰도 기반 필터링** (설정 가능)
- **리스크-리워드 비율** 자동 계산
- **포지션 관리** 자동화

### ⚡ **실시간 성능**
- **WebSocket 기반** 실시간 데이터 처리
- **API 제한 보호** (분당 1200회 제한)
- **10초마다** 기술적 분석
- **30초마다** 거래량 급증 요약

## 🏗️ **프로젝트 구조**

```
crypto-edge/
├── 📁 config/
│   └── integrated_config.py          # 통합 전략 설정
├── 📁 data/
│   ├── binance_client.py             # 바이낸스 API 클라이언트
│   ├── binance_websocket.py          # 실시간 웹소켓 연결
│   └── loader.py                     # 데이터 로딩 유틸리티
├── 📁 indicators/
│   ├── atr.py                        # ATR (Average True Range)
│   ├── bollinger.py                  # 볼린저 밴드
│   ├── macd.py                       # MACD
│   ├── moving_averages.py            # 이동평균선
│   ├── stoch_rsi.py                  # Stochastic RSI
│   └── vpvr.py                       # VPVR (Volume Profile)
├── 📁 signals/
│   ├── hybrid_strategy.py            # 하이브리드 전략 (5분봉 기반)
│   ├── integrated_strategy.py        # 통합 전략 관리
│   ├── liquidation_prediction.py     # 청산 예측 전략
│   ├── liquidation_strategy.py       # 청산 기반 신호 전략
│   └── timing_strategy.py            # 포지션 타이밍 전략
├── integrated_smart_trader.py        # 🎯 메인 트레이더 (리팩토링 완료)
├── requirements.txt                   # Python 의존성
└── README.md                         # 프로젝트 문서
```

## 🚀 **빠른 시작**

### **1. 환경 설정**
```bash
# Python 3.8+ 설치 필요
pip install -r requirements.txt
```

### **2. 설정 파일 수정**
```python
# config/integrated_config.py
class IntegratedConfig:
    symbol: str = "ETHUSDC"                    # 거래 심볼
    enable_hybrid_strategy: bool = True        # 하이브리드 전략 활성화
    enable_liquidation_strategy: bool = True   # 청산 전략 활성화
    hybrid_min_confidence: float = 0.5       # 최소 신뢰도 (45%)
    liquidation_min_confidence: float = 0.6    # 청산 신호 최소 신뢰도
```

### **3. 실행**
```bash
python integrated_smart_trader.py
```

## 🔧 **설정 옵션**

### **📊 하이브리드 전략**
- `hybrid_interval_15m`: 15분봉 간격
- `hybrid_interval_5m`: 5분봉 간격
- `hybrid_min_confidence`: 최소 신뢰도 (0.0 ~ 1.0)
- `hybrid_trend_weight`: 트렌드 가중치
- `hybrid_entry_weight`: 진입점 가중치

### **🔥 청산 전략**
- `liquidation_min_count`: 최소 청산 개수
- `liquidation_min_value`: 최소 청산 가치
- `liquidation_window_minutes`: 분석 윈도우
- `liquidation_min_confidence`: 최소 신뢰도

### **🔮 예측 전략**
- `prediction_min_density`: 최소 밀도
- `prediction_cascade_threshold`: 연쇄 청산 임계값
- `prediction_min_confidence`: 최소 예측 신뢰도
- `max_prediction_horizon_hours`: 최대 예측 시간

### **⏰ 타이밍 전략**
- `timing_max_hold_time_hours`: 최대 보유 시간
- `timing_trailing_stop`: 트레일링 스탑 활성화
- `timing_take_profit_levels`: 익절 레벨 수

## 📈 **신호 유형**

### **🎯 HYBRID 신호**
- **5분봉 + 15분봉** 기술적 분석 기반
- **VPVR 레벨** 근처에서의 진입점
- **EMA, RSI, 볼린저 밴드** 통합 분석

### **🔥 LIQUIDATION 신호**
- **실시간 청산 데이터** 기반
- **청산 밀도** 높은 가격대 감지
- **롱/숏 청산 비율** 분석

### **🔮 PREDICTION 신호**
- **청산 패턴** 기반 미래 예측
- **연쇄 청산** 이벤트 감지
- **시간대별 패턴** 분석

### **⚡ SYNERGY 신호**
- **여러 전략의 신호가 일치**할 때
- **높은 신뢰도**와 **좋은 리스크-리워드**
- **우선순위가 높은** 신호

## 🛡️ **리스크 관리**

### **📊 포지션 크기**
- **ATR 기반** 손절가 계산
- **동적 익절** 레벨 설정
- **트레일링 스탑** 자동 조정

### **⏰ 시간 관리**
- **최대 보유 시간** 제한
- **스캘핑 최적화** (단기 거래)
- **시장 상황** 기반 자동 조정

### **🔒 API 보호**
- **분당 호출 제한** (1200회)
- **자동 재시도** 메커니즘
- **에러 처리** 및 복구

## 📊 **성능 모니터링**

### **📈 실시간 통계**
- **신호 발생 횟수** 및 **성공률**
- **시너지 신호** 발생 빈도
- **API 호출** 통계

### **🔥 청산 분석**
- **청산 밀도** 분포
- **가격별 청산** 패턴
- **시간대별** 청산 트렌드

### **💹 포지션 관리**
- **활성 포지션** 상태
- **손익 현황** 실시간 업데이트
- **리스크 지표** 모니터링

## 🔄 **최근 업데이트**

### **✨ v2.0 - 리팩토링 완료**
- **코드 구조** 대폭 개선
- **메서드 그룹화** 및 가독성 향상
- **에러 처리** 강화
- **성능 최적화** 적용

### **🚀 v1.5 - 청산 예측 통합**
- **liquidation_prediction.py** 통합
- **liquidation_strategy.py** 통합
- **폭등/폭락 예측** 기능 추가

### **📊 v1.0 - 기본 기능**
- **하이브리드 전략** 구현
- **실시간 청산 분석** 구현
- **웹소켓 기반** 데이터 수집

## 🤝 **기여하기**

1. **Fork** 프로젝트
2. **Feature branch** 생성 (`git checkout -b feature/AmazingFeature`)
3. **Commit** 변경사항 (`git commit -m 'Add some AmazingFeature'`)
4. **Push** 브랜치 (`git push origin feature/AmazingFeature`)
5. **Pull Request** 생성

## 📝 **라이선스**

이 프로젝트는 **MIT 라이선스** 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## ⚠️ **면책 조항**

**이 소프트웨어는 교육 및 연구 목적으로만 제공됩니다. 실제 거래에 사용할 경우 발생하는 모든 손실에 대해 개발자는 책임지지 않습니다. 암호화폐 거래는 높은 위험을 수반하므로 신중하게 접근하시기 바랍니다.**

---

**🚀 Crypto Edge로 암호화폐 시장의 엣지를 찾아보세요!**