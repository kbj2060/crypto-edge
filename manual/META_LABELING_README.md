# 메타 라벨링 (Meta-Labeling) 구현

마르코스 로페즈 데 프라도(Marcos López de Prado)가 제안한 메타 라벨링 기법을 구현했습니다.

## 개요

메타 라벨링은 2단계 접근법입니다:

1. **1단계: 방향 예측** (기존 `TradeDecisionEngine`)
   - 매수/매도/보유 결정

2. **2단계: 메타 라벨링** (새로 추가된 `MetaLabelingEngine`)
   - 방향 예측이 맞는지 여부를 예측
   - 거래 실행 여부 결정

## 구조

### 파일 구조

```
engines/
├── meta_labeling_engine.py    # 메타 라벨링 엔진
├── trade_decision_engine.py    # 거래 결정 엔진 (메타 라벨링 통합됨)
└── META_LABELING_README.md     # 이 문서

tools/
└── train_meta_labeling.py     # 모델 학습 스크립트
```

## 사용 방법

### 1. 모델 학습

과거 거래 결정 데이터와 가격 데이터를 사용하여 모델을 학습합니다:

```bash
python tools/train_meta_labeling.py
```

이 스크립트는:
- `agent/decisions_data.parquet`에서 결정 데이터 로드
- Binance API에서 가격 데이터 로드
- 메타 라벨 생성 (방향 예측이 맞았는지 여부)
- Random Forest 모델 학습
- 모델을 `engines/meta_labeling_model.pkl`에 저장

### 2. 자동 사용

`TradeDecisionEngine`은 기본적으로 메타 라벨링을 사용합니다:

```python
from engines.trade_decision_engine import TradeDecisionEngine

# 메타 라벨링 활성화 (기본값)
engine = TradeDecisionEngine(use_meta_labeling=True)

# 메타 라벨링 비활성화
engine = TradeDecisionEngine(use_meta_labeling=False)
```

### 3. 수동 사용

메타 라벨링 엔진을 직접 사용할 수도 있습니다:

```python
from engines.meta_labeling_engine import MetaLabelingEngine

engine = MetaLabelingEngine()
engine.load_model()  # 저장된 모델 로드

# 결정에 메타 라벨링 적용
decision = {
    "action": "LONG",
    "net_score": 0.75,
    "meta": {
        "synergy_meta": {
            "confidence": "HIGH",
            "buy_score": 0.8,
            "sell_score": 0.2
        }
    }
}

market_data = {
    "atr": 50.0,
    "volume": 1000000.0,
    "volatility": 0.02
}

result = engine.predict(decision, market_data)
print(result)
# {
#     "should_execute": True,
#     "prediction": 1,
#     "probability": 0.85,
#     "confidence": "HIGH"
# }
```

## 작동 원리

### 특성 추출

메타 라벨링 모델은 다음 특성들을 사용합니다:

1. **결정 관련**
   - Action (LONG/SHORT/HOLD)
   - Net Score
   - 절대값 Net Score

2. **신뢰도 관련**
   - Confidence (HIGH/MEDIUM/LOW)
   - 전략 사용 수

3. **시너지 메타**
   - Buy Score
   - Sell Score
   - Signals Used
   - 점수 차이

4. **포지션 크기**
   - Risk USD
   - Leverage

5. **카테고리**
   - SHORT_TERM / MEDIUM_TERM / LONG_TERM

6. **시장 데이터**
   - ATR
   - Volume
   - Volatility

### 메타 라벨 생성

과거 결정 데이터에서 메타 라벨을 생성할 때:

1. 결정 시점의 가격 확인
2. 미래 N 기간 후의 가격 확인 (기본값: 20 기간)
3. 방향 예측이 맞았는지 확인:
   - LONG → 가격 상승이면 1, 아니면 0
   - SHORT → 가격 하락이면 1, 아니면 0
   - HOLD → 0 (거래하지 않음)

### 예측 및 실행

실시간 거래 결정 시:

1. `TradeDecisionEngine`이 방향 예측 (LONG/SHORT/HOLD)
2. `MetaLabelingEngine`이 거래 실행 여부 예측
3. `should_execute=False`이면 HOLD로 변경

## 설정

### MetaLabelingEngine 파라미터

```python
MetaLabelingEngine(
    model_type="random_forest",      # "random_forest" or "gradient_boosting"
    min_samples_for_training=100,    # 최소 학습 샘플 수
    confidence_threshold=0.6,         # 거래 실행 최소 신뢰도
    model_save_path="engines/meta_labeling_model.pkl"
)
```

### 모델 타입

- **Random Forest** (기본값): 빠르고 안정적
- **Gradient Boosting**: 더 정확할 수 있지만 느림

## 모델 저장/로드

모델은 자동으로 저장/로드됩니다:

- **저장**: 학습 완료 시 자동 저장
- **로드**: `TradeDecisionEngine` 초기화 시 자동 로드

수동으로 저장/로드:

```python
engine.save_model("path/to/model.pkl")
engine.load_model("path/to/model.pkl")
```

## 모델이 학습되지 않은 경우

모델이 학습되지 않았거나 로드에 실패하면, 기본 휴리스틱을 사용합니다:

- 높은 점수 (|net_score| > 0.3)
- 중간 이상 신뢰도 (confidence >= MEDIUM)

이 경우 거래 실행 여부를 결정합니다.

## 성능 평가

학습 완료 시 다음 메트릭을 제공합니다:

- **정확도 (Accuracy)**: 전체 예측 정확도
- **ROC-AUC**: 이진 분류 성능
- **Precision**: 거래 실행 권장 시 실제로 맞을 비율
- **Recall**: 실제로 거래해야 할 때 찾아낸 비율
- **특성 중요도**: 어떤 특성이 가장 중요한지

## 주의사항

1. **최소 학습 데이터**: 최소 100개의 거래 결정이 필요합니다
2. **데이터 품질**: 과거 결정 데이터와 가격 데이터가 정확해야 합니다
3. **재학습**: 시장 환경이 변하면 주기적으로 재학습하는 것이 좋습니다
4. **백테스팅**: 실제 거래 전에 백테스팅으로 검증하세요

## 향후 개선 사항

- [ ] 더 많은 특성 추가 (시장 상황, 시간대 등)
- [ ] 앙상블 모델 사용
- [ ] 온라인 학습 (점진적 업데이트)
- [ ] 카테고리별 별도 모델
- [ ] 성능 모니터링 및 자동 재학습


