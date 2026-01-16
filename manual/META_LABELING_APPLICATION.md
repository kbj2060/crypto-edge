# 메타 라벨링 적용 가이드

## 현재 상태

✅ **모델이 이미 자동으로 적용되고 있습니다!**

`TradeDecisionEngine`이 초기화될 때 자동으로 메타 라벨링 모델을 로드하고, 모든 거래 결정에 적용합니다.

## 작동 방식

1. **자동 적용**: `binance_websocket.py`에서 `decision_engine.decide_trade_realtime(signals)` 호출 시 자동 적용
2. **필터링**: 메타 라벨링이 거래 실행을 권장하지 않으면 (`should_execute=False`) HOLD로 변경
3. **메타데이터 저장**: 결정의 `meta.meta_labeling`에 예측 결과 저장

## 현재 모델 성능

- **정확도**: 39.9%
- **ROC-AUC**: 56.2%
- **Precision**: 17.7% ⚠️ (낮음 - 거래 실행 권장 시 실제로 맞을 확률)
- **Recall**: 78.7% (높음 - 실제로 거래해야 할 때 많이 찾아냄)

### 현재 설정: 보수적 필터링 모드

- **confidence_threshold**: 0.7 (70%)
- **목적**: Precision 향상 (거래 실행 권장 시 실제로 맞을 확률 증가)
- **효과**: 더 엄격한 기준으로 거래를 필터링하여 품질 향상

### 성능 해석

- **Precision이 낮음**: 거래 실행을 권장했을 때 실제로 수익이 날 확률이 낮음
- **Recall이 높음**: 실제로 거래해야 할 기회를 많이 찾아냄
- **결과**: 보수적인 필터링 (많은 거래를 차단하여 손실 거래를 줄임)
- **보수적 모드**: 70% 이상 확률일 때만 거래 실행 (기본 60%보다 엄격)

## 모니터링 방법

### 1. 테스트 스크립트 실행

```bash
python tools/test_meta_labeling.py
```

샘플 결정에 대한 메타 라벨링 예측 결과를 확인할 수 있습니다.

### 2. 실시간 로그 확인

실제 거래 시스템에서 메타 라벨링이 적용되면:
- 결정의 `reason` 필드에 메타 라벨링 정보 표시
- `meta.meta_labeling`에 상세 예측 결과 저장

### 3. 결정 로그 분석

`logs/decisions_*.parquet` 파일에서 메타 라벨링 결과를 확인:

```python
import pandas as pd

df = pd.read_parquet("logs/decisions_20250116.parquet")
# meta_labeling 정보 확인
```

## 성능 개선 방안

### 1. 임계값 조정

현재 `confidence_threshold=0.6` (60%)로 설정되어 있습니다.

**더 보수적으로** (Precision 향상):
```python
engine = MetaLabelingEngine(confidence_threshold=0.7)  # 70%
```

**더 공격적으로** (Recall 향상):
```python
engine = MetaLabelingEngine(confidence_threshold=0.5)  # 50%
```

### 2. 최소 수익률 임계값 조정

학습 시 `min_profit_threshold`를 조정:

```python
# tools/train_meta_labeling.py에서
result = engine.train(
    ...
    min_profit_threshold=0.01,  # 1%로 증가 (더 엄격)
    # 또는
    min_profit_threshold=0.003,  # 0.3%로 감소 (더 관대)
)
```

### 3. 모델 재학습

더 나은 특성이나 하이퍼파라미터로 재학습:

```python
# Gradient Boosting 시도
engine = MetaLabelingEngine(model_type="gradient_boosting")
```

### 4. 메타 라벨링 비활성화 (비교용)

성능 비교를 위해 일시적으로 비활성화:

```python
decision_engine = TradeDecisionEngine(use_meta_labeling=False)
```

## 권장 설정

현재 모델의 특성상:

1. **보수적 사용**: `confidence_threshold=0.7` 이상
   - Precision 향상 (거래 실행 권장 시 실제로 맞을 확률 증가)
   - 거래 빈도 감소하지만 품질 향상

2. **모니터링 강화**: 
   - 메타 라벨링이 차단한 거래의 실제 결과 추적
   - 차단된 거래 중 실제로 수익이 났던 비율 확인

3. **점진적 적용**:
   - 먼저 모니터링 모드로 실행
   - 실제 성능 확인 후 점진적으로 적용

## 다음 단계

1. ✅ 모델 학습 완료
2. ✅ 자동 적용 확인
3. 🔄 실시간 모니터링
4. 🔄 성능 개선 (필요시)

## 문제 해결

### 모델이 로드되지 않음
```bash
# 모델 파일 확인
ls -lh engines/meta_labeling_model.pkl

# 재학습
python tools/train_meta_labeling.py
```

### 메타 라벨링이 작동하지 않음
- `TradeDecisionEngine` 초기화 시 `use_meta_labeling=True` 확인
- 모델 파일이 올바른 위치에 있는지 확인

