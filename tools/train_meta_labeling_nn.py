#!/usr/bin/env python3
"""
딥러닝 기반 메타 라벨링 모델 학습 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime, timezone, timedelta
from engines.meta_labeling_nn import MetaLabelingNNEngine


def load_decision_data(file_path: str, years_back: int = 1) -> pd.DataFrame:
    """결정 데이터 로드"""
    print(f"📊 결정 데이터 로드 중... ({file_path})")
    
    df = pd.read_parquet(file_path)
    print(f"Decision 데이터 로드 완료: {file_path} ({len(df)}개 레코드)")
    
    # 시간 필터링
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=365 * years_back)
        df = df[df['timestamp'] >= cutoff_date]
        print(f"   필터링 후: {len(df)}개 (지금으로부터 {years_back}년)")
    
    return df


def load_price_data(decisions_df: pd.DataFrame, symbol: str = "ETHUSDT", years_back: int = 1) -> pd.DataFrame:
    """가격 데이터 로드 (기존 train_meta_labeling.py와 동일한 로직)"""
    # 기존 train_meta_labeling.py의 load_price_data 함수 사용
    from tools.train_meta_labeling import load_price_data as load_price_data_original
    
    return load_price_data_original(decisions_df, symbol, years_back)


def prepare_data(decisions_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple:
    """데이터 준비"""
    print("🔧 데이터 준비 중...")
    
    # 타임스탬프 정렬
    if 'timestamp' in decisions_df.columns:
        decisions_df['timestamp'] = pd.to_datetime(decisions_df['timestamp'], utc=True)
        decisions_df = decisions_df.sort_values('timestamp')
    
    if 'timestamp' in price_df.columns:
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)
        price_df = price_df.sort_values('timestamp')
        price_df = price_df.set_index('timestamp')
    
    print(f"✅ 데이터 준비 완료:")
    print(f"   결정 데이터: {len(decisions_df)}개")
    print(f"   가격 데이터: {len(price_df)}개")
    if 'timestamp' in decisions_df.columns:
        print(f"   시간 범위: {decisions_df['timestamp'].min()} ~ {decisions_df['timestamp'].max()}")
    
    return decisions_df, price_df


def main():
    """메인 함수"""
    print("=" * 60)
    print("딥러닝 메타 라벨링 모델 학습 시작 (1년치 데이터)")
    print("=" * 60)
    
    years_back = 1
    
    # 1. 데이터 로드
    try:
        decisions_df = load_decision_data("agent/decisions_data.parquet", years_back=years_back)
        price_df = load_price_data(decisions_df, "ETHUSDT", years_back=years_back)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 데이터 준비
    try:
        decisions_df, price_df = prepare_data(decisions_df, price_df)
    except Exception as e:
        print(f"❌ 데이터 준비 실패: {e}")
        return
    
    # 3. 딥러닝 메타 라벨링 엔진 초기화 (scikit-learn MLPClassifier 사용)
    print("\n🤖 딥러닝 메타 라벨링 엔진 초기화...")
    engine = MetaLabelingNNEngine(
        hidden_layer_sizes=(128, 64, 32),  # 은닉 레이어 크기
        dropout=0.3,                        # L2 정규화 계수
        learning_rate=0.001,               # 학습률
        max_iter=500,                      # 최대 반복 횟수
        confidence_threshold=0.5           # 임계값
    )
    
    # 4. 모델 학습
    print("\n🎓 모델 학습 시작...")
    try:
        result = engine.train(
            decisions_df=decisions_df,
            price_data=price_df,
            test_size=0.2,
            min_profit_threshold=0.005,
            use_profit_based=True
        )
        
        if result["success"]:
            print("\n" + "=" * 60)
            print("✅ 모델 학습 완료!")
            print("=" * 60)
            print(f"정확도: {result['accuracy']:.3f}")
            print(f"ROC-AUC: {result['roc_auc']:.3f}")
            print(f"Precision: {result['precision']:.3f}")
            print(f"Recall: {result['recall']:.3f}")
            print(f"입력 차원: {result['input_dim']}")
            print(f"학습 샘플: {result['train_samples']}개")
            print(f"테스트 샘플: {result['test_samples']}개")
        else:
            print(f"\n❌ 모델 학습 실패: {result.get('message', '알 수 없는 오류')}")
    
    except Exception as e:
        print(f"\n❌ 모델 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

