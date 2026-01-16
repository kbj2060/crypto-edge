#!/usr/bin/env python3
"""
메타 라벨링 모델 학습 스크립트

과거 거래 결정 데이터와 가격 데이터를 사용하여
메타 라벨링 모델을 학습합니다.
"""

import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from engines.meta_labeling_engine import MetaLabelingEngine
from agent.decision_generator import load_decisions_from_parquet
from managers.binance_dataloader import BinanceDataLoader


def load_price_data(symbol: str = "ETHUSDT", limit: int = 10000) -> pd.DataFrame:
    """가격 데이터 로드"""
    print(f"📊 가격 데이터 로드 중... (심볼: {symbol}, 최대 {limit}개)")
    
    dataloader = BinanceDataLoader()
    
    # 최근 데이터 가져오기
    df = dataloader.fetch_data(
        interval="3m",
        symbol=symbol,
        limit=limit
    )
    
    if df is None or df.empty:
        raise ValueError(f"가격 데이터를 가져올 수 없습니다: {symbol}")
    
    # 인덱스를 timestamp로 설정
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("가격 데이터에 timestamp가 없습니다")
    
    print(f"✅ 가격 데이터 로드 완료: {len(df)}개 캔들")
    return df


def load_decision_data(filename: str = "agent/decisions_data.parquet") -> pd.DataFrame:
    """결정 데이터 로드"""
    print(f"📊 결정 데이터 로드 중... ({filename})")
    
    df = load_decisions_from_parquet(filename)
    
    if df is None or df.empty:
        raise ValueError(f"결정 데이터를 로드할 수 없습니다: {filename}")
    
    # timestamp 처리
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("결정 데이터에 timestamp가 없습니다")
    
    print(f"✅ 결정 데이터 로드 완료: {len(df)}개 레코드")
    return df


def prepare_data(decisions_df: pd.DataFrame, price_df: pd.DataFrame) -> tuple:
    """데이터 준비 및 정렬"""
    print("🔧 데이터 준비 중...")
    
    # timestamp 기준으로 정렬
    if 'timestamp' in decisions_df.columns:
        decisions_df = decisions_df.set_index('timestamp')
    
    decisions_df = decisions_df.sort_index()
    price_df = price_df.sort_index()
    
    # 시간 범위 맞추기
    min_time = max(decisions_df.index.min(), price_df.index.min())
    max_time = min(decisions_df.index.max(), price_df.index.max())
    
    decisions_df = decisions_df[
        (decisions_df.index >= min_time) & 
        (decisions_df.index <= max_time)
    ]
    price_df = price_df[
        (price_df.index >= min_time) & 
        (price_df.index <= max_time)
    ]
    
    print(f"✅ 데이터 준비 완료:")
    print(f"   결정 데이터: {len(decisions_df)}개")
    print(f"   가격 데이터: {len(price_df)}개")
    print(f"   시간 범위: {min_time} ~ {max_time}")
    
    return decisions_df.reset_index(), price_df


def main():
    """메인 함수"""
    print("=" * 60)
    print("메타 라벨링 모델 학습 시작")
    print("=" * 60)
    
    # 1. 데이터 로드
    try:
        decisions_df = load_decision_data("agent/decisions_data.parquet")
        price_df = load_price_data("ETHUSDT", limit=20000)
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return
    
    # 2. 데이터 준비
    try:
        decisions_df, price_df = prepare_data(decisions_df, price_df)
    except Exception as e:
        print(f"❌ 데이터 준비 실패: {e}")
        return
    
    # 3. 메타 라벨링 엔진 초기화
    print("\n🤖 메타 라벨링 엔진 초기화...")
    engine = MetaLabelingEngine(
        model_type="random_forest",
        min_samples_for_training=100,
        confidence_threshold=0.6
    )
    
    # 4. 모델 학습
    print("\n🎓 모델 학습 시작...")
    try:
        result = engine.train(
            decisions_df=decisions_df,
            price_data=price_df,
            test_size=0.2,
            retrain=True
        )
        
        if result["success"]:
            print("\n" + "=" * 60)
            print("✅ 모델 학습 완료!")
            print("=" * 60)
            print(f"정확도: {result['accuracy']:.3f}")
            print(f"ROC-AUC: {result['roc_auc']:.3f}")
            print(f"학습 샘플: {result['train_samples']}개")
            print(f"테스트 샘플: {result['test_samples']}개")
            
            if result.get('feature_importance'):
                print("\n특성 중요도 (상위 5개):")
                sorted_features = sorted(
                    result['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                for feature, importance in sorted_features:
                    print(f"  {feature}: {importance:.3f}")
        else:
            print(f"\n❌ 모델 학습 실패: {result.get('message', '알 수 없는 오류')}")
    
    except Exception as e:
        print(f"\n❌ 모델 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


