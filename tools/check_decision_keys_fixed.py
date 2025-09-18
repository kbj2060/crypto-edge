#!/usr/bin/env python3
"""
decisions_data.parquet 내 각 행(레코드)의 키 개수/구성이 동일한지 점검하는 유틸 (수정버전).

이 parquet 파일은 각 행이 이미 flatten된 딕셔너리 형태로 저장되어 있음.
각 행의 컬럼들이 키가 되고, 모든 행이 동일한 키 세트를 가져야 함.

사용법:
  python tools/check_decision_keys_fixed.py [--path data/decision/decisions_data.parquet]

동작:
  - Parquet 로드
  - 모든 행이 동일한 컬럼 구조를 가지는지 확인 (스키마 일관성)
  - NaN 값 분포 분석 (데이터 완성도)
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


def load_parquet(path: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        # pyarrow 미설치·오류 시 fastparquet 시도
        return pd.read_parquet(path, engine="fastparquet")


def analyze_schema_consistency(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """스키마 일관성 분석 (모든 행이 동일한 컬럼 구조를 가지는지)"""
    if df.empty:
        return True, {"total_rows": 0, "total_columns": 0, "column_names": []}
    
    # 모든 행이 동일한 컬럼을 가짐 (Parquet의 특성상 당연함)
    total_columns = len(df.columns)
    column_names = list(df.columns)
    
    # 스키마는 항상 일관됨 (Parquet 파일의 특성)
    is_schema_consistent = True
    
    return is_schema_consistent, {
        "total_rows": len(df),
        "total_columns": total_columns,
        "column_names": column_names
    }


def analyze_data_completeness(df: pd.DataFrame) -> Dict[str, Any]:
    """데이터 완성도 분석 (각 행에서 NaN이 아닌 값의 개수)"""
    if df.empty:
        return {"non_nan_counts": {}, "non_nan_distribution": {}}
    
    # 각 행별로 NaN이 아닌 값의 개수 계산
    non_nan_counts = {}
    for idx, row in df.iterrows():
        non_nan_count = row.notna().sum()
        non_nan_counts[idx] = non_nan_count
    
    # 값 개수 분포
    non_nan_distribution = {}
    for count in non_nan_counts.values():
        non_nan_distribution[count] = non_nan_distribution.get(count, 0) + 1
    
    # 누락된 값이 많은 행들 찾기
    max_possible = len(df.columns)
    incomplete_rows = {idx: count for idx, count in non_nan_counts.items() 
                      if count < max_possible}
    
    return {
        "non_nan_counts": non_nan_counts,
        "non_nan_distribution": non_nan_distribution,
        "incomplete_rows": incomplete_rows,
        "max_possible_values": max_possible
    }


def analyze_missing_keys_per_row(df: pd.DataFrame, max_examples: int = 10) -> Dict[str, Any]:
    """각 행별로 누락된 키들 분석"""
    if df.empty:
        return {"missing_keys_per_row": {}}
    
    all_columns = set(df.columns)
    missing_keys_per_row = {}
    
    for idx, row in df.iterrows():
        # NaN인 컬럼들을 누락된 키로 간주
        nan_columns = set(row[row.isna()].index)
        if nan_columns:
            missing_keys_per_row[idx] = nan_columns
    
    return {"missing_keys_per_row": missing_keys_per_row}


def print_analysis_report(schema_analysis: Dict[str, Any], 
                         completeness_analysis: Dict[str, Any],
                         missing_keys_analysis: Dict[str, Any],
                         max_examples: int = 10) -> None:
    """분석 결과를 출력"""
    print(f"\n=== 스키마 일관성 분석 결과 ===")
    print(f"총 행 수: {schema_analysis['total_rows']}")
    print(f"총 컬럼 수: {schema_analysis['total_columns']}")
    print(f"컬럼 이름들: {schema_analysis['column_names'][:10]}{'...' if len(schema_analysis['column_names']) > 10 else ''}")
    
    print(f"\n=== 데이터 완성도 분석 ===")
    print(f"최대 가능한 값 개수: {completeness_analysis['max_possible_values']}")
    print(f"값 개수 분포:")
    for count, freq in sorted(completeness_analysis['non_nan_distribution'].items()):
        print(f"  {count}개 값: {freq}행")
    
    # 불완전한 행들
    incomplete_rows = completeness_analysis['incomplete_rows']
    if incomplete_rows:
        print(f"\n값이 누락된 행들 (최대 {max_examples}개):")
        for i, (idx, count) in enumerate(list(incomplete_rows.items())[:max_examples]):
            missing_count = completeness_analysis['max_possible_values'] - count
            print(f"  행 {idx}: {count}개 값 (누락: {missing_count}개)")
    
    # 누락된 키들
    missing_keys_per_row = missing_keys_analysis['missing_keys_per_row']
    if missing_keys_per_row:
        print(f"\n누락된 키가 있는 행들 (최대 {max_examples}개):")
        for i, (idx, missing) in enumerate(list(missing_keys_per_row.items())[:max_examples]):
            print(f"  행 {idx}: 누락된 키 {len(missing)}개 - {sorted(list(missing))[:5]}{'...' if len(missing) > 5 else ''}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="agent/decisions_data.parquet", help="Parquet 파일 경로")
    parser.add_argument("--max-examples", type=int, default=10, help="차이 사례 출력 최대 개수")
    args = parser.parse_args()

    try:
        df = load_parquet(args.path)
    except Exception as e:
        print(f"Parquet 로드 실패: {e}")
        return 1

    if df.empty:
        print("파일이 비어있습니다.")
        return 0

    print(f"파일 로드 완료: {len(df)}행, {len(df.columns)}컬럼")
    
    # 스키마 일관성 분석
    is_schema_consistent, schema_analysis = analyze_schema_consistency(df)
    
    # 데이터 완성도 분석
    completeness_analysis = analyze_data_completeness(df)
    
    # 누락된 키 분석
    missing_keys_analysis = analyze_missing_keys_per_row(df, max_examples=args.max_examples)
    
    # 결과 출력
    print_analysis_report(schema_analysis, completeness_analysis, missing_keys_analysis, max_examples=args.max_examples)
    
    # 결론
    max_values = completeness_analysis['max_possible_values']
    value_counts = list(completeness_analysis['non_nan_distribution'].keys())
    
    if len(value_counts) == 1 and value_counts[0] == max_values:
        print(f"\n✅ 결론: 모든 행이 완전한 데이터를 가지고 있습니다! ({max_values}개 값)")
    else:
        print(f"\n⚠️  결론: 일부 행에 누락된 값이 있습니다!")
        print(f"완전한 행: {completeness_analysis['non_nan_distribution'].get(max_values, 0)}개")
        print(f"불완전한 행: {len(completeness_analysis['incomplete_rows'])}개")

    return 0


if __name__ == "__main__":
    sys.exit(main())

