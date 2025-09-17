#!/usr/bin/env python3
"""
decisions_data.parquet 내 각 행(레코드)의 키 개수/구성이 동일한지 점검하는 유틸.

이 parquet 파일은 각 행이 이미 flatten된 딕셔너리 형태로 저장되어 있음.
각 행의 컬럼들이 키가 되고, 모든 행이 동일한 키 세트를 가져야 함.

사용법:
  python tools/check_decision_keys.py [--path data/decision/decisions_data.parquet]

동작:
  - Parquet 로드
  - 각 행의 키 개수와 키 세트를 비교
  - 차이점이 있는 행들을 보고
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


def analyze_key_consistency(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """각 행의 키 개수와 키 세트 일관성 분석"""
    if df.empty:
        return True, {"total_rows": 0, "key_counts": {}, "key_sets": {}}
    
    # 모든 컬럼명이 키가 됨
    expected_keys = set(df.columns)
    expected_key_count = len(expected_keys)
    
    # 각 행별로 키 개수와 누락된 키 체크
    key_counts = {}
    missing_keys_per_row = {}
    extra_keys_per_row = {}
    
    for idx, row in df.iterrows():
        # 실제로 값이 있는 키들 (NaN이 아닌 키들)
        actual_keys = set(row.dropna().index)
        key_count = len(actual_keys)
        
        key_counts[idx] = key_count
        
        # 누락된 키들
        missing = expected_keys - actual_keys
        if missing:
            missing_keys_per_row[idx] = missing
        
        # 예상보다 많은 키들 (일반적으로 없어야 함)
        extra = actual_keys - expected_keys
        if extra:
            extra_keys_per_row[idx] = extra
    
    # 키 개수 분포
    key_count_distribution = {}
    for count in key_counts.values():
        key_count_distribution[count] = key_count_distribution.get(count, 0) + 1
    
    # 일관성 체크
    is_consistent = (
        len(key_count_distribution) == 1 and  # 모든 행이 동일한 키 개수
        list(key_count_distribution.keys())[0] == expected_key_count and  # 예상 키 개수와 일치
        not missing_keys_per_row and  # 누락된 키 없음
        not extra_keys_per_row  # 추가 키 없음
    )
    
    return is_consistent, {
        "total_rows": len(df),
        "expected_key_count": expected_key_count,
        "expected_keys": expected_keys,
        "key_counts": key_counts,
        "key_count_distribution": key_count_distribution,
        "missing_keys_per_row": missing_keys_per_row,
        "extra_keys_per_row": extra_keys_per_row
    }


def print_analysis_report(analysis: Dict[str, Any], max_examples: int = 10) -> None:
    """분석 결과를 출력"""
    print(f"\n=== 키 일관성 분석 결과 ===")
    print(f"총 행 수: {analysis['total_rows']}")
    print(f"예상 키 개수: {analysis['expected_key_count']}")
    print(f"예상 키들: {sorted(list(analysis['expected_keys']))}")
    
    print(f"\n키 개수 분포:")
    for count, freq in sorted(analysis['key_count_distribution'].items()):
        print(f"  {count}개: {freq}행")
    
    # 누락된 키가 있는 행들
    if analysis['missing_keys_per_row']:
        print(f"\n누락된 키가 있는 행들 (최대 {max_examples}개):")
        for i, (idx, missing) in enumerate(list(analysis['missing_keys_per_row'].items())[:max_examples]):
            print(f"  행 {idx}: 누락된 키 {len(missing)}개 - {sorted(list(missing))}")
    
    # 추가 키가 있는 행들
    if analysis['extra_keys_per_row']:
        print(f"\n추가 키가 있는 행들 (최대 {max_examples}개):")
        for i, (idx, extra) in enumerate(list(analysis['extra_keys_per_row'].items())[:max_examples]):
            print(f"  행 {idx}: 추가 키 {len(extra)}개 - {sorted(list(extra))}")
    
    # 키 개수가 다른 행들
    expected_count = analysis['expected_key_count']
    different_count_rows = {idx: count for idx, count in analysis['key_counts'].items() 
                           if count != expected_count}
    if different_count_rows:
        print(f"\n키 개수가 다른 행들 (예상: {expected_count}개):")
        for idx, count in list(different_count_rows.items())[:max_examples]:
            print(f"  행 {idx}: {count}개")


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
    
    is_consistent, analysis = analyze_key_consistency(df)
    print_analysis_report(analysis, max_examples=args.max_examples)
    
    if is_consistent:
        print(f"\n✅ 결론: 모든 행의 키 구성이 일관됩니다!")
    else:
        print(f"\n❌ 결론: 행마다 키 구성이 다릅니다!")
        
        # 문제가 있는 행들의 인덱스 목록
        problem_rows = set()
        problem_rows.update(analysis['missing_keys_per_row'].keys())
        problem_rows.update(analysis['extra_keys_per_row'].keys())
        problem_rows.update({idx for idx, count in analysis['key_counts'].items() 
                           if count != analysis['expected_key_count']})
        
        if problem_rows:
            print(f"문제가 있는 행들: {sorted(list(problem_rows))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())