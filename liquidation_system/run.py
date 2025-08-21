#!/usr/bin/env python3
"""
청산 데이터 수집 시스템 실행 스크립트
독립적인 실행을 위한 간단한 진입점입니다.
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_liquidation_collector import main

if __name__ == "__main__":
    main()

