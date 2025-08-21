#!/usr/bin/env python3
"""
실시간 청산 데이터 수집기 실행 스크립트
"""

import asyncio
import argparse
import logging
import sys

# 상대 경로로 import
from data.liquidation_collector import LiquidationCollector

# 로깅 설정 - 파일에만 로그 저장, 콘솔은 청산 데이터만
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('liquidation_collector.log')
    ]
)
logger = logging.getLogger(__name__)

# 추가 로거들도 INFO 레벨로 설정 (파일에만)
logging.getLogger('data.liquidation_collector').setLevel(logging.INFO)
logging.getLogger('data.binance_client').setLevel(logging.INFO)
logging.getLogger('websockets').setLevel(logging.INFO)


async def run_collector(symbols: list, db_path: str = "data/liquidations.db"):
    """실제 청산 데이터 수집기 실행"""
    collector = None
    try:
        logger.info("실제 청산 데이터 수집기 시작")
        collector = LiquidationCollector(symbols, db_path)
        await collector.start()
        
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"수집기 실행 오류: {e}")
    finally:
        if collector:
            await collector.stop()


async def run_realtime_display(symbols: list, db_path: str = "data/liquidations.db"):
    """실시간 데이터 출력 모드"""
    try:
        collector = LiquidationCollector(symbols, db_path)
        await collector.start()
        
    except Exception as e:
        logger.error(f"실시간 출력 모드 오류: {e}")


def show_status():
    """데이터베이스 상태 표시"""
    try:
        from data.liquidation_database import LiquidationDatabase
        
        db_path = "data/liquidations.db"
        db = LiquidationDatabase(db_path)
        
        # 데이터베이스 상태 조회
        total_events = db.get_total_events()
        recent_24h = db.get_recent_events(24)
        recent_1h = db.get_recent_events(1)
        db_size = db.get_database_size()
        symbol_stats = db.get_symbol_statistics()
        side_stats = db.get_side_statistics()
        
        print("=" * 60)
        print("청산 데이터베이스 상태")
        print("=" * 60)
        print(f"총 이벤트 수: {total_events:,}")
        print(f"최근 24시간: {recent_24h:,}")
        print(f"최근 1시간: {recent_1h:,}")
        print(f"데이터베이스 크기: {db_size:.2f} MB")
        
        if symbol_stats:
            print("🔸 심볼별 이벤트 수:")
            for symbol, count in symbol_stats.items():
                print(f"  {symbol}: {count:,}")
        
        if side_stats:
            print("🔸 사이드별 이벤트 수:")
            for side, count in side_stats.items():
                side_name = "BUY (숏 청산)" if side == "BUY" else "SELL (롱 청산)"
                print(f"  {side_name}: {count:,}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"상태 조회 오류: {e}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='실시간 청산 데이터 수집 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 실제 청산 데이터 수집
  python run.py --collect
  
  # 실시간 데이터 출력 모드
  python run.py --realtime
  
  # 데이터베이스 상태 확인
  python run.py --status
        """
    )
    
    # 주요 동작
    parser.add_argument('--collect', action='store_true', help='실제 청산 데이터 수집 시작')
    parser.add_argument('--realtime', action='store_true', help='실시간 데이터 출력 모드')
    parser.add_argument('--status', action='store_true', help='데이터베이스 상태 확인')
    
    # 수집기 옵션
    parser.add_argument('--symbols', nargs='+', 
                        default=['ETHUSDT'],
                        help='수집할 심볼 목록')
    parser.add_argument('--db-path', default='data/liquidations.db', help='데이터베이스 경로')
    
    args = parser.parse_args()
    
    try:
        if args.status:
            # 상태 확인
            show_status()
            
        elif args.collect:
            # 실제 데이터 수집
            logger.info("실제 청산 데이터 수집 시작")
            asyncio.run(run_collector(args.symbols, db_path=args.db_path))
            
        elif args.realtime:
            # 실시간 데이터 출력 모드
            logger.info("실시간 데이터 출력 모드 시작")
            asyncio.run(run_realtime_display(args.symbols, db_path=args.db_path))
            
        else:
            # 기본 동작: 상태 확인
            show_status()
            print("\n사용법을 보려면 --help를 사용하세요.")
            
    except KeyboardInterrupt:
        logger.info("프로그램이 중단됨")
    except Exception as e:
        logger.error(f"프로그램 실행 오류: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
