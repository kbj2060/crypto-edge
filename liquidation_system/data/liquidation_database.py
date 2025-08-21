#!/usr/bin/env python3
"""
실시간 청산 데이터 데이터베이스 관리자
SQLite를 사용하여 청산 이벤트를 효율적으로 저장하고 조회합니다.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiquidationDatabase:
    """청산 데이터 데이터베이스 관리자"""
    
    def __init__(self, db_path: str = "data/liquidations.db"):
        """데이터베이스 초기화"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """데이터베이스 테이블 초기화"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 청산 이벤트 테이블
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS liquidation_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,  -- 'BUY' (롱 청산) 또는 'SELL' (숏 청산)
                        size REAL NOT NULL,
                        price REAL NOT NULL,
                        lpi REAL,  -- Liquidation Price Index
                        timestamp DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # 인덱스 생성
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON liquidation_events (symbol, timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_side_timestamp ON liquidation_events (side, timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON liquidation_events (timestamp)")
                
                # 청산 집계 테이블 (1분 단위)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS liquidation_aggregates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        total_size REAL NOT NULL,
                        avg_price REAL NOT NULL,
                        avg_lpi REAL,
                        event_count INTEGER NOT NULL,
                        period_start DATETIME NOT NULL,
                        period_end DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, side, period_start)
                    )
                """)
                
                # 집계 테이블 인덱스
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_period ON liquidation_aggregates (symbol, period_start)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_side_period ON liquidation_aggregates (side, period_start)")
                
                # 청산 강도 테이블 (1시간 단위)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS liquidation_intensity (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        side TEXT NOT NULL,
                        total_size REAL NOT NULL,
                        avg_lpi REAL,
                        event_count INTEGER NOT NULL,
                        period_start DATETIME NOT NULL,
                        period_end DATETIME NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, side, period_start)
                    )
                """)
                
                # 강도 테이블 인덱스
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_intensity_symbol_period ON liquidation_intensity (symbol, period_start)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_intensity_side_period ON liquidation_intensity (side, period_start)")
                
                conn.commit()
                logger.info(f"데이터베이스 초기화 완료: {self.db_path}")
                
        except Exception as e:
            logger.error(f"데이터베이스 초기화 오류: {e}")
            raise
    
    def insert_liquidation_event(self, event: Dict[str, Any]) -> bool:
        """개별 청산 이벤트 삽입"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO liquidation_events 
                    (symbol, side, size, price, lpi, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event.get('symbol', 'UNKNOWN'),
                    event.get('side', 'UNKNOWN'),
                    event.get('size', 0.0),
                    event.get('price', 0.0),
                    event.get('lpi', 0.0),
                    event.get('timestamp', datetime.now())
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"청산 이벤트 삽입 오류: {e}")
            return False
    
    def insert_liquidation_events_batch(self, events: List[Dict[str, Any]]) -> int:
        """여러 청산 이벤트를 일괄 삽입"""
        if not events:
            return 0
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 배치 삽입을 위한 데이터 준비
                data = [
                    (
                        event.get('symbol', 'UNKNOWN'),
                        event.get('side', 'UNKNOWN'),
                        event.get('size', 0.0),
                        event.get('price', 0.0),
                        event.get('lpi', 0.0),
                        event.get('timestamp', datetime.now())
                    )
                    for event in events
                ]
                
                cursor.executemany("""
                    INSERT INTO liquidation_events 
                    (symbol, side, size, price, lpi, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, data)
                
                conn.commit()
                inserted_count = cursor.rowcount
                return inserted_count
                
        except Exception as e:
            logger.error(f"배치 삽입 오류: {e}")
            return 0
    
    def get_recent_liquidations(self, symbol: str = None, 
                               side: str = None, 
                               hours: int = 24,
                               limit: int = 1000) -> pd.DataFrame:
        """최근 청산 데이터 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT * FROM liquidation_events 
                    WHERE timestamp >= datetime('now', '-{} hours')
                """.format(hours)
                
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if side:
                    query += " AND side = ?"
                    params.append(side)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['created_at'] = pd.to_datetime(df['created_at'])
                
                return df
                
        except Exception as e:
            logger.error(f"데이터 조회 오류: {e}")
            return pd.DataFrame()
    
    def get_liquidation_summary(self, symbol: str = None, 
                               hours: int = 24) -> Dict[str, Any]:
        """청산 데이터 요약 통계"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        side,
                        COUNT(*) as event_count,
                        SUM(size) as total_size,
                        AVG(price) as avg_price,
                        AVG(lpi) as avg_lpi,
                        MIN(price) as min_price,
                        MAX(price) as max_price
                    FROM liquidation_events 
                    WHERE timestamp >= datetime('now', '-{} hours')
                """.format(hours)
                
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                query += " GROUP BY side"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                summary = {
                    'period_hours': hours,
                    'symbol': symbol,
                    'total_events': df['event_count'].sum() if not df.empty else 0,
                    'by_side': {}
                }
                
                for _, row in df.iterrows():
                    side = row['side']
                    summary['by_side'][side] = {
                        'event_count': int(row['event_count']),
                        'total_size': float(row['total_size']),
                        'avg_price': float(row['avg_price']),
                        'avg_lpi': float(row['avg_lpi']) if pd.notna(row['avg_lpi']) else 0.0,
                        'min_price': float(row['min_price']),
                        'max_price': float(row['max_price'])
                    }
                
                return summary
                
        except Exception as e:
            logger.error(f"요약 통계 조회 오류: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30) -> int:
        """오래된 데이터 정리"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 오래된 청산 이벤트 삭제
                cursor.execute("""
                    DELETE FROM liquidation_events 
                    WHERE timestamp < datetime('now', '-{} days')
                """.format(days))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"오래된 데이터 정리 완료: {deleted_count}개 이벤트 삭제")
                return deleted_count
                
        except Exception as e:
            logger.error(f"데이터 정리 오류: {e}")
            return 0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 전체 이벤트 수
                cursor.execute("SELECT COUNT(*) FROM liquidation_events")
                total_events = cursor.fetchone()[0]
                
                # 최근 24시간 이벤트 수
                cursor.execute("""
                    SELECT COUNT(*) FROM liquidation_events 
                    WHERE timestamp >= datetime('now', '-24 hours')
                """)
                recent_24h = cursor.fetchone()[0]
                
                # 최근 1시간 이벤트 수
                cursor.execute("""
                    SELECT COUNT(*) FROM liquidation_events 
                    WHERE timestamp >= datetime('now', '-1 hour')
                """)
                recent_1h = cursor.fetchone()[0]
                
                # 심볼별 이벤트 수
                cursor.execute("""
                    SELECT symbol, COUNT(*) as count 
                    FROM liquidation_events 
                    GROUP BY symbol 
                    ORDER BY count DESC 
                    LIMIT 10
                """)
                symbol_counts = dict(cursor.fetchall())
                
                # 사이드별 이벤트 수
                cursor.execute("""
                    SELECT side, COUNT(*) as count 
                    FROM liquidation_events 
                    GROUP BY side
                """)
                side_counts = dict(cursor.fetchall())
                
                return {
                    'total_events': total_events,
                    'recent_24h': recent_24h,
                    'recent_1h': recent_1h,
                    'symbol_counts': symbol_counts,
                    'side_counts': side_counts,
                    'database_size_mb': self.db_path.stat().st_size / (1024 * 1024)
                }
                
        except Exception as e:
            logger.error(f"데이터베이스 통계 조회 오류: {e}")
            return {}


if __name__ == "__main__":
    # 테스트 코드
    db = LiquidationDatabase()
    
    # 샘플 데이터 삽입
    sample_events = [
        {
            'symbol': 'ETHUSDT',
            'side': 'SELL',  # 롱 청산
            'size': 100.5,
            'price': 2500.0,
            'lpi': 0.8,
            'timestamp': datetime.now()
        },
        {
            'symbol': 'BTCUSDT',
            'side': 'BUY',   # 숏 청산
            'size': 2.5,
            'price': 45000.0,
            'lpi': 0.9,
            'timestamp': datetime.now()
        }
    ]
    
    # 이벤트 삽입
    db.insert_liquidation_events_batch(sample_events)
    
    # 통계 조회
    stats = db.get_database_stats()
    print("데이터베이스 통계:")
    print(json.dumps(stats, indent=2, default=str))
    
    # 최근 데이터 조회
    recent_data = db.get_recent_liquidations(hours=1)
    print(f"\n최근 1시간 데이터: {len(recent_data)}개")
    if not recent_data.empty:
        print(recent_data.head())
