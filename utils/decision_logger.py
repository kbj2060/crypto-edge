#!/usr/bin/env python3
"""
Decision Logger - 매일 Parquet 파일로 decision 로그 저장
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from utils.time_manager import get_time_manager
from utils.session_manager import get_session_manager


class DecisionLogger:
    """Decision 로그를 매일 Parquet 파일로 저장하는 클래스"""
    
    def __init__(self, symbol: str = "ETHUSDC", logs_dir: str = "logs"):
        """
        DecisionLogger 초기화
        
        Args:
            symbol: 거래 심볼
            logs_dir: 로그 저장 디렉토리
        """
        self.symbol = symbol.upper()
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        self.time_manager = get_time_manager()
        self.session_manager = get_session_manager()
        
        # 한국 시간대 (UTC+9)
        self.kst_timezone = timezone(timedelta(hours=9))
        
        # 현재 로그 파일과 버퍼
        self.current_log_file = None
        self.decision_buffer = []
        
        # 로그 파일 초기화
        self._ensure_log_file_exists()
    
    def _get_log_file_path(self) -> Path:
        """오늘 날짜의 로그 파일 경로 반환 (한국 시간 기준)"""
        # UTC 시간을 한국 시간으로 변환
        utc_time = self.time_manager.get_current_time()
        kst_time = utc_time.astimezone(self.kst_timezone)
        today = kst_time.date()
        return self.logs_dir / f"decisions_{today.strftime('%Y%m%d')}.parquet"
    
    def _ensure_log_file_exists(self):
        """로그 파일이 존재하지 않으면 생성"""
        log_file = self._get_log_file_path()
        
        if self.current_log_file != str(log_file):
            self.current_log_file = str(log_file)
            
            # 기존 파일이 있으면 로드, 없으면 새로 생성
            if log_file.exists():
                try:
                    existing_df = pd.read_parquet(log_file)
                    # timestamp 컬럼을 datetime 타입 (UTC)으로 변환
                    if 'timestamp' in existing_df.columns:
                        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], utc=True)
                    self.decision_buffer = existing_df.to_dict('records')
                    print(f"📂 기존 로그 파일 로드: {log_file} ({len(self.decision_buffer)}개 기록)")
                except Exception as e:
                    print(f"⚠️ 기존 로그 파일 로드 실패: {e}")
                    self.decision_buffer = []
            else:
                self.decision_buffer = []
                print(f"📝 새 로그 파일 생성: {log_file}")
    
    def log_decision(self, decision: Dict[str, Any]) -> bool:
        """
        Decision을 로그에 저장
        
        Args:
            decision: 저장할 decision 딕셔너리
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 로그 파일 확인 (날짜가 바뀌었을 수 있음)
            self._ensure_log_file_exists()
            
            # 현재 시간을 한국 시간으로 변환하여 추가
            utc_time = self.time_manager.get_current_time()
            kst_time = utc_time.astimezone(self.kst_timezone)
            
            decision_with_timestamp = {
                'timestamp': kst_time,  # datetime 객체로 저장
                'symbol': self.symbol,
                **decision
            }
            
            # 버퍼에 추가
            self.decision_buffer.append(decision_with_timestamp)
            
            # DataFrame으로 변환
            df = pd.DataFrame(self.decision_buffer)
            
            # timestamp 컬럼을 datetime 타입 (UTC)으로 명시적 변환
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Parquet으로 저장 (datetime 타입 보존)
            log_file = self._get_log_file_path()
            df.to_parquet(log_file, index=False, engine='pyarrow')
            
            print(f"📝 Decision 로그 저장: {log_file} (총 {len(self.decision_buffer)}개 기록)")
            return True
            
        except Exception as e:
            print(f"❌ Decision 로그 저장 실패: {e}")
            return False
    
    def get_today_decisions(self) -> pd.DataFrame:
        """오늘의 모든 decision 반환"""
        try:
            log_file = self._get_log_file_path()
            if log_file.exists():
                df = pd.read_parquet(log_file)
                # timestamp 컬럼을 datetime 타입 (UTC)으로 변환
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"❌ 오늘의 decision 로드 실패: {e}")
            return pd.DataFrame()
    
    def get_decision_count_today(self) -> int:
        """오늘의 decision 개수 반환"""
        return len(self.decision_buffer)
    
    def get_log_files(self) -> list:
        """모든 로그 파일 목록 반환"""
        try:
            return sorted([f for f in self.logs_dir.glob("decisions_*.parquet")])
        except Exception as e:
            print(f"❌ 로그 파일 목록 조회 실패: {e}")
            return []
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """오래된 로그 파일 정리"""
        try:
            log_files = self.get_log_files()
            # 한국 시간 기준으로 오래된 파일 정리
            utc_time = self.time_manager.get_current_time()
            kst_time = utc_time.astimezone(self.kst_timezone)
            cutoff_date = kst_time.date() - pd.Timedelta(days=days_to_keep)
            
            deleted_count = 0
            for log_file in log_files:
                # 파일명에서 날짜 추출 (decisions_YYYYMMDD.parquet)
                try:
                    date_str = log_file.stem.split('_')[1]
                    file_date = datetime.strptime(date_str, '%Y%m%d').date()
                    
                    if file_date < cutoff_date:
                        log_file.unlink()
                        deleted_count += 1
                        print(f"🗑️ 오래된 로그 파일 삭제: {log_file}")
                except Exception as e:
                    print(f"⚠️ 파일 날짜 파싱 실패 {log_file}: {e}")
            
            if deleted_count > 0:
                print(f"✅ {deleted_count}개의 오래된 로그 파일 정리 완료")
            else:
                print("📁 정리할 오래된 로그 파일 없음")
                
        except Exception as e:
            print(f"❌ 로그 파일 정리 실패: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """로그 통계 정보 반환"""
        try:
            log_files = self.get_log_files()
            total_files = len(log_files)
            today_count = self.get_decision_count_today()
            
            # 전체 decision 수 계산
            total_decisions = 0
            for log_file in log_files:
                try:
                    df = pd.read_parquet(log_file)
                    total_decisions += len(df)
                except:
                    pass
            
            return {
                'total_log_files': total_files,
                'total_decisions': total_decisions,
                'today_decisions': today_count,
                'logs_directory': str(self.logs_dir),
                'current_log_file': self.current_log_file
            }
        except Exception as e:
            print(f"❌ 통계 정보 조회 실패: {e}")
            return {}


# 전역 DecisionLogger 인스턴스
_global_decision_logger: Optional[DecisionLogger] = None

def get_decision_logger(symbol: str = "ETHUSDC") -> DecisionLogger:
    """전역 DecisionLogger 인스턴스 반환 (싱글톤 패턴)"""
    global _global_decision_logger
    
    if _global_decision_logger is None:
        _global_decision_logger = DecisionLogger(symbol)
    
    return _global_decision_logger
