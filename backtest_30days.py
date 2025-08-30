#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
30일 백테스팅 시스템
- 30일 이전부터 현재까지의 데이터로 백테스팅 수행
- 모든 전략의 성과 분석
- 수익률, 승률, 최대 낙폭 등 주요 지표 계산
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
import json
import os

# 프로젝트 모듈 import
from data.binance_dataloader import BinanceDataLoader
from data.data_manager import get_data_manager
from indicators.global_indicators import initialize_global_indicators, get_global_indicator_manager
from signals.vpvr_golden_strategy import LVNGoldenPocket
from signals.bollinger_squeeze_strategy import BollingerSqueezeStrategy
from signals.vwap_pinball_strategy import VWAPPinballStrategy
from signals.liquidation_strategies_lite import FadeReentryStrategy, SqueezeMomentumStrategy
from signals.session_or_lite import SessionORLite
from utils.time_manager import get_time_manager

    
class BacktestEngine:
    # Backtest configuration defaults
    DEFAULT_MIN_SCORE = 0.30  # minimum score required to accept a signal when scoring is provided
    DEFAULT_USE_SCORE_FILTER = True  # if True, backtest will only open trades with score >= min_score

    """백테스팅 엔진"""
    
    def __init__(self, symbol: str = "ETHUSDT", initial_capital: float = 10000.0, min_score: Optional[float]=None, use_score_filter: Optional[bool]=None):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []
        self.closed_trades = []
        self.trade_history = []
        
        # scoring / filtering config
        self.min_score = float(min_score) if min_score is not None else float(self.DEFAULT_MIN_SCORE)
        self.use_score_filter = bool(use_score_filter) if use_score_filter is not None else bool(self.DEFAULT_USE_SCORE_FILTER)

        # 전략들 초기화
        self.strategies = {
            'vpvr_golden': LVNGoldenPocket(),
            'bollinger_squeeze': BollingerSqueezeStrategy(),
            'vwap_pinball': VWAPPinballStrategy(),
            'fade_reentry': FadeReentryStrategy(),
            'squeeze_momentum': SqueezeMomentumStrategy(),
            'session_or': SessionORLite()
        }
        
        # 백테스팅 결과
        self.results = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'strategy_performance': {}
        }
    
    def load_historical_data(self, days_back: int = 30) -> pd.DataFrame:
        """과거 데이터 로드"""
        print(f"📊 {days_back}일 이전부터 현재까지 데이터 로딩 중...")
        
        try:
            # Binance 클라이언트로 과거 데이터 가져오기
            client = BinanceDataLoader()
            
            # 현재 시간
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=days_back)
            
            # 3분봉 데이터 가져오기
            klines = client.fetch_data(
                start_time=start_time,
                end_time=end_time
            )
            
            # DataFrame으로 변환
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # 데이터 타입 변환
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 타임스탬프 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
            
            # 인덱스 설정
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ {len(df)}개의 캔들 데이터 로드 완료")
            
            return df
            
        except Exception as e:
            print(f"❌ 데이터 로딩 오류: {e}")
            return pd.DataFrame()
    
    def initialize_indicators(self, df: pd.DataFrame):
        """지표 초기화"""
        print("🚀 지표 시스템 초기화 중...")
        
        try:
            # DataManager 초기화
            data_manager = get_data_manager()
            
            # 글로벌 지표 초기화
            global_manager = initialize_global_indicators()
            
            # 초기 데이터로 지표들 업데이트
            for i, (timestamp, row) in enumerate(df.iterrows()):
                if i < 100:  # 처음 100개 캔들로 지표 초기화
                    global_manager.update_all_indicators(row)
            
            print("✅ 지표 시스템 초기화 완료")
            
        except Exception as e:
            print(f"❌ 지표 초기화 오류: {e}")
    
    def run_backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """백테스팅 실행"""
        print(f"🔄 백테스팅 시작: {len(df)}개 캔들")
        
        # 지표 초기화
        self.initialize_indicators(df)
        
        # 각 캔들에 대해 전략 실행
        for i, (timestamp, candle) in enumerate(df.iterrows()):
            if i < 100:  # 처음 100개는 지표 초기화용으로 스킵
                continue
                
            # 지표 업데이트
            try:
                global_manager = get_global_indicator_manager()
                global_manager.update_all_indicators(candle)
            except Exception as e:
                print(f"⚠️ 지표 업데이트 오류: {e}")
                continue
            
            # 각 전략 실행 (웹소켓과 동일한 방식으로)
            try:
                # 1. VPVR 골든 포켓 전략
                if self.strategies['vpvr_golden'] is not None:
                    signal = self._evaluate_vpvr_strategy(self.strategies['vpvr_golden'], df.iloc[:i+1])
                    if signal:
                        self._process_signal(signal, timestamp, candle, 'vpvr_golden')
                
                # 2. 볼린저 스퀴즈 전략
                if self.strategies['bollinger_squeeze'] is not None:
                    signal = self._evaluate_bollinger_strategy(self.strategies['bollinger_squeeze'], df.iloc[:i+1])
                    if signal:
                        self._process_signal(signal, timestamp, candle, 'bollinger_squeeze')
                
                # 3. VWAP 피니언 전략
                if self.strategies['vwap_pinball'] is not None:
                    signal = self._evaluate_vwap_strategy(self.strategies['vwap_pinball'], df.iloc[:i+1])
                    if signal:
                        self._process_signal(signal, timestamp, candle, 'vwap_pinball')
                
                # 4. 페이드 리입 전략 (3분봉)
                if self.strategies['fade_reentry'] is not None:
                    signal = self._evaluate_fade_reentry_strategy(self.strategies['fade_reentry'], df.iloc[:i+1])
                    if signal:
                        self._process_signal(signal, timestamp, candle, 'fade_reentry')
                
                # 5. 스퀴즈 모멘텀 전략 (1분봉)
                if self.strategies['squeeze_momentum'] is not None:
                    signal = self._evaluate_squeeze_momentum_strategy(self.strategies['squeeze_momentum'], df.iloc[:i+1])
                    if signal:
                        self._process_signal(signal, timestamp, candle, 'squeeze_momentum')
                
                # 6. 세션 OR 전략
                if self.strategies['session_or'] is not None:
                    signal = self._evaluate_session_strategy(self.strategies['session_or'], df.iloc[:i+1])
                    if signal:
                        self._process_signal(signal, timestamp, candle, 'session_or')
                        
            except Exception as e:
                print(f"⚠️ 전략 실행 오류: {e}")
                continue
            
            # 진행률 표시
            if i % 100 == 0:
                progress = (i / len(df)) * 100
                print(f"📈 진행률: {progress:.1f}% ({i}/{len(df)})")
        
        # 백테스팅 결과 계산
        self._calculate_results()
        
        return self.results
    
    def _evaluate_vpvr_strategy(self, strategy: LVNGoldenPocket, df: pd.DataFrame) -> Optional[Dict]:
        """VPVR 골든 포켓 전략 평가 (웹소켓과 동일한 방식)"""
        try:
            # 웹소켓과 동일한 방식으로 VPVRConfig 사용
            config = strategy.VPVRConfig()
            df_3m = df.tail(config.lookback_bars + 5)
            return strategy.evaluate(df_3m)
        except Exception as e:
            print(f"⚠️ VPVR 전략 평가 오류: {e}")
            return None
    
    def _evaluate_bollinger_strategy(self, strategy: BollingerSqueezeStrategy, df: pd.DataFrame) -> Optional[Dict]:
        """볼린저 스퀴즈 전략 평가 (웹소켓과 동일한 방식)"""
        try:
            # 웹소켓과 동일한 방식으로 evaluate() 메서드 호출
            return strategy.evaluate()
        except Exception as e:
            print(f"⚠️ 볼린저 전략 평가 오류: {e}")
            return None
    
    def _evaluate_vwap_strategy(self, strategy: VWAPPinballStrategy, df: pd.DataFrame) -> Optional[Dict]:
        """VWAP 피니언 전략 평가 (웹소켓과 동일한 방식)"""
        try:
            # 웹소켓과 동일한 방식으로 3분봉 데이터 사용
            df_3m = df.tail(4)  # 최근 4개 캔들
            
            # VWAP 값이 None인 경우 처리
            try:
                result = strategy.on_kline_close_3m(df_3m)
                if result is None:
                    return None
                return result
            except Exception as vwap_error:
                if "vwap_val is None" in str(vwap_error):
                    print(f"⚠️ VWAP 값이 없어 VWAP 전략을 건너뜁니다")
                    return None
                else:
                    raise vwap_error
                    
        except Exception as e:
            print(f"⚠️ VWAP 전략 평가 오류: {e}")
            return None
    
    def _evaluate_fade_reentry_strategy(self, strategy: FadeReentryStrategy, df: pd.DataFrame) -> Optional[Dict]:
        """페이드 리입 전략 평가 (웹소켓과 동일한 방식)"""
        try:
            # 데이터 길이 체크
            if len(df) < 2:
                return None
                
            # 웹소켓과 동일한 방식으로 3분봉 마감 시 실행
            return strategy.on_kline_close_3m()
        except Exception as e:
            print(f"⚠️ 페이드 리입 전략 평가 오류: {e}")
            return None
    
    def _evaluate_squeeze_momentum_strategy(self, strategy: SqueezeMomentumStrategy, df: pd.DataFrame) -> Optional[Dict]:
        """스퀴즈 모멘텀 전략 평가 (웹소켓과 동일한 방식)"""
        try:
            # 스퀴즈 모멘텀 전략은 1분봉 기반이므로 1분봉 데이터로 평가
            df_1m = df.tail(1)  # 최근 1분봉
            return strategy.on_kline_close_1m(df_1m)
        except Exception as e:
            print(f"⚠️ 스퀴즈 모멘텀 전략 평가 오류: {e}")
            return None
    
    def _evaluate_session_strategy(self, strategy: SessionORLite, df: pd.DataFrame) -> Optional[Dict]:
        """세션 OR 전략 평가 (웹소켓과 동일한 방식)"""
        try:
            # 전략이 제대로 초기화되었는지 확인
            if strategy is None:
                print("⚠️ 세션 OR 전략이 초기화되지 않았습니다")
                return None
                
            # 웹소켓과 동일한 방식으로 3분봉 데이터와 세션 상태 전달
            df_3m = df.tail(2)  # 최근 2개 캔들
            session_activated = True  # 백테스팅에서는 항상 활성화된 것으로 가정
            return strategy.on_kline_close_3m(df_3m, session_activated)
        except Exception as e:
            print(f"⚠️ 세션 OR 전략 평가 오류: {e}")
            return None
    
    def _process_signal(self, signal: Dict, timestamp: datetime, candle: pd.Series, strategy_name: str):
        """신호 처리 및 거래 실행 (score/ confidence를 고려해서 거래 수락 여부 결정)
        - signal에 'score' 필드가 있으면 self.min_score 이상일 때만 실제 포지션을 오픈합니다.
        - 모든 신호는 trade_history에 기록되며, 'accepted' 플래그로 실제 오픈 여부를 구분합니다.
        - PnL은 목표(첫 번째 target) 또는 스탑에 도달했다고 가정한 단순 시뮬레이션을 사용합니다.
        """
        try:
            action = signal.get('action')
            entry_price = float(signal.get('entry')) if signal.get('entry') is not None else None
            stop_loss = float(signal.get('stop')) if signal.get('stop') is not None else None
            targets = signal.get('targets', []) or []
            score = float(signal.get('score')) if signal.get('score') is not None else None
            confidence = signal.get('confidence')
            components = signal.get('components', None)

            # always record the raw signal to trade_history for analysis (accepted may be False)
            sig_record = {
                'timestamp': timestamp,
                'strategy': strategy_name,
                'signal': signal,
                'score': score,
                'confidence': confidence,
                'components': components,
                'accepted': False,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'targets': targets,
                'quantity': None,
                'pnl': None,
                'status': 'RECORD'
            }
            self.trade_history.append(sig_record)

            # Basic validation
            if not all([action, entry_price, stop_loss]):
                return

            # Score filtering logic
            accepted = True
            if self.use_score_filter and (score is not None):
                if score < self.min_score:
                    accepted = False

            if not accepted:
                print(f"🚫 [{strategy_name}] signal rejected by score filter: score={score} min_score={self.min_score} conf={confidence}")
                return

            # Determine position size (10% of current capital)
            position_size = self.current_capital * 0.1
            quantity = position_size / entry_price if entry_price > 0 else 0.0

            # Simple execution simulation: assume entry fills at entry_price,
            # and outcome is determined by comparing first target to entry (or stop)
            pnl = 0.0
            executed = True
            if targets and len(targets) > 0:
                tp = float(targets[0])
                if action.upper() == 'BUY':
                    pnl = (tp - entry_price) * quantity
                else:
                    pnl = (entry_price - tp) * quantity
            else:
                # fallback: use stop (assume immediate stop hit => loss)
                if action.upper() == 'BUY':
                    pnl = (stop_loss - entry_price) * quantity
                else:
                    pnl = (entry_price - stop_loss) * quantity

            # record position/trade
            trade = {
                'timestamp': timestamp,
                'strategy': strategy_name,
                'action': action,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'targets': targets,
                'quantity': quantity,
                'status': 'CLOSED',  # we simulate closed immediately for simple backtest
                'pnl': float(pnl),
                'score': score,
                'confidence': confidence,
                'components': components
            }

            self.positions.append(trade)
            self.closed_trades.append(trade)
            self.current_capital += float(pnl)
            # update aggregate results
            self.results['total_trades'] += 1
            if pnl > 0:
                self.results['winning_trades'] += 1
            else:
                self.results['losing_trades'] += 1
            self.results['total_pnl'] += float(pnl)

            print(f"📊 [{strategy_name}] {action} 신호 ACCEPTED: 진입=${entry_price:.2f}, 손절=${stop_loss:.2f}, score={score}, conf={confidence}, pnl={pnl:.2f}")

        except Exception as e:
            print(f"⚠️ 신호 처리 오류: {e}")

    def _calculate_results(self):
        """백테스팅 결과 계산"""
        print("📊 백테스팅 결과 계산 중...")
        
        if not self.trade_history:
            print("⚠️ 거래 기록이 없습니다")
            return
        
        # 기본 통계
        self.results['total_trades'] = len(self.trade_history)
        
        # 수익률 계산
        total_pnl = 0.0
        winning_trades = 0
        losing_trades = 0
        
        for trade in self.trade_history:
            # 간단한 수익률 계산 (실제로는 더 복잡한 로직 필요)
            if trade['action'] == 'BUY':
                pnl = (trade.get('targets', [trade['entry_price']])[0] - trade['entry_price']) * trade['quantity']
            else:
                pnl = (trade['entry_price'] - trade.get('targets', [trade['entry_price']])[0]) * trade['quantity']
            
            total_pnl += pnl
            
            if pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1
        
        self.results['total_pnl'] = total_pnl
        self.results['winning_trades'] = winning_trades
        self.results['losing_trades'] = losing_trades
        self.results['win_rate'] = (winning_trades / self.results['total_trades']) * 100 if self.results['total_trades'] > 0 else 0
        
        # 최대 낙폭 계산
        peak_capital = self.initial_capital
        max_drawdown = 0.0
        
        for trade in self.trade_history:
            # 간단한 드로우다운 계산
            current_capital = self.initial_capital + trade.get('pnl', 0)
            if current_capital > peak_capital:
                peak_capital = current_capital
            
            drawdown = (peak_capital - current_capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)
        
        self.results['max_drawdown'] = max_drawdown * 100
        
        print("✅ 백테스팅 결과 계산 완료")
    
    def print_results(self):
        """백테스팅 결과 출력"""
        print("\n" + "="*60)
        print("📊 30일 백테스팅 결과")
        print("="*60)
        
        print(f"💰 초기 자본: ${self.initial_capital:,.2f}")
        print(f"📈 총 수익: ${self.results['total_pnl']:,.2f}")
        print(f"📊 총 거래 수: {self.results['total_trades']}")
        print(f"✅ 승리 거래: {self.results['winning_trades']}")
        print(f"❌ 패배 거래: {self.results['losing_trades']}")
        print(f"🎯 승률: {self.results['win_rate']:.1f}%")
        print(f"📉 최대 낙폭: {self.results['max_drawdown']:.1f}%")
        
        # 연간 수익률 (30일 기준으로 추정)
        annual_return = (self.results['total_pnl'] / self.initial_capital) * (365 / 30) * 100
        print(f"📅 연간 수익률 (추정): {annual_return:.1f}%")
        
        print("="*60)
    
    def save_results(self, filename: str = "backtest_results.json"):
        """결과를 JSON 파일로 저장"""
        try:
            # datetime 객체를 문자열로 변환
            results_copy = self.results.copy()
            results_copy['timestamp'] = datetime.now().isoformat()
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results_copy, f, indent=2, ensure_ascii=False)
            
            print(f"💾 백테스팅 결과가 {filename}에 저장되었습니다")
            
        except Exception as e:
            print(f"❌ 결과 저장 오류: {e}")


def main():
    """메인 함수"""
    print("🚀 30일 백테스팅 시스템 시작")
    
    # 백테스팅 엔진 초기화
    engine = BacktestEngine(symbol="ETHUSDT", initial_capital=10000.0)
    
    # 과거 데이터 로드
    df = engine.load_historical_data(days_back=30)
    
    if df.empty:
        print("❌ 데이터를 로드할 수 없습니다")
        return
    
    # 백테스팅 실행
    results = engine.run_backtest(df)
    
    # 결과 출력
    engine.print_results()
    
    # 결과 저장
    engine.save_results()
    
    print("🎉 백테스팅 완료!")


if __name__ == "__main__":
    main()
