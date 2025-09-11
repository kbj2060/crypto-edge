#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
통합 미래 예측 시스템
- 웹소켓과 연동하여 실시간으로 미래 예측을 생성하고 업데이트
- 3분봉 데이터와 전략 신호를 실시간으로 분석하여 미래 그래프 생성
"""

import asyncio
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# 프로젝트 컴포넌트들
from utils.future_predictor import FuturePredictor, PredictionPoint
from data.strategy_executor import StrategyExecutor
from engines.trade_decision_engine import TradeDecisionEngine
from data.data_manager import get_data_manager
from utils.display_utils import print_decision_interpretation
from core.trader_core import TraderCore
from config.integrated_config import IntegratedConfig

class IntegratedPredictor:
    """통합 미래 예측 시스템"""
    
    def __init__(self, symbol: str = "ETHUSDC"):
        self.symbol = symbol
        
        # 핵심 컴포넌트들
        self.predictor = FuturePredictor()
        self.strategy_executor = StrategyExecutor()
        self.decision_engine = TradeDecisionEngine()
        self.data_manager = get_data_manager()
        
        # TraderCore를 통한 웹소켓 관리
        self.trader_core = None
        self.websocket = None
        self.use_websocket = False
        
        # 예측 데이터 저장소
        self.current_predictions = []
        self.historical_predictions = []
        self.prediction_history = []
        
        # 업데이트 상태
        self.is_running = False
        self.last_update_time = None
        
        # 콜백 함수들
        self.callbacks = {
            'prediction_updated': [],
            'new_signal': [],
            'market_context_changed': []
        }
    
    def add_callback(self, event_type: str, callback):
        """콜백 함수 등록"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback):
        """콜백 함수 제거"""
        if event_type in self.callbacks:
            if callback in self.callbacks[event_type]:
                self.callbacks[event_type].remove(callback)
    
    def _execute_callbacks(self, event_type: str, data: Any = None):
        """콜백 함수 실행"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                print(f"❌ 콜백 실행 오류 ({event_type}): {e}")
    
    def start_data_loader(self):
        """데이터 로더 시작"""
        self.use_websocket = False
        
        try:
            # 1. DataManager 초기화
            from data.data_manager import get_data_manager
            data_manager = get_data_manager()
            data_loaded = data_manager.load_initial_data(self.symbol)
            
            if not data_loaded:
                print("❌ DataManager 초기 데이터 로딩 실패")
                return
            
            # 2. 글로벌 지표 초기화
            from indicators.global_indicators import get_global_indicator_manager
            global_manager = get_global_indicator_manager()
            global_manager.initialize_indicators()
            
            # 3. BinanceDataLoader 초기화
            from data.binance_dataloader import BinanceDataLoader
            self.data_loader = BinanceDataLoader()
            
            print(f"🌐 데이터 로더 시작됨: {self.symbol}")
            
        except Exception as e:
            print(f"❌ 데이터 로더 초기화 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_and_update_predictions(self):
        """데이터 로드 및 예측 업데이트"""
        try:
            # 최근 24시간 3분봉 데이터 로드
            df = self.data_loader.fetch_recent_3m(self.symbol, 24)
            
            if df is None or df.empty:
                print("❌ 데이터 로드 실패")
                return
            
            # 현재 가격
            current_price = df['close'].iloc[-1]
            
            # 과거 데이터를 예측기에 추가
            self.predictor.add_historical_data(df)
            
            # 전략 신호 생성
            self.strategy_executor.execute_all_strategies()
            signals = self.strategy_executor.get_signals()
            print(signals)
            # 예측 업데이트
            self.update_predictions(signals, {'close': current_price})
            
            # 콜백 실행
            self._execute_callbacks('new_signal', signals)
            
            print(f"✅ 데이터 로드 및 예측 업데이트 완료: ${current_price:.2f}")
            
        except Exception as e:
            print(f"❌ 데이터 로드 및 예측 업데이트 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def update_predictions(self, signals: Dict[str, Any], price_data: Dict = None):
        """예측 데이터 업데이트"""
        try:
            # 현재 가격 결정
            if price_data:
                current_price = price_data.get('close', 0.0)
            else:
                # 데이터 매니저에서 최신 가격 가져오기
                latest_data = self.data_manager.get_latest_data()
                current_price = latest_data.get('close', 3000.0) if latest_data else 3000.0
            
            # 과거 데이터 추가 (데이터 매니저에서)
            historical_df = self.data_manager.get_historical_data()
            if not historical_df.empty:
                self.predictor.add_historical_data(historical_df)
            
            # 미래 예측 생성
            new_predictions = self.predictor.generate_predictions(signals, current_price)
            
            # 예측 데이터 업데이트
            self.current_predictions = new_predictions
            self.last_update_time = datetime.now(timezone.utc)
            
            # 예측 히스토리에 추가 (최근 100개만 유지)
            self.prediction_history.append({
                'timestamp': self.last_update_time,
                'predictions': new_predictions.copy(),
                'signals': signals.copy(),
                'current_price': current_price
            })
            
            if len(self.prediction_history) > 100:
                self.prediction_history.pop(0)
            
            # 콜백 실행
            self._execute_callbacks('prediction_updated', {
                'predictions': new_predictions,
                'signals': signals,
                'current_price': current_price,
                'timestamp': self.last_update_time
            })
            
            print(f"✅ 예측 업데이트 완료: {len(new_predictions)}개 예측 포인트")
            
        except Exception as e:
            print(f"❌ 예측 업데이트 오류: {e}")
    
    def get_current_predictions(self) -> List[PredictionPoint]:
        """현재 예측 데이터 반환"""
        return self.current_predictions.copy()
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """예측 요약 정보 반환"""
        if not self.current_predictions:
            return {'message': '예측 데이터가 없습니다.'}
        
        # 전략별 분석
        strategy_analysis = {}
        for pred in self.current_predictions:
            strategy = pred.strategy_type
            if strategy not in strategy_analysis:
                strategy_analysis[strategy] = {
                    'count': 0,
                    'avg_confidence': 0,
                    'actions': {'BUY': 0, 'SELL': 0, 'HOLD': 0},
                    'avg_net_score': 0
                }
            
            strategy_analysis[strategy]['count'] += 1
            strategy_analysis[strategy]['actions'][pred.action] += 1
        
        # 평균값 계산
        for strategy, data in strategy_analysis.items():
            strategy_preds = [p for p in self.current_predictions if p.strategy_type == strategy]
            data['avg_confidence'] = np.mean([p.confidence for p in strategy_preds])
            data['avg_net_score'] = np.mean([p.net_score for p in strategy_preds])
        
        # 가격 범위
        prices = [p.price for p in self.current_predictions]
        current_price = self.current_predictions[0].price if self.current_predictions else 0
        
        return {
            'timestamp': self.last_update_time.isoformat() if self.last_update_time else None,
            'total_predictions': len(self.current_predictions),
            'strategy_analysis': strategy_analysis,
            'price_range': {
                'min': min(prices) if prices else 0,
                'max': max(prices) if prices else 0,
                'current': current_price
            },
            'confidence_stats': {
                'min': min([p.confidence for p in self.current_predictions]) if self.current_predictions else 0,
                'max': max([p.confidence for p in self.current_predictions]) if self.current_predictions else 0,
                'avg': np.mean([p.confidence for p in self.current_predictions]) if self.current_predictions else 0
            }
        }
    
    def create_prediction_chart(self, hours_ahead: int = 24, save_path: str = None):
        """예측 차트 생성"""
        if not self.current_predictions:
            print("❌ 예측 데이터가 없습니다.")
            return None
        
        # 차트 생성
        fig = self.predictor.create_future_graph(hours_ahead=hours_ahead)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 차트 저장됨: {save_path}")
        
        return fig
    
    def start_manual_mode(self):
        """수동 모드 시작 (데이터 로더 사용)"""
        print("🔧 수동 모드 시작...")
        self.is_running = True
        
        # 데이터 로더 시작
        self.start_data_loader()
        
        def manual_update_loop():
            while self.is_running:
                try:
                    # 데이터 로드 및 예측 업데이트
                    self._load_and_update_predictions()
                    
                    # 3분 대기
                    time.sleep(180)
                    
                except Exception as e:
                    print(f"❌ 수동 모드 업데이트 오류: {e}")
                    time.sleep(60)  # 오류 시 1분 대기
        
        update_thread = threading.Thread(target=manual_update_loop, daemon=True)
        update_thread.start()
    
    def stop(self):
        """시스템 중지"""
        self.is_running = False
        if self.trader_core:
            self.trader_core.stop_websocket()
        print("🛑 통합 예측 시스템 중지됨")
    
    def run_with_data_loader(self):
        """데이터 로더와 함께 실행"""
        print("🚀 통합 미래 예측 시스템 시작 (데이터 로더 모드)")
        self.is_running = True
        self.start_data_loader()
        
        try:
            # 메인 루프
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중지됨")
        finally:
            self.stop()
    
    def run_manual_mode(self):
        """수동 모드로 실행"""
        print("🚀 통합 미래 예측 시스템 시작 (수동 모드)")
        self.start_manual_mode()
        
        try:
            # 메인 루프
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중지됨")
        finally:
            self.stop()

def create_prediction_demo():
    """예측 시스템 데모"""
    print("🚀 미래 예측 시스템 데모 시작...")
    
    # 통합 예측 시스템 생성
    predictor = IntegratedPredictor("ETHUSDC")
    
    # 콜백 추가
    def on_prediction_updated(data):
        print(f"📊 예측 업데이트: {len(data['predictions'])}개 포인트")
        
        # 요약 정보 출력
        summary = predictor.get_prediction_summary()
        print(f"   💰 현재 가격: ${summary['price_range']['current']:.2f}")
        print(f"   📈 예측 범위: ${summary['price_range']['min']:.2f} ~ ${summary['price_range']['max']:.2f}")
        print(f"   🎯 평균 신뢰도: {summary['confidence_stats']['avg']:.2f}")
    
    predictor.add_callback('prediction_updated', on_prediction_updated)
    
    # 수동 모드로 실행
    predictor.run_manual_mode()

if __name__ == "__main__":
    create_prediction_demo()
