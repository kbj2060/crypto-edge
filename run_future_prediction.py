#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
미래 예측 그래프 실행 스크립트
- 3분봉 데이터와 단기/중기/장기 전략 예측을 활용한 미래 가격 움직임 시각화
- 실시간 업데이트 가능
"""

import sys
import os
import argparse
from datetime import datetime, timezone

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.future_predictor import FuturePredictor, create_future_prediction_demo
from utils.integrated_predictor import IntegratedPredictor
from utils.realtime_dashboard import RealtimeDashboard
from data.binance_dataloader import BinanceDataLoader
from core.trader_core import TraderCore
from config.integrated_config import IntegratedConfig

def run_demo():
    """데모 실행"""
    print("🚀 미래 예측 그래프 데모 실행...")
    create_future_prediction_demo()

def run_integrated_predictor(use_data_loader=False):
    """통합 예측 시스템 실행"""
    print("🚀 통합 미래 예측 시스템 실행...")
    
    predictor = IntegratedPredictor("ETHUSDC")
    
    # 콜백 추가
    def on_prediction_updated(data):
        print(f"\n📊 예측 업데이트: {len(data['predictions'])}개 포인트")
        
        # 요약 정보 출력
        summary = predictor.get_prediction_summary()
        print(f"   💰 현재 가격: ${summary['price_range']['current']:.2f}")
        print(f"   📈 예측 범위: ${summary['price_range']['min']:.2f} ~ ${summary['price_range']['max']:.2f}")
        print(f"   🎯 평균 신뢰도: {summary['confidence_stats']['avg']:.2f}")
        
        # 전략별 분석
        for strategy, analysis in summary['strategy_analysis'].items():
            print(f"   📊 {strategy}: {analysis['count']}개 예측, 신뢰도 {analysis['avg_confidence']:.2f}")
    
    predictor.add_callback('prediction_updated', on_prediction_updated)
    
    if use_data_loader:
        predictor.run_with_data_loader()
    else:
        predictor.run_manual_mode()

def run_dashboard():
    """웹 대시보드 실행"""
    print("🚀 실시간 미래 예측 대시보드 실행...")
    
    dashboard = RealtimeDashboard("ETHUSDC")
    dashboard.run(host='0.0.0.0', port=5000, debug=False)

def run_historical_analysis(hours=24):
    """과거 데이터 분석"""
    print(f"📊 과거 {hours}시간 데이터 분석...")
    
    # 데이터 로더로 과거 데이터 가져오기
    data_loader = BinanceDataLoader()
    df = data_loader.fetch_recent_3m("ETHUSDC", hours)
    
    if df is None or df.empty:
        print("❌ 데이터를 가져올 수 없습니다.")
        return
    
    print(f"✅ 데이터 로드 완료: {len(df)}개 캔들")
    print(f"   📅 기간: {df.index[0]} ~ {df.index[-1]}")
    print(f"   💰 가격 범위: ${df['low'].min():.2f} ~ ${df['high'].max():.2f}")
    
    # 예측기 생성 및 데이터 추가
    predictor = FuturePredictor()
    predictor.add_historical_data(df)
    
    # 샘플 신호 생성 (실제로는 전략에서 가져와야 함)
    sample_signals = {
        'VWAP_PINBALL': {'action': 'BUY', 'score': 0.8},
        'LIQUIDITY_GRAB': {'action': 'BUY', 'score': 0.7},
        'HTF_TREND': {'action': 'BUY', 'score': 0.9},
        'SUPPORT_RESISTANCE': {'action': 'BUY', 'score': 0.6},
        'OI_DELTA': {'action': 'SELL', 'score': 0.8},
        'VPVR': {'action': 'BUY', 'score': 0.7}
    }
    
    # 예측 생성
    current_price = df['close'].iloc[-1]
    predictions = predictor.generate_predictions(sample_signals, current_price)
    
    # 차트 생성 및 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'historical_prediction_{timestamp}.png'
    predictor.save_prediction_graph(filename, hours_ahead=12)
    
    # 요약 정보 출력
    summary = predictor.get_prediction_summary()
    print(f"\n📊 예측 요약:")
    print(f"   총 예측 포인트: {summary['total_predictions']}개")
    print(f"   신뢰도 범위: {summary['confidence_stats']['min']:.2f} ~ {summary['confidence_stats']['max']:.2f}")
    print(f"   가격 범위: ${summary['price_range']['min']:.2f} ~ ${summary['price_range']['max']:.2f}")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='미래 예측 그래프 실행 도구')
    parser.add_argument('mode', choices=['demo', 'predictor', 'dashboard', 'historical'], 
                       help='실행 모드 선택')
    parser.add_argument('--data-loader', action='store_true', 
                       help='데이터 로더 모드 사용 (predictor 모드에서만)')
    parser.add_argument('--hours', type=int, default=24, 
                       help='분석할 시간 (historical 모드에서만)')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'demo':
            run_demo()
        elif args.mode == 'predictor':
            run_integrated_predictor(use_data_loader=args.data_loader)
        elif args.mode == 'dashboard':
            run_dashboard()
        elif args.mode == 'historical':
            run_historical_analysis(hours=args.hours)
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중지됨")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
