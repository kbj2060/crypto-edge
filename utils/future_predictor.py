#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
미래 예측 그래프 시각화 도구
- 3분봉 데이터와 단기/중기/장기 전략 예측을 활용한 미래 가격 움직임 시각화
- 실시간 업데이트 가능한 대시보드
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
import seaborn as sns
from dataclasses import dataclass

# 전략 엔진들 import
from engines.short_term_synergy_engine import ShortTermSynergyEngine
from engines.medium_term_synergy_engine import MediumTermSynergyEngine  
from engines.long_term_synergy_engine import LongTermSynergyEngine

@dataclass
class PredictionPoint:
    """예측 포인트 데이터"""
    timestamp: datetime
    price: float
    confidence: float
    strategy_type: str  # 'SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM'
    action: str  # 'BUY', 'SELL', 'HOLD'
    market_context: str
    net_score: float

class FuturePredictor:
    """미래 예측 그래프 생성기"""
    
    def __init__(self):
        self.short_engine = ShortTermSynergyEngine()
        self.medium_engine = MediumTermSynergyEngine()
        self.long_engine = LongTermSynergyEngine()
        
        # 예측 데이터 저장
        self.historical_data = pd.DataFrame()
        self.predictions = []
        
        # 그래프 스타일 설정
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
    def add_historical_data(self, df: pd.DataFrame):
        """과거 3분봉 데이터 추가"""
        if not df.empty:
            self.historical_data = df.copy()
    
    def generate_predictions(self, signals: Dict[str, Any], current_price: float) -> List[PredictionPoint]:
        """전략 신호를 바탕으로 미래 예측 생성"""
        predictions = []
        current_time = datetime.now(timezone.utc)
        
        # 각 전략별로 예측 생성
        strategies = [
            ('SHORT_TERM', self.short_engine, 1),  # 1시간 예측
            ('MEDIUM_TERM', self.medium_engine, 4),  # 4시간 예측  
            ('LONG_TERM', self.long_engine, 24)  # 24시간 예측
        ]
        
        for strategy_name, engine, hours_ahead in strategies:
            try:
                # 해당 전략의 신호만 필터링
                strategy_signals = self._filter_strategy_signals(signals, strategy_name)
                
                if not strategy_signals:
                    continue
                    
                # 시너지 점수 계산
                result = engine.calculate_synergy_score(strategy_signals)
                
                if result['action'] == 'HOLD':
                    continue
                
                # 예측 포인트들 생성 (시간 간격별)
                time_interval = 3  # 3분 간격
                num_points = (hours_ahead * 60) // time_interval
                
                for i in range(1, num_points + 1):
                    pred_time = current_time + timedelta(minutes=i * time_interval)
                    
                    # 가격 예측 (간단한 선형 추정)
                    price_change_pct = self._calculate_price_change_pct(result, i, hours_ahead)
                    predicted_price = current_price * (1 + price_change_pct)
                    
                    # 신뢰도 감소 (시간이 지날수록)
                    confidence_decay = max(0.3, 1.0 - (i / num_points) * 0.7)
                    
                    # confidence가 시퀀스인 경우 처리
                    confidence = result.get('confidence', 0.5)
                    if isinstance(confidence, (list, tuple)):
                        confidence = confidence[0] if confidence else 0.5
                    
                    try:
                        confidence = float(confidence)
                    except (ValueError, TypeError):
                        confidence = 0.5
                    
                    final_confidence = confidence * confidence_decay
                    
                    prediction = PredictionPoint(
                        timestamp=pred_time,
                        price=predicted_price,
                        confidence=final_confidence,
                        strategy_type=strategy_name,
                        action=result['action'],
                        market_context=result.get('market_context', 'UNKNOWN'),
                        net_score=result['net_score']
                    )
                    predictions.append(prediction)
                    
            except Exception as e:
                print(f"❌ {strategy_name} 예측 생성 오류: {e}")
                continue
        
        self.predictions = predictions
        return predictions
    
    def _filter_strategy_signals(self, signals: Dict[str, Any], strategy_type: str) -> Dict[str, Any]:
        """전략 타입별 신호 필터링"""
        strategy_mapping = {
            'SHORT_TERM': ['VWAP_PINBALL', 'LIQUIDITY_GRAB', 'ZSCORE_MEAN_REVERSION', 'VOL_SPIKE', 'ORDERFLOW_CVD'],
            'MEDIUM_TERM': ['HTF_TREND', 'MULTI_TIMEFRAME', 'SUPPORT_RESISTANCE', 'EMA_CONFLUENCE', 'BOLLINGER_SQUEEZE'],
            'LONG_TERM': ['OI_DELTA', 'VPVR', 'ICHIMOKU', 'FUNDING_RATE']
        }
        
        target_strategies = strategy_mapping.get(strategy_type, [])
        filtered = {}
        
        for name, signal_data in signals.items():
            if name in target_strategies:
                filtered[name] = signal_data
                
        return filtered
    
    def _calculate_price_change_pct(self, result: Dict[str, Any], point_index: int, total_hours: int) -> float:
        """가격 변화율 계산"""
        net_score = result.get('net_score', 0.0)
        action = result.get('action', 'HOLD')
        
        # net_score가 시퀀스인 경우 첫 번째 요소 사용
        if isinstance(net_score, (list, tuple)):
            net_score = net_score[0] if net_score else 0.0
        
        # net_score를 float로 변환
        try:
            net_score = float(net_score)
        except (ValueError, TypeError):
            net_score = 0.0
        
        # point_index와 total_hours를 int로 변환
        try:
            point_index = int(point_index)
            total_hours = int(total_hours)
        except (ValueError, TypeError):
            point_index = 0
            total_hours = 24
        
        # 기본 변화율 (net_score 기반)
        base_change = float(net_score) * 0.02  # 2% per unit score
        
        # 시간에 따른 감소
        time_decay = 1.0 - (point_index / (total_hours * 20))  # 20 = 3분봉 per hour
        
        # 액션에 따른 방향
        direction = 1 if action == 'BUY' else -1 if action == 'SELL' else 0
        
        return base_change * time_decay * direction
    
    def create_future_graph(self, 
                          hours_ahead: int = 24,
                          show_historical: bool = True,
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """미래 예측 그래프 생성"""
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('🚀 미래 가격 예측 대시보드', fontsize=16, fontweight='bold')
        
        # 1. 메인 가격 차트
        ax1 = axes[0, 0]
        self._plot_main_price_chart(ax1, hours_ahead, show_historical)
        
        # 2. 전략별 예측 신호
        ax2 = axes[0, 1] 
        self._plot_strategy_signals(ax2, hours_ahead)
        
        # 3. 신뢰도 히트맵
        ax3 = axes[1, 0]
        self._plot_confidence_heatmap(ax3, hours_ahead)
        
        # 4. 시장 상황 분석
        ax4 = axes[1, 1]
        self._plot_market_context(ax4)
        
        plt.tight_layout()
        return fig
    
    def _plot_main_price_chart(self, ax, hours_ahead: int, show_historical: bool):
        """메인 가격 차트 그리기"""
        ax.set_title('📈 가격 예측 차트', fontweight='bold')
        
        current_time = datetime.now(timezone.utc)
        future_time = current_time + timedelta(hours=hours_ahead)
        
        # 과거 데이터
        if show_historical and not self.historical_data.empty:
            hist_data = self.historical_data.tail(100)  # 최근 100개 캔들
            ax.plot(hist_data.index, hist_data['close'], 
                   color='blue', alpha=0.7, linewidth=1, label='과거 가격')
        
        # 현재 가격 라인
        if not self.historical_data.empty:
            current_price = self.historical_data['close'].iloc[-1]
            ax.axhline(y=current_price, color='red', linestyle='--', alpha=0.8, label='현재 가격')
        
        # 예측 데이터
        if self.predictions:
            pred_df = pd.DataFrame([(p.timestamp, p.price, p.strategy_type, p.confidence, p.action) 
                                  for p in self.predictions],
                                 columns=['timestamp', 'price', 'strategy_type', 'confidence', 'action'])
            
            # 전략별로 색상 구분
            colors = {'SHORT_TERM': 'green', 'MEDIUM_TERM': 'orange', 'LONG_TERM': 'purple'}
            
            for strategy_type in ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']:
                strategy_data = pred_df[pred_df['strategy_type'] == strategy_type]
                if not strategy_data.empty:
                    # 신뢰도에 따른 투명도 조정
                    alpha_values = strategy_data['confidence'].values
                    ax.scatter(strategy_data['timestamp'], strategy_data['price'], 
                             c=colors[strategy_type], alpha=alpha_values, s=30, 
                             label=f'{strategy_type} 예측')
            
            # 예측 트렌드 라인
            pred_df_sorted = pred_df.sort_values('timestamp')
            ax.plot(pred_df_sorted['timestamp'], pred_df_sorted['price'], 
                   color='red', alpha=0.6, linewidth=2, label='예측 트렌드')
        
        ax.set_xlabel('시간')
        ax.set_ylabel('가격 (USDC)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 시간축 포맷팅
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_strategy_signals(self, ax, hours_ahead: int):
        """전략별 신호 차트"""
        ax.set_title('🎯 전략별 신호 강도', fontweight='bold')
        
        if not self.predictions:
            ax.text(0.5, 0.5, '예측 데이터 없음', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 전략별 데이터 그룹화
        strategy_data = {}
        for pred in self.predictions:
            if pred.strategy_type not in strategy_data:
                strategy_data[pred.strategy_type] = []
            strategy_data[pred.strategy_type].append(pred)
        
        # 각 전략별로 신호 강도 플롯
        y_pos = 0
        colors = {'SHORT_TERM': 'green', 'MEDIUM_TERM': 'orange', 'LONG_TERM': 'purple'}
        
        for strategy_type, preds in strategy_data.items():
            timestamps = [p.timestamp for p in preds]
            scores = [p.net_score for p in preds]
            confidences = [p.confidence for p in preds]
            
            # 신뢰도에 따른 크기 조정
            sizes = [c * 100 for c in confidences]
            
            ax.scatter(timestamps, [y_pos] * len(timestamps), 
                      c=colors[strategy_type], s=sizes, alpha=0.7, 
                      label=f'{strategy_type} (신뢰도 기반 크기)')
            
            # 신호 강도 라인
            ax.plot(timestamps, [y_pos + s * 0.3 for s in scores], 
                   color=colors[strategy_type], alpha=0.5, linewidth=2)
            
            y_pos += 1
        
        ax.set_xlabel('시간')
        ax.set_ylabel('전략')
        ax.set_yticks(range(len(strategy_data)))
        ax.set_yticklabels(strategy_data.keys())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_confidence_heatmap(self, ax, hours_ahead: int):
        """신뢰도 히트맵"""
        ax.set_title('🔥 예측 신뢰도 히트맵', fontweight='bold')
        
        if not self.predictions:
            ax.text(0.5, 0.5, '예측 데이터 없음', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 시간별, 전략별 신뢰도 매트릭스 생성
        strategies = ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']
        time_slots = pd.date_range(
            start=datetime.now(timezone.utc),
            end=datetime.now(timezone.utc) + timedelta(hours=hours_ahead),
            freq='3min'
        )
        
        confidence_matrix = np.zeros((len(strategies), len(time_slots)))
        
        for pred in self.predictions:
            strategy_idx = strategies.index(pred.strategy_type)
            time_idx = time_slots.get_indexer([pred.timestamp], method='nearest')[0]
            if 0 <= time_idx < len(time_slots):
                confidence_matrix[strategy_idx, time_idx] = pred.confidence
        
        # 히트맵 그리기
        im = ax.imshow(confidence_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # 축 설정
        ax.set_xticks(range(0, len(time_slots), len(time_slots)//8))
        ax.set_xticklabels([time_slots[i].strftime('%H:%M') for i in range(0, len(time_slots), len(time_slots)//8)])
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies)
        
        # 컬러바
        plt.colorbar(im, ax=ax, label='신뢰도')
        
        ax.set_xlabel('시간')
        ax.set_ylabel('전략')
    
    def _plot_market_context(self, ax):
        """시장 상황 분석"""
        ax.set_title('🌍 시장 상황 분석', fontweight='bold')
        
        if not self.predictions:
            ax.text(0.5, 0.5, '예측 데이터 없음', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 시장 상황별 카운트
        context_counts = {}
        for pred in self.predictions:
            context = pred.market_context
            if context not in context_counts:
                context_counts[context] = 0
            context_counts[context] += 1
        
        if context_counts:
            # 파이 차트
            labels = list(context_counts.keys())
            sizes = list(context_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            
            # 텍스트 스타일링
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax.text(0.5, 0.5, '시장 상황 데이터 없음', ha='center', va='center', transform=ax.transAxes)
    
    def save_prediction_graph(self, filename: str = None, hours_ahead: int = 24):
        """예측 그래프 저장"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'future_prediction_{timestamp}.png'
        
        fig = self.create_future_graph(hours_ahead=hours_ahead)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ 예측 그래프 저장됨: {filename}")
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """예측 요약 정보 반환"""
        if not self.predictions:
            return {'message': '예측 데이터가 없습니다.', 'total_predictions': 0}
        
        summary = {
            'total_predictions': len(self.predictions),
            'strategy_breakdown': {},
            'confidence_stats': {},
            'price_range': {},
            'market_contexts': {}
        }
        
        # 전략별 분석
        for pred in self.predictions:
            strategy = pred.strategy_type
            if strategy not in summary['strategy_breakdown']:
                summary['strategy_breakdown'][strategy] = {
                    'count': 0, 'avg_confidence': 0, 'actions': {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                }
            
            summary['strategy_breakdown'][strategy]['count'] += 1
            summary['strategy_breakdown'][strategy]['actions'][pred.action] += 1
        
        # 신뢰도 통계
        confidences = [p.confidence for p in self.predictions]
        summary['confidence_stats'] = {
            'min': min(confidences),
            'max': max(confidences),
            'avg': np.mean(confidences),
            'std': np.std(confidences)
        }
        
        # 가격 범위
        prices = [p.price for p in self.predictions]
        summary['price_range'] = {
            'min': min(prices),
            'max': max(prices),
            'current': self.historical_data['close'].iloc[-1] if not self.historical_data.empty else 0
        }
        
        # 시장 상황
        contexts = [p.market_context for p in self.predictions]
        summary['market_contexts'] = dict(pd.Series(contexts).value_counts())
        
        return summary

def create_future_prediction_demo():
    """미래 예측 데모 실행"""
    print("🚀 미래 예측 그래프 데모 시작...")
    
    # 예측기 생성
    predictor = FuturePredictor()
    
    # 샘플 과거 데이터 생성 (실제로는 BinanceDataLoader 사용)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='3min')
    prices = 3000 + np.cumsum(np.random.randn(100) * 10)
    
    sample_data = pd.DataFrame({
        'open': prices,
        'high': prices + np.random.rand(100) * 5,
        'low': prices - np.random.rand(100) * 5,
        'close': prices + np.random.randn(100) * 2,
        'volume': np.random.rand(100) * 1000
    }, index=dates)
    
    predictor.add_historical_data(sample_data)
    
    # 샘플 신호 생성
    sample_signals = {
        'VWAP_PINBALL': {'action': 'BUY', 'score': 0.8},
        'LIQUIDITY_GRAB': {'action': 'BUY', 'score': 0.7},
        'HTF_TREND': {'action': 'BUY', 'score': 0.9},
        'SUPPORT_RESISTANCE': {'action': 'BUY', 'score': 0.6},
        'OI_DELTA': {'action': 'SELL', 'score': 0.8},
        'VPVR': {'action': 'BUY', 'score': 0.7}
    }
    
    # 예측 생성
    current_price = sample_data['close'].iloc[-1]
    predictions = predictor.generate_predictions(sample_signals, current_price)
    
    # 그래프 생성 및 저장
    predictor.save_prediction_graph('demo_future_prediction.png', hours_ahead=12)
    
    # 요약 정보 출력
    summary = predictor.get_prediction_summary()
    print("\n📊 예측 요약:")
    print(f"총 예측 포인트: {summary['total_predictions']}개")
    print(f"신뢰도 범위: {summary['confidence_stats']['min']:.2f} ~ {summary['confidence_stats']['max']:.2f}")
    print(f"가격 범위: ${summary['price_range']['min']:.2f} ~ ${summary['price_range']['max']:.2f}")
    
    print("✅ 데모 완료!")

if __name__ == "__main__":
    create_future_prediction_demo()
