"""
강화학습 에이전트와 멀티프레임 트랜스포머 모델 예측 예시
- Signal 데이터만 입력하면 두 모델의 예측 결과를 비교
- 실시간 예측 및 성능 분석
"""

import numpy as np
import pandas as pd
import torch
import random
import os
from datetime import datetime
from typing import Dict, List, Optional
import json

# RL Training System에서 필요한 클래스들 import
from rl_training_system import RLAgent, TradingEnvironment, DataLoader
from multitimeframe_transformer import MultiTimeframeDecisionEngine, DecisionDataLoader

class PredictionComparison:
    """RL 에이전트와 Multi-Timeframe Transformer 예측 비교 클래스"""
    
    def __init__(self, 
                 rl_model_path: str = None,
                 transformer_model_path: str = None):
        
        print("🚀 예측 시스템 초기화 중...")
        
        # 1. RL 에이전트 초기화
        print("\n1️⃣ RL 에이전트 초기화...")
        self.rl_agent = RLAgent(state_size=111)  # 111차원
        
        # RL 모델 로드
        if rl_model_path and os.path.exists(rl_model_path):
            if self.rl_agent.load_model(rl_model_path):
                print(f"✅ RL 모델 로드 성공: {rl_model_path}")
            else:
                print(f"⚠️ RL 모델 로드 실패, 새 모델 사용")
        else:
            print("⚠️ RL 모델 파일이 없어 새 모델 사용")
        
        # 2. Multi-Timeframe Transformer 초기화
        print("\n2️⃣ Multi-Timeframe Transformer 초기화...")
        self.transformer_engine = MultiTimeframeDecisionEngine(
            model_path=transformer_model_path,
            input_size=58,
            d_model=256,
            nhead=8,
            num_layers=6
        )
        
        # Transformer 모델 로드
        if transformer_model_path and os.path.exists(transformer_model_path):
            if self.transformer_engine.load_model(transformer_model_path):
                print(f"✅ Transformer 모델 로드 성공: {transformer_model_path}")
            else:
                print(f"⚠️ Transformer 모델 로드 실패, 새 모델 사용")
        else:
            print("⚠️ Transformer 모델 파일이 없어 새 모델 사용")
        
        # 3. 데이터 로더 초기화
        self.data_loader = DataLoader()
        
        print("\n✅ 예측 시스템 초기화 완료!")
    
    def predict_from_signal(self, signal_data: Dict) -> Dict:
        """단일 Signal 데이터로 두 모델의 예측 수행"""
        
        print(f"\n🔮 Signal 데이터 예측 중...")
        print(f"   타임스탬프: {signal_data.get('timestamp', 'N/A')}")
        print(f"   가격: {signal_data.get('close', 0.0):.2f}")
        
        # 1. RL 에이전트 예측
        print("\n📊 RL 에이전트 예측...")
        rl_prediction = self._predict_with_rl_agent(signal_data)
        
        # 2. Multi-Timeframe Transformer 예측
        print("\n🧠 Multi-Timeframe Transformer 예측...")
        transformer_prediction = self._predict_with_transformer(signal_data)
        
        # 3. 예측 결과 비교
        comparison = self._compare_predictions(rl_prediction, transformer_prediction)
        
        return {
            'signal_data': signal_data,
            'rl_prediction': rl_prediction,
            'transformer_prediction': transformer_prediction,
            'comparison': comparison,
            'timestamp': datetime.now().isoformat()
        }
    
    def _predict_with_rl_agent(self, signal_data: Dict) -> Dict:
        """RL 에이전트로 예측"""
        try:
            # Signal 데이터를 RL 환경 형태로 변환
            signal_list = [signal_data]
            env = TradingEnvironment(signal_list)
            
            # 환경 초기화
            state, _ = env.reset()
            
            # RL 에이전트 예측
            self.rl_agent.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.rl_agent.device)
                q_values = self.rl_agent.q_network(state_tensor)
                action = torch.argmax(q_values).item()
            
            # 액션 해석
            action_names = ['HOLD', 'BUY', 'SELL']
            action_name = action_names[action]
            
            # Q값 분석
            q_values_np = q_values.cpu().numpy()[0]
            confidence = float(torch.softmax(q_values, dim=1)[0][action].item())
            
            return {
                'action': action_name,
                'action_index': action,
                'q_values': q_values_np.tolist(),
                'confidence': confidence,
                'model_type': 'RL Agent (DuelingDQN)',
                'state_dimension': len(state)
            }
            
        except Exception as e:
            print(f"❌ RL 에이전트 예측 실패: {e}")
            return {
                'action': 'HOLD',
                'action_index': 0,
                'q_values': [0.0, 0.0, 0.0],
                'confidence': 0.33,
                'model_type': 'RL Agent (Error)',
                'error': str(e)
            }
    
    def _predict_with_transformer(self, signal_data: Dict) -> Dict:
        """Multi-Timeframe Transformer로 예측"""
        try:
            # Transformer 엔진으로 예측
            prediction = self.transformer_engine.make_decision(signal_data)
            
            return {
                'action': prediction['action'],
                'confidence': prediction['confidence'],
                'position_size': prediction['position_size'],
                'leverage': prediction['leverage'],
                'holding_time': prediction['holding_time'],
                'profit_prediction': prediction['profit_prediction'],
                'timeframe_analysis': prediction['timeframe_analysis'],
                'model_type': 'Multi-Timeframe Transformer',
                'model_version': prediction.get('model_version', 'v1.0')
            }
            
        except Exception as e:
            print(f"❌ Transformer 예측 실패: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'position_size': 0.5,
                'leverage': 1.0,
                'holding_time': 30,
                'profit_prediction': 0.0,
                'timeframe_analysis': {},
                'model_type': 'Transformer (Error)',
                'error': str(e)
            }
    
    def _compare_predictions(self, rl_pred: Dict, transformer_pred: Dict) -> Dict:
        """두 모델의 예측 결과 비교"""
        
        # 액션 일치성 확인
        rl_action = rl_pred['action']
        transformer_action = transformer_pred['action']
        action_match = rl_action == transformer_action
        
        # 신뢰도 비교
        rl_confidence = rl_pred.get('confidence', 0.0)
        transformer_confidence = transformer_pred.get('confidence', 0.0)
        
        # 수익률 예측 비교 (RL은 Q값 기반 추정)
        rl_profit_estimate = self._estimate_profit_from_q_values(rl_pred.get('q_values', [0, 0, 0]))
        transformer_profit = transformer_pred.get('profit_prediction', 0.0)
        
        return {
            'action_match': action_match,
            'rl_action': rl_action,
            'transformer_action': transformer_action,
            'confidence_comparison': {
                'rl_confidence': rl_confidence,
                'transformer_confidence': transformer_confidence,
                'difference': abs(rl_confidence - transformer_confidence)
            },
            'profit_comparison': {
                'rl_estimate': rl_profit_estimate,
                'transformer_prediction': transformer_profit,
                'difference': abs(rl_profit_estimate - transformer_profit)
            },
            'agreement_level': self._calculate_agreement_level(rl_pred, transformer_pred)
        }
    
    def _estimate_profit_from_q_values(self, q_values: List[float]) -> float:
        """Q값에서 수익률 추정"""
        if len(q_values) != 3:
            return 0.0
        
        # Q값의 차이를 수익률로 근사
        max_q = max(q_values)
        min_q = min(q_values)
        return (max_q - min_q) * 0.1  # 스케일링
    
    def _calculate_agreement_level(self, rl_pred: Dict, transformer_pred: Dict) -> str:
        """두 모델의 일치도 계산"""
        score = 0
        
        # 액션 일치
        if rl_pred['action'] == transformer_pred['action']:
            score += 3
        
        # 신뢰도 차이 (낮을수록 좋음)
        conf_diff = abs(rl_pred.get('confidence', 0) - transformer_pred.get('confidence', 0))
        if conf_diff < 0.1:
            score += 2
        elif conf_diff < 0.3:
            score += 1
        
        # 수익률 예측 차이
        rl_profit = self._estimate_profit_from_q_values(rl_pred.get('q_values', [0, 0, 0]))
        transformer_profit = transformer_pred.get('profit_prediction', 0)
        profit_diff = abs(rl_profit - transformer_profit)
        if profit_diff < 0.01:
            score += 2
        elif profit_diff < 0.05:
            score += 1
        
        if score >= 6:
            return "높음 (High Agreement)"
        elif score >= 4:
            return "보통 (Medium Agreement)"
        else:
            return "낮음 (Low Agreement)"
    
    def batch_predict(self, signal_data_list: List[Dict], max_samples: int = 10) -> List[Dict]:
        """여러 Signal 데이터에 대한 배치 예측"""
        
        print(f"\n🔄 배치 예측 시작 ({min(len(signal_data_list), max_samples)}개 샘플)...")
        
        results = []
        for i, signal_data in enumerate(signal_data_list[:max_samples]):
            print(f"\n--- 샘플 {i+1}/{min(len(signal_data_list), max_samples)} ---")
            result = self.predict_from_signal(signal_data)
            results.append(result)
        
        return results
    
    def print_prediction_result(self, result: Dict):
        """예측 결과를 보기 좋게 출력"""
        
        print("\n" + "="*80)
        print("🔮 예측 결과")
        print("="*80)
        
        # Signal 정보
        signal = result['signal_data']
        print(f"📊 Signal 정보:")
        print(f"   타임스탬프: {signal.get('timestamp', 'N/A')}")
        print(f"   가격: {signal.get('close', 0.0):.2f}")
        print(f"   거래량: {signal.get('volume', 0.0):.0f}")
        
        # RL 에이전트 예측
        rl_pred = result['rl_prediction']
        print(f"\n🤖 RL 에이전트 예측:")
        print(f"   액션: {rl_pred['action']} (인덱스: {rl_pred['action_index']})")
        print(f"   신뢰도: {rl_pred['confidence']:.3f}")
        print(f"   Q값: {[f'{q:.3f}' for q in rl_pred['q_values']]}")
        print(f"   모델: {rl_pred['model_type']}")
        
        # Transformer 예측
        transformer_pred = result['transformer_prediction']
        print(f"\n🧠 Multi-Timeframe Transformer 예측:")
        print(f"   액션: {transformer_pred['action']}")
        print(f"   신뢰도: {transformer_pred['confidence']:.3f}")
        print(f"   포지션 크기: {transformer_pred['position_size']:.3f}")
        print(f"   레버리지: {transformer_pred['leverage']:.1f}x")
        print(f"   보유시간: {transformer_pred['holding_time']}분")
        print(f"   수익률 예측: {transformer_pred['profit_prediction']:.3f}")
        
        # 시간프레임 분석
        if 'timeframe_analysis' in transformer_pred:
            print(f"\n📈 시간프레임 분석:")
            for timeframe, analysis in transformer_pred['timeframe_analysis'].items():
                print(f"   {timeframe}: {analysis['trend']} (강도: {analysis['strength']:.3f})")
        
        # 비교 결과
        comparison = result['comparison']
        print(f"\n⚖️ 모델 비교:")
        print(f"   액션 일치: {'✅' if comparison['action_match'] else '❌'} ({comparison['rl_action']} vs {comparison['transformer_action']})")
        print(f"   신뢰도 차이: {comparison['confidence_comparison']['difference']:.3f}")
        print(f"   수익률 예측 차이: {comparison['profit_comparison']['difference']:.3f}")
        print(f"   일치도: {comparison['agreement_level']}")
        
        print("="*80)

def create_sample_signal_data() -> Dict:
    """샘플 Signal 데이터 생성"""
    return {
        'timestamp': datetime.now().isoformat(),
        'open': 2500.0 + random.uniform(-50, 50),
        'high': 2550.0 + random.uniform(-50, 50),
        'low': 2450.0 + random.uniform(-50, 50),
        'close': 2500.0 + random.uniform(-50, 50),
        'volume': random.uniform(1000, 10000),
        'quote_volume': random.uniform(1000000, 10000000),
        
        # Indicator 데이터
        'indicator_vwap': 2500.0 + random.uniform(-20, 20),
        'indicator_atr': random.uniform(10, 50),
        'indicator_poc': 2500.0 + random.uniform(-20, 20),
        'indicator_hvn': 2500.0 + random.uniform(-20, 20),
        'indicator_lvn': 2500.0 + random.uniform(-20, 20),
        'indicator_vwap_std': random.uniform(5, 25),
        'indicator_prev_day_high': 2500.0 + random.uniform(-20, 20),
        'indicator_prev_day_low': 2500.0 + random.uniform(-20, 20),
        'indicator_opening_range_high': 2500.0 + random.uniform(-20, 20),
        'indicator_opening_range_low': 2500.0 + random.uniform(-20, 20),
        
        # Strategy 데이터 (16개 전략)
        'session_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'session_score': random.uniform(0, 1),
        'session_confidence': random.choice(['HIGH', 'MEDIUM', 'LOW']),
        'session_entry': 2500.0 + random.uniform(-50, 50),
        'session_stop': 2500.0 + random.uniform(-50, 50),
        
        'vpvr_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'vpvr_score': random.uniform(0, 1),
        'vpvr_confidence': random.choice(['HIGH', 'MEDIUM', 'LOW']),
        'vpvr_entry': 2500.0 + random.uniform(-50, 50),
        'vpvr_stop': 2500.0 + random.uniform(-50, 50),
        
        # 나머지 전략들도 비슷하게...
        'bollinger_squeeze_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'bollinger_squeeze_score': random.uniform(0, 1),
        'orderflow_cvd_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'orderflow_cvd_score': random.uniform(0, 1),
        'ichimoku_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'ichimoku_score': random.uniform(0, 1),
        'vwap_pinball_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'vwap_pinball_score': random.uniform(0, 1),
        'vol_spike_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'vol_spike_score': random.uniform(0, 1),
        'liquidity_grab_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'liquidity_grab_score': random.uniform(0, 1),
        'vpvr_micro_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'vpvr_micro_score': random.uniform(0, 1),
        'zscore_mean_reversion_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'zscore_mean_reversion_score': random.uniform(0, 1),
        'htf_trend_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'htf_trend_score': random.uniform(0, 1),
        'oi_delta_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'oi_delta_score': random.uniform(0, 1),
        'funding_rate_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'funding_rate_score': random.uniform(0, 1),
        'multi_timeframe_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'multi_timeframe_score': random.uniform(0, 1),
        'support_resistance_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'support_resistance_score': random.uniform(0, 1),
        'ema_confluence_action': random.choice(['HOLD', 'BUY', 'SELL']),
        'ema_confluence_score': random.uniform(0, 1),
    }

def main():
    """메인 실행 함수"""
    print("🔮 RL 에이전트 vs Multi-Timeframe Transformer 예측 비교")
    print("="*80)
    
    try:
        # 1. 예측 시스템 초기화
        print("\n1️⃣ 예측 시스템 초기화...")
        
        # 사용 가능한 모델 파일 찾기
        rl_model_path = None
        transformer_model_path = None
        
        # RL 모델 찾기
        rl_models = [
            'agent/best_test_performance_model_return0.012.pth',
            'agent/best_test_performance_model_return0.011.pth',
            'agent/final_optimized_model_111d.pth'
        ]
        
        for model_path in rl_models:
            if os.path.exists(model_path):
                rl_model_path = model_path
                break
        
        # Transformer 모델 찾기
        transformer_models = [
            'agent/multitimeframe_transformer_trained.pth',
            'agent/best_multitimeframe_model.pth'
        ]
        
        for model_path in transformer_models:
            if os.path.exists(model_path):
                transformer_model_path = model_path
                break
        
        # 예측 시스템 초기화
        predictor = PredictionComparison(rl_model_path, transformer_model_path)
        
        # 2. 실제 데이터 로드 (선택사항)
        print("\n2️⃣ 데이터 로드...")
        signal_data = DataLoader.load_signal_data()
        
        if signal_data and len(signal_data) > 0:
            print(f"✅ 실제 데이터 로드: {len(signal_data):,}개")
            # 실제 데이터에서 샘플 선택
            sample_signals = random.sample(signal_data, min(5, len(signal_data)))
        else:
            print("⚠️ 실제 데이터 없음, 샘플 데이터 생성")
            # 샘플 데이터 생성
            sample_signals = [create_sample_signal_data() for _ in range(5)]
        
        # 3. 예측 실행
        print("\n3️⃣ 예측 실행...")
        
        for i, signal in enumerate(sample_signals):
            print(f"\n{'='*60}")
            print(f"📊 예측 {i+1}/{len(sample_signals)}")
            print(f"{'='*60}")
            
            result = predictor.predict_from_signal(signal)
            predictor.print_prediction_result(result)
        
        # 4. 배치 예측 (선택사항)
        print(f"\n4️⃣ 배치 예측 요약...")
        batch_results = predictor.batch_predict(sample_signals, max_samples=3)
        
        # 전체 통계
        action_matches = sum(1 for r in batch_results if r['comparison']['action_match'])
        avg_agreement = sum(1 for r in batch_results if '높음' in r['comparison']['agreement_level']) / len(batch_results)
        
        print(f"\n📈 배치 예측 통계:")
        print(f"   총 예측 수: {len(batch_results)}")
        print(f"   액션 일치: {action_matches}/{len(batch_results)} ({action_matches/len(batch_results)*100:.1f}%)")
        print(f"   높은 일치도: {avg_agreement*100:.1f}%")
        
        print(f"\n✅ 예측 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
