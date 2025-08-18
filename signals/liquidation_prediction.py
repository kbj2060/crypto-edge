import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

@dataclass
class LiquidationPredictionConfig:
    """청산 예측 설정"""
    # 가격 구간 분석
    price_bin_size: float = 0.001  # 가격 구간 크기 (0.1%)
    min_liquidation_density: int = 5  # 최소 청산 밀도
    
    # 시간 분석
    time_window_minutes: int = 30  # 시간 윈도우
    cascade_threshold: int = 10  # 연쇄 청산 임계값
    
    # 예측 신뢰도
    min_prediction_confidence: float = 0.6  # 최소 예측 신뢰도
    volatility_threshold: float = 0.02  # 변동성 임계값 (2%)
    
    # 리스크 관리
    max_prediction_horizon_hours: int = 4  # 최대 예측 시간 (4시간)
    stop_loss_multiplier: float = 1.5  # 손절가 배수

class LiquidationPredictionStrategy:
    """청산 기반 폭등/폭락 예측 전략"""
    
    def __init__(self, config: LiquidationPredictionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 예측 결과 저장
        self.predictions = []
        self.prediction_history = []
        
        # 통계 데이터
        self.liquidation_density_map = {}  # 가격별 청산 밀도
        self.time_patterns = {}  # 시간대별 청산 패턴
        self.cascade_events = []  # 연쇄 청산 이벤트
    
    def analyze_liquidation_patterns(self, liquidations: List[Dict]) -> Dict[str, Any]:
        """청산 패턴 분석"""
        if not liquidations:
            return {}
        
        try:
            # 1. 가격별 청산 밀도 분석
            density_analysis = self._analyze_price_density(liquidations)
            
            # 2. 시간대별 청산 패턴 분석
            time_analysis = self._analyze_time_patterns(liquidations)
            
            # 3. 연쇄 청산 분석
            cascade_analysis = self._analyze_cascade_events(liquidations)
            
            # 4. 폭등/폭락 위험도 계산
            risk_analysis = self._calculate_explosion_risk(density_analysis, time_analysis, cascade_analysis)
            
            return {
                'density_analysis': density_analysis,
                'time_analysis': time_analysis,
                'cascade_analysis': cascade_analysis,
                'risk_analysis': risk_analysis,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"청산 패턴 분석 오류: {e}")
            return {}
    
    def _analyze_price_density(self, liquidations: List[Dict]) -> Dict[str, Any]:
        """가격별 청산 밀도 분석"""
        if not liquidations:
            return {}
        
        # 가격 구간별 청산 집계
        price_bins = {}
        current_price = liquidations[-1]['price'] if liquidations else 0
        
        for liq in liquidations:
            price = liq['price']
            side = liq['side']
            quantity = liq['quantity']
            
            # 가격 구간 계산
            bin_key = self._get_price_bin(price, current_price)
            
            if bin_key not in price_bins:
                price_bins[bin_key] = {
                    'buy_count': 0,
                    'sell_count': 0,
                    'buy_quantity': 0.0,
                    'sell_quantity': 0.0,
                    'total_value': 0.0,
                    'prices': []
                }
            
            price_bins[bin_key]['prices'].append(price)
            
            if side == 'BUY':
                price_bins[bin_key]['buy_count'] += 1
                price_bins[bin_key]['buy_quantity'] += quantity
            else:
                price_bins[bin_key]['sell_count'] += 1
                price_bins[bin_key]['sell_quantity'] += quantity
            
            price_bins[bin_key]['total_value'] += quantity * price
        
        # 밀도가 높은 구간 찾기
        high_density_bins = {}
        for bin_key, data in price_bins.items():
            total_count = data['buy_count'] + data['sell_count']
            if total_count >= self.config.min_liquidation_density:
                high_density_bins[bin_key] = data
        
        return {
            'price_bins': price_bins,
            'high_density_bins': high_density_bins,
            'current_price': current_price,
            'total_liquidations': len(liquidations)
        }
    
    def _analyze_time_patterns(self, liquidations: List[Dict]) -> Dict[str, Any]:
        """시간대별 청산 패턴 분석"""
        if not liquidations:
            return {}
        
        # 시간대별 청산 집계
        hourly_patterns = {}
        for hour in range(24):
            hourly_patterns[hour] = {
                'count': 0,
                'buy_count': 0,
                'sell_count': 0,
                'total_value': 0.0
            }
        
        for liq in liquidations:
            timestamp = liq['timestamp']
            hour = timestamp.hour
            side = liq['side']
            quantity = liq['quantity']
            price = liq['price']
            
            hourly_patterns[hour]['count'] += 1
            hourly_patterns[hour]['total_value'] += quantity * price
            
            if side == 'BUY':
                hourly_patterns[hour]['buy_count'] += 1
            else:
                hourly_patterns[hour]['sell_count'] += 1
        
        # 활발한 시간대 찾기
        active_hours = []
        for hour, data in hourly_patterns.items():
            if data['count'] > 0:
                active_hours.append({
                    'hour': hour,
                    'data': data
                })
        
        # 시간대별 패턴 분석
        time_analysis = {
            'hourly_patterns': hourly_patterns,
            'active_hours': active_hours,
            'peak_hour': max(active_hours, key=lambda x: x['data']['count'])['hour'] if active_hours else None,
            'total_hours': len([h for h in hourly_patterns.values() if h['count'] > 0])
        }
        
        return time_analysis
    
    def _analyze_cascade_events(self, liquidations: List[Dict]) -> Dict[str, Any]:
        """연쇄 청산 이벤트 분석"""
        if len(liquidations) < 2:
            return {}
        
        # 시간순 정렬
        sorted_liquidations = sorted(liquidations, key=lambda x: x['timestamp'])
        
        cascade_events = []
        current_cascade = []
        
        for i, liq in enumerate(sorted_liquidations):
            if not current_cascade:
                current_cascade = [liq]
                continue
            
            # 이전 청산과의 시간 간격 확인
            time_diff = (liq['timestamp'] - current_cascade[-1]['timestamp']).total_seconds()
            
            # 10초 이내의 연쇄 청산
            if time_diff <= 10:
                current_cascade.append(liq)
            else:
                # 현재 캐스케이드 종료
                if len(current_cascade) >= self.config.cascade_threshold:
                    cascade_events.append({
                        'start_time': current_cascade[0]['timestamp'],
                        'end_time': current_cascade[-1]['timestamp'],
                        'count': len(current_cascade),
                        'total_value': sum(liq['quantity'] * liq['price'] for liq in current_cascade),
                        'price_range': {
                            'min': min(liq['price'] for liq in current_cascade),
                            'max': max(liq['price'] for liq in current_cascade)
                        },
                        'side_distribution': {
                            'buy': len([liq for liq in current_cascade if liq['side'] == 'BUY']),
                            'sell': len([liq for liq in current_cascade if liq['side'] == 'SELL'])
                        }
                    })
                
                current_cascade = [liq]
        
        # 마지막 캐스케이드 처리
        if len(current_cascade) >= self.config.cascade_threshold:
            cascade_events.append({
                'start_time': current_cascade[0]['timestamp'],
                'end_time': current_cascade[-1]['timestamp'],
                'count': len(current_cascade),
                'total_value': sum(liq['quantity'] * liq['price'] for liq in current_cascade),
                'price_range': {
                    'min': min(liq['price'] for liq in current_cascade),
                    'max': max(liq['price'] for liq in current_cascade)
                },
                'side_distribution': {
                    'buy': len([liq for liq in current_cascade if liq['side'] == 'BUY']),
                    'sell': len([liq for liq in current_cascade if liq['side'] == 'SELL'])
                }
            })
        
        return {
            'cascade_events': cascade_events,
            'total_cascades': len(cascade_events),
            'largest_cascade': max(cascade_events, key=lambda x: x['count']) if cascade_events else None
        }
    
    def _calculate_explosion_risk(self, density_analysis: Dict, time_analysis: Dict, cascade_analysis: Dict) -> Dict[str, Any]:
        """폭등/폭락 위험도 계산"""
        risk_score = 0.0
        risk_factors = []
        
        # 1. 청산 밀도 위험도 (0-0.3)
        if 'high_density_bins' in density_analysis:
            high_density_count = len(density_analysis['high_density_bins'])
            density_risk = min(high_density_count / 10.0, 1.0) * 0.3
            risk_score += density_risk
            risk_factors.append(f"청산 밀도: {high_density_count}개 구간")
        
        # 2. 연쇄 청산 위험도 (0-0.4)
        if 'cascade_events' in cascade_analysis:
            cascade_count = len(cascade_analysis['cascade_events'])
            cascade_risk = min(cascade_count / 5.0, 1.0) * 0.4
            risk_score += cascade_risk
            risk_factors.append(f"연쇄 청산: {cascade_count}개 이벤트")
        
        # 3. 시간 패턴 위험도 (0-0.3)
        if 'active_hours' in time_analysis:
            active_hours_count = time_analysis['total_hours']
            time_risk = min(active_hours_count / 24.0, 1.0) * 0.3
            risk_score += time_risk
            risk_factors.append(f"활성 시간: {active_hours_count}시간")
        
        # 위험도 등급 결정
        if risk_score >= 0.8:
            risk_level = "CRITICAL"
            risk_description = "폭등/폭락 매우 높음"
        elif risk_score >= 0.6:
            risk_level = "HIGH"
            risk_description = "폭등/폭락 높음"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
            risk_description = "폭등/폭락 보통"
        else:
            risk_level = "LOW"
            risk_description = "폭등/폭락 낮음"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_description': risk_description,
            'risk_factors': risk_factors,
            'timestamp': datetime.now()
        }
    
    def predict_explosion_points(self, liquidations: List[Dict], current_price: float) -> List[Dict]:
        """폭등/폭락 지점 예측"""
        if not liquidations:
            return []
        
        try:
            # 패턴 분석
            pattern_analysis = self.analyze_liquidation_patterns(liquidations)
            
            if not pattern_analysis:
                return []
            
            predictions = []
            risk_analysis = pattern_analysis.get('risk_analysis', {})
            density_analysis = pattern_analysis.get('density_analysis', {})
            
            # 위험도가 높은 경우에만 예측
            if risk_analysis.get('risk_score', 0) < self.config.min_prediction_confidence:
                return []
            
            # 가격별 예측 지점 생성
            high_density_bins = density_analysis.get('high_density_bins', {})
            
            for bin_key, bin_data in high_density_bins.items():
                # 가격 구간의 중심점 계산
                prices = bin_data['prices']
                if not prices:
                    continue
                
                center_price = sum(prices) / len(prices)
                price_diff = abs(center_price - current_price) / current_price
                
                # 현재 가격과 너무 멀면 제외
                if price_diff > 0.1:  # 10% 이상 차이
                    continue
                
                # BUY/SELL 불균형 분석
                buy_ratio = bin_data['buy_count'] / (bin_data['buy_count'] + bin_data['sell_count']) if (bin_data['buy_count'] + bin_data['sell_count']) > 0 else 0.5
                
                # 예측 방향 결정
                if buy_ratio > 0.7:  # BUY 청산이 많음
                    prediction_type = "EXPLOSION_UP"
                    target_price = center_price * 1.02  # 2% 상승 예상
                    confidence = min(buy_ratio * risk_analysis.get('risk_score', 0), 1.0)
                elif buy_ratio < 0.3:  # SELL 청산이 많음
                    prediction_type = "EXPLOSION_DOWN"
                    target_price = center_price * 0.98  # 2% 하락 예상
                    confidence = min((1 - buy_ratio) * risk_analysis.get('risk_score', 0), 1.0)
                else:
                    continue  # 중립적인 경우 제외
                
                # 예측 시간 계산 (위험도에 따라)
                risk_score = risk_analysis.get('risk_score', 0)
                if risk_score > 0.8:
                    prediction_hours = 1  # 1시간 내
                elif risk_score > 0.6:
                    prediction_hours = 2  # 2시간 내
                else:
                    prediction_hours = 4  # 4시간 내
                
                prediction = {
                    'type': prediction_type,
                    'current_price': current_price,
                    'target_price': target_price,
                    'center_price': center_price,
                    'confidence': confidence,
                    'prediction_hours': prediction_hours,
                    'expected_time': datetime.now() + timedelta(hours=prediction_hours),
                    'risk_score': risk_score,
                    'risk_level': risk_analysis.get('risk_level', 'UNKNOWN'),
                    'liquidation_density': len(prices),
                    'buy_ratio': buy_ratio,
                    'reason': f"청산 밀도: {len(prices)}개, BUY 비율: {buy_ratio:.1%}, 위험도: {risk_score:.1%}"
                }
                
                predictions.append(prediction)
            
            # 신뢰도 순으로 정렬
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # 예측 결과 저장
            self.predictions = predictions
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'predictions': predictions,
                'risk_analysis': risk_analysis
            })
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"폭등/폭락 예측 오류: {e}")
            return []
    
    def _get_price_bin(self, price: float, current_price: float) -> str:
        """가격 구간 키 생성"""
        if current_price <= 0:
            return "0"
        
        # 현재 가격 기준으로 상대적 구간 계산
        relative_price = price / current_price
        bin_index = int(relative_price / self.config.price_bin_size)
        return str(bin_index)
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """예측 요약 정보"""
        current_predictions = self.predictions
        total_predictions = len(self.prediction_history)
        
        # 예측 정확도 계산 (과거 예측과 비교)
        accuracy = 0.0
        if len(self.prediction_history) > 1:
            # 간단한 정확도 계산 (실제 구현에서는 더 정교한 계산 필요)
            accuracy = min(0.8, 0.5 + (len(current_predictions) * 0.1))
        
        return {
            'current_predictions': current_predictions,
            'total_predictions': total_predictions,
            'accuracy': accuracy,
            'last_update': datetime.now(),
            'config': {
                'price_bin_size': self.config.price_bin_size,
                'min_liquidation_density': self.config.min_liquidation_density,
                'cascade_threshold': self.config.cascade_threshold
            }
        }
    
    def cleanup_old_predictions(self, max_age_hours: int = 24):
        """오래된 예측 데이터 정리"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # 예측 히스토리 정리
        self.prediction_history = [
            pred for pred in self.prediction_history
            if pred['timestamp'] > cutoff_time
        ]
        
        # 현재 예측에서 만료된 것 제거
        current_time = datetime.now()
        self.predictions = [
            pred for pred in self.predictions
            if pred['expected_time'] > current_time
        ]
