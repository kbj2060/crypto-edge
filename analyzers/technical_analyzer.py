#!/usr/bin/env python3
"""
기술적 분석 엔진
"""

import datetime
from typing import Dict, Optional
import pandas as pd
from data.loader import build_df
from indicators.vpvr import vpvr_key_levels
from config.integrated_config import IntegratedConfig


class TechnicalAnalyzer:
    """기술적 분석 엔진"""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
    
    def analyze_realtime_technical(self, websocket, integrated_strategy, liquidation_analyzer) -> Optional[Dict]:
        """실시간 기술적 하이브리드 분석"""
        try:
            # 데이터 로딩
            df_15m = build_df(self.config.symbol, '15m', self.config.hybrid_limit_15m, 14,
                             market='futures', price_source='last', ma_type='ema')
            df_5m = build_df(self.config.symbol, '5m', self.config.hybrid_limit_5m, 14,
                             market='futures', price_source='last', ma_type='ema')
            if df_15m.empty or df_5m.empty:
                return None
            
            # VPVR 레벨 계산
            vpvr_levels = vpvr_key_levels(df_15m,
                                         self.config.liquidation_vpvr_bins,
                                         self.config.liquidation_vpvr_lookback,
                                         topn=8)
            
            # 하이브리드 전략 즉시 분석
            hybrid_signal = integrated_strategy.analyze_hybrid_strategy(df_15m, df_5m, vpvr_levels)
            
            # 최신 청산/예측과 통합
            recent_liqs = websocket.get_recent_liquidations(self.config.liquidation_window_minutes)
            current_price = websocket.price_history[-1]['price'] if websocket.price_history else df_5m['close'].iloc[-1]
            
            # 청산 밀도 분석 추가
            liquidation_density = websocket.get_liquidation_density_analysis(current_price, 2.0)  # ±2% 범위
            
            # 청산 데이터와 기술적 지표 통합 강화
            enhanced_liquidation_signal = liquidation_analyzer.analyze_liquidation_with_technical(
                recent_liqs, liquidation_density, df_5m, current_price
            )
            
            prediction_signal = integrated_strategy.analyze_liquidation_prediction(recent_liqs, current_price)
            
            # 통합 신호 생성
            integrated_signal = integrated_strategy.get_integrated_signal(
                hybrid_signal=hybrid_signal,
                liquidation_signal=enhanced_liquidation_signal,
                prediction_signal=prediction_signal
            )
            
            return integrated_signal
                
        except Exception as e:
            print(f"❌ 실시간 기술 분석 오류: {e}")
            return None
    
    def analyze_hybrid_strategy(self, websocket, integrated_strategy) -> Optional[Dict]:
        """하이브리드 전략 분석 (5분봉 기반)"""
        try:
            # 데이터 로딩
            df_15m = build_df(self.config.symbol, '15m', self.config.hybrid_limit_15m, 14, 
                             market='futures', price_source='last', ma_type='ema')
            df_5m = build_df(self.config.symbol, '5m', self.config.hybrid_limit_5m, 14, 
                             market='futures', price_source='last', ma_type='ema')
            
            if not df_15m.empty and not df_5m.empty:
                # VPVR 레벨 계산
                vpvr_levels = vpvr_key_levels(df_15m, self.config.liquidation_vpvr_bins, 
                                              self.config.liquidation_vpvr_lookback, topn=8)
                
                # 하이브리드 전략 분석
                hybrid_signal = integrated_strategy.analyze_hybrid_strategy(df_15m, df_5m, vpvr_levels)
                
                if hybrid_signal:
                    # 통합 신호 생성
                    integrated_signal = integrated_strategy.get_integrated_signal(
                        hybrid_signal=hybrid_signal,
                        liquidation_signal=integrated_strategy.last_liquidation_signal
                    )
                    
                    return integrated_signal
            
            return None
                    
        except Exception as e:
            print(f"❌ 하이브리드 분석 오류: {e}")
            return None
