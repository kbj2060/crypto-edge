#!/usr/bin/env python3
"""
통합 스마트 자동 트레이더
하이브리드 전략(5분봉) + 실시간 청산 전략의 시너지 효과를 활용합니다.
"""

import asyncio
import time
import datetime
import threading
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from data.binance_websocket import BinanceWebSocket
from data.loader import build_df
from indicators.vpvr import vpvr_key_levels
from config.integrated_config import IntegratedConfig
from signals.integrated_strategy import IntegratedStrategy
from signals.timing_strategy import TimingStrategy

class IntegratedSmartTrader:
	"""통합 스마트 자동 트레이더"""
	
	def __init__(self, config: IntegratedConfig):
		self.config = config
		self.running = False
		
		# 웹소켓 클라이언트 (실시간 청산 데이터)
		self.websocket = BinanceWebSocket(config.symbol)
		
		# 통합 전략
		self.integrated_strategy = IntegratedStrategy(config)
		
		# 타이밍 전략 (포지션 관리)
		self.timing_strategy = TimingStrategy(self.integrated_strategy.timing_cfg)
		
		# 콜백 등록
		self._setup_callbacks()
		
		# 통계
		self.signal_count = 0
		self.synergy_count = 0
		self.last_signal_time = None
		self.last_5min_analysis = None
		
		# 신호 중복 방지
		self.last_signal_hash = None
		self.signal_cooldown = 8  # 15초에서 8초로 줄임 (스캘핑용)
		
		# 조용한 모드 (과도한 로그 방지)
		self.quiet_mode = True
		
		# 거래량 급증 집계
		self.volume_spike_buffer = []
		self.last_volume_summary = None
		self.volume_summary_cooldown = 30  # 30초마다 요약 출력
		
		# 스레드
		self.hybrid_thread = None
		self.websocket_thread = None
	
	def _setup_callbacks(self):
		"""웹소켓 콜백 설정"""
		self.websocket.add_callback('liquidation', self._on_liquidation)
		self.websocket.add_callback('volume', self._on_volume_spike)
		self.websocket.add_callback('price', self._on_price_update)
		self.websocket.add_callback('kline', self._on_kline)
	
	def _on_liquidation(self, liquidation_data: Dict):
		"""청산 이벤트 콜백"""
		# 간단한 한 줄 출력
		side = liquidation_data['side']
		quantity = liquidation_data['quantity']
		price = liquidation_data['price']
		value = quantity * price
		
		print(f"🔥 {side} 청산: {quantity:.2f} ETH (${value:,.0f}) @ ${price:.2f}")
		
		# 현재 호가 ±3% 범위 청산 밀도 분석 출력
		self._print_current_liquidation_density()
		
		# 실시간 청산 신호 분석
		self._analyze_realtime_liquidation()
	
	def _on_volume_spike(self, volume_data: Dict):
		"""거래량 급증 콜백"""
		# 거래량 급증을 버퍼에 추가
		self.volume_spike_buffer.append({
			'timestamp': datetime.datetime.now(),
			'data': volume_data
		})
		
		# 30초마다 요약 출력
		now = datetime.datetime.now()
		if (not self.last_volume_summary or 
			(now - self.last_volume_summary).total_seconds() >= self.volume_summary_cooldown):
			
			self._print_volume_spike_summary()
			self.last_volume_summary = now
			self.volume_spike_buffer.clear()
		
		# 실시간 청산 신호 분석
		self._analyze_realtime_liquidation()
	
	def _print_volume_spike_summary(self):
		"""거래량 급증 요약 출력"""
		if not self.volume_spike_buffer:
			return
		
		# 방향성 분석
		long_liquidation = 0
		short_liquidation = 0
		neutral_pressure = 0
		
		for spike in self.volume_spike_buffer:
			trend = spike['data'].get('trend', 'NEUTRAL')
			if trend == 'LONG_LIQUIDATION':
				long_liquidation += 1
			elif trend == 'SHORT_LIQUIDATION':
				short_liquidation += 1
			else:
				neutral_pressure += 1
		
		# 전체적인 시장 방향성 판단
		if short_liquidation > long_liquidation * 1.5:
			print(f"📊 거래량급증: 📈 숏청산우세 ({short_liquidation}회) - 상승압력")
		elif long_liquidation > short_liquidation * 1.5:
			print(f"📊 거래량급증: 📉 롱청산우세 ({long_liquidation}회) - 하락압력")
		else:
			print(f"📊 거래량급증: ➡️ 중립 ({long_liquidation}롱청산/{short_liquidation}숏청산)")
	
	def _explain_volume_spike(self, ratio: float, trend: str, price_change: float, micro_trend: float = 0) -> str:
		"""거래량 급증 의미 설명"""
		if ratio >= 3.0:
			intensity = "매우 강한"
		elif ratio >= 2.0:
			intensity = "강한"
		elif ratio >= 1.5:
			intensity = "중간"
		else:
			intensity = "약한"
		
		# 미세 트렌드가 있으면 우선 사용
		if abs(micro_trend) > 0.001:
			if micro_trend > 0:
				return f"{intensity} 매수 압력 - 미세 상승 추세 감지"
			else:
				return f"{intensity} 매도 압력 - 미세 하락 추세 감지"
		
		# 기존 로직
		if trend == "BUY" and price_change > 0:
			return f"{intensity} 매수 압력 - 가격 상승 동반"
		elif trend == "SELL" and price_change < 0:
			return f"{intensity} 매도 압력 - 가격 하락 동반"
		elif trend == "BUY" and price_change < 0:
			return f"{intensity} 매수 압력 - 가격 하락 (반등 가능성)"
		elif trend == "SELL" and price_change > 0:
			return f"{intensity} 매도 압력 - 가격 상승 (조정 가능성)"
		else:
			return f"{intensity} 거래량 급증 - 방향성 불명확"
	
	def _calculate_current_atr(self) -> Optional[float]:
		"""현재 ATR 계산"""
		try:
			if len(self.websocket.price_history) >= 14:
				prices = [p['price'] for p in self.websocket.price_history[-14:]]
				price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
				atr = sum(price_changes) / len(price_changes)
				return atr
		except Exception:
			pass
		return None
	
	def _on_price_update(self, price_data: Dict):
		"""가격 업데이트 콜백"""
		# 가격 변동이 클 때만 출력 (스캘핑용으로 더 민감하게)
		if len(self.websocket.price_history) >= 2:
			prev_price = self.websocket.price_history[-2]['price']
			current_price = price_data['price']
			change_pct = ((current_price - prev_price) / prev_price) * 100
			
			if abs(change_pct) > 0.1:  # 0.2%에서 0.1%로 낮춤 (스캘핑용)
				print(f"💰 가격 변동: ${prev_price:.2f} → ${current_price:.2f} ({change_pct:+.2f}%)")
				# 큰 가격 변동 시에만 실시간 기술적 분석
				self._analyze_realtime_technical()
	
	def _on_kline(self, kline_data: Dict):
		"""1분봉 K라인 업데이트 콜백"""
		# K라인이 닫힐 때(x=True)만 분석
		if kline_data.get('x', False):
			self._analyze_realtime_technical()
	
	def _analyze_realtime_technical(self):
		"""실시간 기술적 하이브리드 분석 (대기 없이 즉시 실행)"""
		try:
			# 데이터 로딩 (REST 기반이지만 대기 없이 즉시 조회)
			df_15m = build_df(self.config.symbol, '15m', self.config.hybrid_limit_15m, 14,
						 market='futures', price_source='last', ma_type='ema')
			df_5m = build_df(self.config.symbol, '5m', self.config.hybrid_limit_5m, 14,
						market='futures', price_source='last', ma_type='ema')
			if df_15m.empty or df_5m.empty:
				return
			
			# VPVR 레벨 계산
			vpvr_levels = vpvr_key_levels(df_15m,
										 self.config.liquidation_vpvr_bins,
										 self.config.liquidation_vpvr_lookback,
										 topn=8)
			
			# 하이브리드 전략 즉시 분석
			hybrid_signal = self.integrated_strategy.analyze_hybrid_strategy(df_15m, df_5m, vpvr_levels)
			
			# 최신 청산/예측과 통합 (스캘핑용으로 더 민감하게)
			recent_liqs = self.websocket.get_recent_liquidations(self.config.liquidation_window_minutes)
			current_price = self.websocket.price_history[-1]['price'] if self.websocket.price_history else df_5m['close'].iloc[-1]
			
			# 청산 밀도 분석 추가
			liquidation_density = self.websocket.get_liquidation_density_analysis(current_price, 2.0)  # ±2% 범위
			
			# 청산 데이터와 기술적 지표 통합 강화
			enhanced_liquidation_signal = self._enhance_liquidation_with_technical(
				recent_liqs, liquidation_density, df_5m, current_price
			)
			
			prediction_signal = self.integrated_strategy.analyze_liquidation_prediction(recent_liqs, current_price)
			
			# 통합 신호 생성 (스캘핑용으로 더 민감하게)
			integrated_signal = self.integrated_strategy.get_integrated_signal(
				hybrid_signal=hybrid_signal,
				liquidation_signal=enhanced_liquidation_signal,
				prediction_signal=prediction_signal
			)
			
			if integrated_signal:
				self._process_integrated_signal(integrated_signal)
				
		except Exception as e:
			print(f"❌ 실시간 기술 분석 오류: {e}")
	
	def _enhance_liquidation_with_technical(self, liquidations: List, density_analysis: Dict, df_5m: pd.DataFrame, current_price: float) -> Dict:
		"""청산 데이터와 기술적 지표를 통합하여 강화된 신호 생성"""
		if not liquidations or df_5m.empty:
			return None
		
		try:
			# 최근 가격 데이터
			recent_close = df_5m['close'].iloc[-5:].values
			recent_volume = df_5m['volume'].iloc[-5:].values
			
			# 기술적 지표 계산
			price_momentum = (recent_close[-1] - recent_close[0]) / recent_close[0] * 100
			volume_trend = recent_volume[-1] / np.mean(recent_volume[:-1]) if len(recent_volume) > 1 else 1.0
			
			# EMA 기울기 계산 (EMA_20이 없으면 close 가격 사용)
			ema_20 = df_5m['EMA_20'].iloc[-3:].values if 'EMA_20' in df_5m.columns else df_5m['close'].iloc[-3:].values
			ema_slope = (ema_20[-1] - ema_20[0]) / ema_20[0] * 100 if len(ema_20) > 1 else 0
			
			# RSI 확인 (StochRSI_K가 없으면 기본값 사용)
			rsi_k = df_5m['StochRSI_K'].iloc[-1] if 'StochRSI_K' in df_5m.columns else 50
			
			# 청산 패턴 분석
			long_liquidations = [liq for liq in liquidations if liq.get('side') == 'BUY']
			short_liquidations = [liq for liq in liquidations if liq.get('side') == 'SELL']
			
			long_volume = sum(liq.get('quantity', 0) for liq in long_liquidations)
			short_volume = sum(liq.get('quantity', 0) for liq in short_liquidations)
			
			# 청산 밀도 정보
			max_density_price = density_analysis.get('max_density_price', current_price)
			max_density_volume = density_analysis.get('max_density_volume', 0)
			
			# 통합 신호 생성
			signal_strength = 0
			signal_bias = 'NEUTRAL'
			confidence = 0
			
			# 롱 청산 우세 + 기술적 하락 신호 (롱 청산 많음 = 가격 하락 = 숏 진입)
			if (long_volume > short_volume * 1.2 and 
				price_momentum < -0.05 and 
				ema_slope < -0.02 and 
				rsi_k > 20):
				
				signal_bias = 'SHORT'  # 롱 청산 많음 → 숏 진입
				signal_strength = min(0.8, (long_volume / max(short_volume, 1)) * 0.3 + abs(price_momentum) * 0.4)
				confidence = min(0.9, signal_strength + (volume_trend - 1) * 0.2)
				
			# 숏 청산 우세 + 기술적 상승 신호 (숏 청산 많음 = 가격 상승 = 롱 진입)
			elif (short_volume > long_volume * 1.2 and 
				  price_momentum > 0.05 and 
				  ema_slope > 0.02 and 
				  rsi_k < 80):
				
				signal_bias = 'LONG'  # 숏 청산 많음 → 롱 진입
				signal_strength = min(0.8, (short_volume / max(long_volume, 1)) * 0.3 + abs(price_momentum) * 0.4)
				confidence = min(0.9, signal_strength + (volume_trend - 1) * 0.2)
			
			# 청산 밀도가 높은 가격대 근처에서의 신호
			if max_density_volume > 0:
				density_distance = abs(max_density_price - current_price) / current_price * 100
				if density_distance < 0.5:  # 0.5% 이내
					confidence = min(0.95, confidence + 0.1)  # 신뢰도 10% 증가
			
			# 신호 임계값 체크
			if confidence < self.config.liquidation_min_confidence:
				return None
			
			# 손절가와 익절가 계산 (스캘핑용)
			atr = self._calculate_current_atr()
			if atr:
				if signal_bias == 'LONG':
					stop_loss = current_price - (atr * 1.5)  # ATR 1.5배
					take_profit1 = current_price + (atr * 2.0)  # ATR 2배
					take_profit2 = current_price + (atr * 3.0)  # ATR 3배
				elif signal_bias == 'SHORT':
					stop_loss = current_price + (atr * 1.5)  # ATR 1.5배
					take_profit1 = current_price - (atr * 2.0)  # ATR 2배
					take_profit2 = current_price - (atr * 3.0)  # ATR 3배
				else:
					return None
			else:
				# ATR이 없을 때 기본값
				if signal_bias == 'LONG':
					stop_loss = current_price * 0.995  # 0.5% 손절
					take_profit1 = current_price * 1.008  # 0.8% 익절
					take_profit2 = current_price * 1.015  # 1.5% 익절
				elif signal_bias == 'SHORT':
					stop_loss = current_price * 1.005  # 0.5% 손절
					take_profit1 = current_price * 0.992  # 0.8% 익절
					take_profit2 = current_price * 0.985  # 1.5% 익절
				else:
					return None
			
			return {
				'signal_type': 'ENHANCED_LIQUIDATION',
				'action': 'BUY' if signal_bias == 'LONG' else 'SELL' if signal_bias == 'SHORT' else 'NEUTRAL',
				'confidence': confidence,
				'entry_price': current_price,
				'stop_loss': stop_loss,
				'take_profit1': take_profit1,
				'take_profit2': take_profit2,
				'liquidation_volume': max(long_volume, short_volume),
				'price_momentum': price_momentum,
				'volume_trend': volume_trend,
				'ema_slope': ema_slope,
				'rsi_k': rsi_k,
				'timestamp': datetime.datetime.now()
			}
			
		except Exception as e:
			print(f"❌ 청산-기술 통합 신호 생성 오류: {e}")
			return None
	
	def _analyze_realtime_liquidation(self):
		"""실시간 청산 신호 분석"""
		try:
			# 현재 가격과 ATR 가져오기
			if not self.websocket.price_history:
				return
			
			current_price = self.websocket.price_history[-1]['price']
			
			# ATR 계산 (간단한 변동성 계산)
			if len(self.websocket.price_history) >= 14:
				prices = [p['price'] for p in self.websocket.price_history[-14:]]
				price_changes = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
				atr = sum(price_changes) / len(price_changes)
			else:
				atr = current_price * 0.02  # 기본값
			
			# 청산 통계 분석
			liquidation_stats = self.websocket.get_liquidation_stats(self.config.liquidation_window_minutes)
			volume_analysis = self.websocket.get_volume_analysis(3)
			
			# 청산 신호 분석
			liquidation_signal = self.integrated_strategy.analyze_liquidation_strategy(
				liquidation_stats, volume_analysis, current_price, atr
			)
			
			# 청산 예측 분석
			recent_liquidations = self.websocket.get_recent_liquidations(self.config.liquidation_window_minutes)
			prediction_signal = self.integrated_strategy.analyze_liquidation_prediction(
				recent_liquidations, current_price
			)
			
			# 폭등/폭락 경고 생성
			explosion_alert = self.integrated_strategy.get_explosion_alert(
				hybrid_signal=self.integrated_strategy.last_hybrid_signal,
				liquidation_signal=liquidation_signal,
				prediction_signal=prediction_signal
			)
			
			if explosion_alert:
				self._process_explosion_alert(explosion_alert)
			
			if liquidation_signal or prediction_signal:
				# 통합 신호 생성
				integrated_signal = self.integrated_strategy.get_integrated_signal(
					hybrid_signal=self.integrated_strategy.last_hybrid_signal,
					liquidation_signal=liquidation_signal,
					prediction_signal=prediction_signal
				)
				
				if integrated_signal:
					self._process_integrated_signal(integrated_signal)
			
		except Exception as e:
			print(f"❌ 실시간 청산 분석 오류: {e}")
	
	def _run_hybrid_analysis(self):
		"""하이브리드 전략 분석 (5분봉 기반)"""
		while self.running:
			try:
				# 5분봉 타이밍 계산
				next_candle = self._get_next_5min_candle_time()
				now = datetime.datetime.now()
				
				if now >= next_candle:
					# 1초 후 분석 시작
					time.sleep(1)
					
					print(f"\n⏰ {now.strftime('%H:%M:%S')} - 5분봉 하이브리드 분석 시작")
					
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
						hybrid_signal = self.integrated_strategy.analyze_hybrid_strategy(df_15m, df_5m, vpvr_levels)
						
						if hybrid_signal:
							# 통합 신호 생성
							integrated_signal = self.integrated_strategy.get_integrated_signal(
								hybrid_signal=hybrid_signal,
								liquidation_signal=self.integrated_strategy.last_liquidation_signal
							)
							
							if integrated_signal:
								self._process_integrated_signal(integrated_signal)
						
						self.last_5min_analysis = now
						print(f"✅ {now.strftime('%H:%M:%S')} - 5분봉 분석 완료")
					
					# 다음 5분봉까지 대기
					time.sleep(60)  # 1분 대기
				else:
					# 다음 5분봉까지 대기
					time.sleep(1)
					
			except Exception as e:
				print(f"❌ 하이브리드 분석 오류: {e}")
				time.sleep(10)
	
	def _get_next_5min_candle_time(self) -> datetime.datetime:
		"""다음 5분봉 시간 계산"""
		now = datetime.datetime.now()
		minutes_to_next = 5 - (now.minute % 5)
		if minutes_to_next == 5:
			minutes_to_next = 0
		
		next_candle = now.replace(second=0, microsecond=0)
		if minutes_to_next > 0:
			next_candle = next_candle + datetime.timedelta(minutes=minutes_to_next)
		
		return next_candle
	
	def _process_integrated_signal(self, signal: Dict):
		"""통합 신호 처리"""
		try:
			signal_type = signal.get('signal_type', 'UNKNOWN')
			action = signal.get('final_signal') or signal.get('action')
			confidence = signal.get('confidence', 0)
			risk_reward = signal.get('risk_reward', 0)
			entry_price = signal.get('entry_price', 0)
			stop_loss = signal.get('stop_loss', 0)
			take_profit1 = signal.get('take_profit1', 0)
			take_profit2 = signal.get('take_profit2', 0)
			
			# 신호 중복 방지
			signal_hash = f"{signal_type}_{action}_{entry_price:.2f}_{confidence:.1%}"
			now = datetime.datetime.now()
			
			if (self.last_signal_hash == signal_hash and 
				self.last_signal_time and 
				(now - self.last_signal_time).total_seconds() < self.signal_cooldown):
				return  # 중복 신호 무시
			
			# 시너지 신호 특별 처리
			if signal_type == 'SYNERGY':
				print(f"\n🔥🔥🔥 SYNERGY 신호! 🔥🔥🔥")
				print(f"🎯 {action} - {now.strftime('%H:%M:%S')}")
				print(f"💰 ${entry_price:.2f} | 📊 {confidence:.1%} | ⚖️ {risk_reward:.1f}")
				print(f"🛑 ${stop_loss:.2f} | 💎 ${take_profit1:.2f} | 💎 ${take_profit2:.2f}")
				print(f"🔍 {signal.get('synergy_reason', '')}")
				self.synergy_count += 1
			else:
				# 일반 신호 출력
				if action == "BUY":
					print(f"\n📈 {signal_type} BUY 신호 - {now.strftime('%H:%M:%S')}")
					print(f"💰 ${entry_price:.2f} | 📊 {confidence:.1%} | ⚖️ {risk_reward:.1f}")
					print(f"🛑 ${stop_loss:.2f} | 💎 ${take_profit1:.2f} | 💎 ${take_profit2:.2f}")
				elif action == "SELL":
					print(f"\n📉 {signal_type} SELL 신호 - {now.strftime('%H:%M:%S')}")
					print(f"💰 ${entry_price:.2f} | 📊 {confidence:.1%} | ⚖️ {risk_reward:.1f}")
					print(f"🛑 ${stop_loss:.2f} | 💎 ${take_profit1:.2f} | 💎 ${take_profit2:.2f}")
			
			# 청산 가격 정보 표시
			if stop_loss and entry_price:
				# ATR 기반 청산 가격 계산
				atr = self._calculate_current_atr()
				if atr:
					# 트레일링 스탑 청산 가격
					if action == "BUY":
						trailing_stop = entry_price - (atr * 2.0)  # ATR 2배
						print(f"🔴 손절가: ${stop_loss:.2f} (고정)")
						print(f"🔄 트레일링 스탑: ${trailing_stop:.2f} (ATR 2배)")
						print(f"⚠️  청산 위험: ${stop_loss:.2f} 도달 시")
					elif action == "SELL":
						trailing_stop = entry_price + (atr * 2.0)  # ATR 2배
						print(f"🔴 손절가: ${stop_loss:.2f} (고정)")
						print(f"🔄 트레일링 스탑: ${trailing_stop:.2f} (ATR 2배)")
						print(f"⚠️  청산 위험: ${stop_loss:.2f} 도달 시")
			
			# 타이밍 분석
			timing_analysis = self.timing_strategy.analyze_entry_timing(signal, entry_price)
			
			if timing_analysis and timing_analysis.get('action') in ['BUY', 'SELL']:
				# 포지션 오픈
				position_id = self.timing_strategy.open_position(timing_analysis)
				print(f"🚀 포지션 오픈: {position_id}")
				
				# 통계 업데이트
				self.signal_count += 1
				self.last_signal_time = now
				self.last_signal_hash = signal_hash
				
				# 포지션 요약
				position_summary = self.timing_strategy.get_position_summary()
				if position_summary['active_positions'] > 0:
					print(f"📊 활성 포지션: {position_summary['active_positions']}개 | 💰 일일 손익: {position_summary['daily_pnl']:.4f}")
			
		except Exception as e:
			print(f"❌ 통합 신호 처리 오류: {e}")
	
	def _process_explosion_alert(self, alert: Dict):
		"""폭등/폭락 경고 처리"""
		try:
			total_alerts = alert.get('total_alerts', 0)
			critical_alerts = alert.get('critical_alerts', 0)
			high_alerts = alert.get('high_alerts', 0)
			
			print(f"\n🚨 폭등/폭락 경고 - {datetime.datetime.now().strftime('%H:%M:%S')}")
			print(f"📊 총 경고: {total_alerts}개 (🔥🔥🔥 {critical_alerts}개, 🔥🔥 {high_alerts}개)")
			
			# 개별 경고 출력
			for alert_item in alert.get('alerts', []):
				alert_type = alert_item.get('type', 'UNKNOWN')
				level = alert_item.get('level', 'UNKNOWN')
				message = alert_item.get('message', '')
				
				if level == 'CRITICAL':
					print(f"🔥🔥🔥 {message}")
				elif level == 'HIGH':
					print(f"🔥🔥 {message}")
				elif level == 'MEDIUM':
					print(f"🔥 {message}")
				
				# 예측 정보가 있으면 추가 출력
				if 'expected_time' in alert_item:
					expected_time = alert_item['expected_time']
					time_until = expected_time - datetime.datetime.now()
					hours = int(time_until.total_seconds() // 3600)
					minutes = int((time_until.total_seconds() % 3600) // 60)
					print(f"⏰ 예상 시간: {expected_time.strftime('%H:%M:%S')} (약 {hours}시간 {minutes}분 후)")
			
			print("=" * 60)
			
		except Exception as e:
			print(f"❌ 폭등/폭락 경고 처리 오류: {e}")
	
	def _print_status(self):
		"""상태 출력"""
		liquidation_stats = self.websocket.get_liquidation_stats(5)
		volume_analysis = self.websocket.get_volume_analysis(3)
		signal_summary = self.integrated_strategy.get_signal_summary()
		
		# 예측 요약 정보
		prediction_summary = self.integrated_strategy.prediction_strategy.get_prediction_summary()
		
		print(f"\n📊 통합 상태 - {datetime.datetime.now().strftime('%H:%M:%S')}")
		print(f"🔥 최근 5분 청산: {liquidation_stats['total_count']}개 (${liquidation_stats['total_value']:,.0f})")
		print(f"📈 거래량 트렌드: {volume_analysis['volume_trend']} ({volume_analysis['volume_ratio']:.1f}x)")
		print(f"🎯 총 신호: {self.signal_count}개 | 🔥🔥🔥 시너지: {self.synergy_count}개")
		print(f"🔮 예측 신호: {len(prediction_summary.get('current_predictions', []))}개 | 정확도: {prediction_summary.get('accuracy', 0):.1%}")
		print(f"⚙️ 하이브리드: {'활성' if signal_summary['config']['enable_hybrid'] else '비활성'}")
		print(f"⚙️ 청산: {'활성' if signal_summary['config']['enable_liquidation'] else '비활성'}")
		print(f"⚙️ 시너지: {'활성' if signal_summary['config']['enable_synergy'] else '비활성'}")
		print(f"⚙️ 예측: {'활성' if self.config.enable_liquidation_prediction else '비활성'}")
		
		# 현재 포지션 청산 정보 표시
		self._print_position_liquidation_info()
		self._print_liquidation_density_analysis()
		
		if self.last_signal_time:
			time_since = datetime.datetime.now() - self.last_signal_time
			print(f"⏰ 마지막 신호: {time_since.total_seconds():.0f}초 전")
		
		if self.last_5min_analysis:
			time_since = datetime.datetime.now() - self.last_5min_analysis
			print(f"⏰ 마지막 5분봉 분석: {time_since.total_seconds():.0f}초 전")
		
		# 현재 예측 신호 출력
		current_predictions = prediction_summary.get('current_predictions', [])
		if current_predictions:
			# 현재 가격 가져오기
			current_price = self.websocket.price_history[-1]['price'] if self.websocket.price_history else 0
			
			print(f"\n🔮 현재 예측 신호 (현재가: ${current_price:.2f}):")
			for i, pred in enumerate(current_predictions[:3]):  # 상위 3개만
				pred_type = pred.get('type', 'UNKNOWN')
				confidence = pred.get('confidence', 0)
				target_price = pred.get('target_price', 0)
				
				if current_price > 0 and target_price > 0:
					# 퍼센트 변화 계산
					price_change = ((target_price - current_price) / current_price) * 100
					change_sign = "+" if price_change > 0 else ""
					
					if pred_type == 'EXPLOSION_UP':
						print(f"  {i+1}. 🚀 폭등 예측: ${target_price:.2f} ({change_sign}{price_change:.2f}%) | 신뢰도: {confidence:.1%}")
					elif pred_type == 'EXPLOSION_DOWN':
						print(f"  {i+1}. 💥 폭락 예측: ${target_price:.2f} ({change_sign}{price_change:.2f}%) | 신뢰도: {confidence:.1%}")
				else:
					# 가격 정보가 없을 때 기본 출력
					if pred_type == 'EXPLOSION_UP':
						print(f"  {i+1}. 🚀 폭등 예측: ${target_price:.2f} | 신뢰도: {confidence:.1%}")
					elif pred_type == 'EXPLOSION_DOWN':
						print(f"  {i+1}. 💥 폭락 예측: ${target_price:.2f} | 신뢰도: {confidence:.1%}")
	
	def _print_position_liquidation_info(self):
		"""현재 포지션 청산 정보 출력"""
		position_summary = self.timing_strategy.get_position_summary()
		active_positions = position_summary.get('active_positions', 0)
		
		if active_positions > 0:
			print(f"\n📊 현재 포지션 청산 정보:")
			positions = self.timing_strategy.active_positions
			
			for pos_id, position in positions.items():
				bias = position.get('bias', 'UNKNOWN')
				entry_price = position.get('entry_price', 0)
				stop_loss = position.get('stop_loss', 0)
				take_profit1 = position.get('take_profit1', 0)
				take_profit2 = position.get('take_profit2', 0)
				size = position.get('size', 0)
				atr = position.get('atr', 0)
				
				if bias == "LONG":
					print(f"  📈 LONG #{pos_id}: ${entry_price:.2f}")
					print(f"     🔴 손절가: ${stop_loss:.2f} (청산 위험)")
					if atr:
						trailing_stop = position.get('high_price', entry_price) - (atr * 2.0)
						print(f"     🔄 트레일링: ${trailing_stop:.2f}")
					print(f"     💎 익절1: ${take_profit1:.2f} | 익절2: ${take_profit2:.2f}")
					
				elif bias == "SHORT":
					print(f"  📉 SHORT #{pos_id}: ${entry_price:.2f}")
					print(f"     🔴 손절가: ${stop_loss:.2f} (청산 위험)")
					if atr:
						trailing_stop = position.get('low_price', entry_price) + (atr * 2.0)
						print(f"     🔄 트레일링: ${trailing_stop:.2f}")
					print(f"     💎 익절1: ${take_profit1:.2f} | 익절2: ${take_profit2:.2f}")
				
				# 현재 가격과의 거리 계산
				if self.websocket.price_history:
					current_price = self.websocket.price_history[-1]['price']
					if bias == "LONG":
						stop_distance = ((entry_price - stop_loss) / entry_price) * 100
						print(f"     ⚠️  손절까지: {stop_distance:.2f}% (${current_price:.2f})")
					elif bias == "SHORT":
						stop_distance = ((stop_loss - entry_price) / entry_price) * 100
						print(f"     ⚠️  손절까지: {stop_distance:.2f}% (${current_price:.2f})")
	
	def _print_liquidation_density_analysis(self):
		"""청산 밀도 분석 출력"""
		if not self.websocket.price_history:
			return
		
		current_price = self.websocket.price_history[-1]['price']
		density_analysis = self.websocket.get_liquidation_density_analysis(current_price, 3.0)
		
		if density_analysis['total_liquidations'] == 0:
			return
		
		print(f"\n🔥 청산 밀도 분석 (±3% 범위):")
		print(f"  💰 현재 가격: ${current_price:.2f}")
		print(f"  📊 범위: ${density_analysis['range_min']:.2f} ~ ${density_analysis['range_max']:.2f}")
		print(f"  🔥 총 청산: {density_analysis['total_liquidations']}개")
		print(f"  🎯 최대 밀도: ${density_analysis['max_density_price']:.2f}")
		print(f"  📈 최대 밀도 물량: {density_analysis['max_density_volume']:.2f} ETH")
		print(f"  💵 최대 밀도 가치: ${density_analysis['max_density_value']:,.0f}")
		
		# 상위 5개 청산 밀도 가격대
		if density_analysis['price_levels']:
			print(f"\n  📊 상위 청산 밀도 가격대:")
			for i, level in enumerate(density_analysis['price_levels'][:5]):
				price = level['price']
				total_vol = level['total_volume']
				long_vol = level['long_volume']
				short_vol = level['short_volume']
				long_count = level['long_count']
				short_count = level['short_count']
				total_value = level['total_value']
				
				# 현재 가격과의 거리
				distance_pct = ((price - current_price) / current_price) * 100
				distance_sign = "+" if distance_pct > 0 else ""
				
				print(f"    {i+1}. ${price:.2f} ({distance_sign}{distance_pct:.2f}%)")
				print(f"       📈 롱 청산: {long_vol:.2f} ETH ({long_count}개)")
				print(f"       📉 숏 청산: {short_vol:.2f} ETH ({short_count}개)")
				print(f"       💰 총 가치: ${total_value:,.0f}")
				
				# 청산 밀도 해석 (가격 방향과 청산 방향을 모두 고려)
				distance_pct = ((price - current_price) / current_price) * 100
				
				if long_vol > short_vol * 1.5:
					if distance_pct > 0:
						# 가격이 올라간 가격대에서 롱 청산 우세 → 매도 압력 증가 (가격 하락 압력)
						print(f"       🔍 해석: 롱 청산 우세 (가격상승구간에서 롱청산 = 매도압력 증가)")
					else:
						# 가격이 내려간 가격대에서 롱 청산 우세 → 매도 압력 증가 (가격 하락 압력)
						print(f"       🔍 해석: 롱 청산 우세 (가격하락구간에서 롱청산 = 매도압력 증가)")
				elif short_vol > long_vol * 1.5:
					if distance_pct > 0:
						# 가격이 올라간 가격대에서 숏 청산 우세 → 매수 압력 증가 (가격 상승 압력)
						print(f"       🔍 해석: 숏 청산 우세 (가격상승구간에서 숏청산 = 매수압력 증가)")
					else:
						# 가격이 내려간 가격대에서 숏 청산 우세 → 매수 압력 증가 (가격 상승 압력)
						print(f"       🔍 해석: 숏 청산 우세 (가격하락구간에서 숏청산 = 매수압력 증가)")
				else:
					print(f"       🔍 해석: 균형 (방향성 불명확)")
	
	def _print_current_liquidation_density(self):
		"""현재 호가 ±3% 범위 청산 밀도 분석 출력 - 가격 방향에 따른 청산 분석"""
		if not self.websocket.price_history:
			return
		
		current_price = self.websocket.price_history[-1]['price']
		density_analysis = self.websocket.get_liquidation_density_analysis(current_price, 3.0)
		
		if density_analysis['total_liquidations'] == 0:
			return
		
		# 가격 방향에 따른 청산 분석
		# 가격이 올라간 가격대 (+%) → 숏 포지션들이 청산 (숏청산)
		# 가격이 내려간 가격대 (-%) → 롱 포지션들이 청산 (롱청산)
		
		# 숏청산 최고 레벨 찾기 (가격이 올라간 가격대에서 숏 포지션들이 청산)
		max_short_liquidation_level = None
		max_short_liquidation_volume = 0
		
		# 롱청산 최고 레벨 찾기 (가격이 내려간 가격대에서 롱 포지션들이 청산)
		max_long_liquidation_level = None
		max_long_liquidation_volume = 0
		
		for level in density_analysis['price_levels']:
			price = level['price']
			distance_pct = ((price - current_price) / current_price) * 100
			
			# 가격이 올라간 가격대 (+%) → 숏 포지션들이 청산
			if distance_pct > 0:
				short_vol = level.get('short_volume', 0)
				if short_vol > max_short_liquidation_volume:
					max_short_liquidation_volume = short_vol
					max_short_liquidation_level = level
			
			# 가격이 내려간 가격대 (-%) → 롱 포지션들이 청산
			elif distance_pct < 0:
				long_vol = level.get('long_volume', 0)
				if long_vol > max_long_liquidation_volume:
					max_long_liquidation_volume = long_vol
					max_long_liquidation_level = level
		
		# 숏청산 최고 레벨 출력 (가격 상승 시 숏 포지션들이 청산)
		if max_short_liquidation_level and max_short_liquidation_level.get('short_volume', 0) > 0:
			short_distance_pct = ((max_short_liquidation_level['price'] - current_price) / current_price) * 100
			short_distance_sign = "+" if short_distance_pct > 0 else ""
			short_value = max_short_liquidation_level['short_volume'] * current_price
			print(f"📉 숏청산최고: ${max_short_liquidation_level['price']:.2f} ({short_distance_sign}{short_distance_pct:.2f}%) | {max_short_liquidation_level['short_volume']:.1f} ETH | ${short_value:,.0f} | 💡 가격상승시 숏청산 = 매수압력")
		
		# 롱청산 최고 레벨 출력 (가격 하락 시 롱 포지션들이 청산)
		if max_long_liquidation_level and max_long_liquidation_level.get('long_volume', 0) > 0:
			long_distance_pct = ((max_long_liquidation_level['price'] - current_price) / current_price) * 100
			long_distance_sign = "+" if long_distance_pct > 0 else ""
			long_value = max_long_liquidation_level['long_volume'] * current_price
			print(f"📈 롱청산최고: ${max_long_liquidation_level['price']:.2f} ({long_distance_sign}{long_distance_pct:.2f}%) | {max_long_liquidation_level['long_volume']:.1f} ETH | ${long_value:,.0f} | 💡 가격하락시 롱청산 = 매도압력")
	
	def start(self):
		"""트레이더 시작"""
		print(f"🚀 {self.config.symbol} 통합 스마트 자동 트레이더 시작!")
		print(f"📊 하이브리드 전략: {'활성' if self.config.enable_hybrid_strategy else '비활성'}")
		print(f"🔥 청산 전략: {'활성' if self.config.enable_liquidation_strategy else '비활성'}")
		print(f"🎯 시너지 신호: {'활성' if self.config.enable_synergy_signals else '비활성'}")
		print(f"🔮 청산 예측: {'활성' if self.config.enable_liquidation_prediction else '비활성'}")
		print(f"⏰ 모드: {'주기(5m)' if self.config.use_periodic_hybrid else '실시간'}")
		print(f"🔇 조용한 모드: {'활성' if self.quiet_mode else '비활성'}")
		print(f"📈 신호 민감도: 높음 (신뢰도 임계값: {self.config.hybrid_min_confidence:.1%})")
		print(f"📊 주기적 분석: 10초마다 (스캘핑용 - API 제한 고려)")
		print(f"📊 거래량 급증 집계: 30초마다 요약 출력 (개별 출력 제한)")
		print(f"💰 가격 변동 감지: 0.1% 이상 (스캘핑용)")
		print(f"🛡️ API 제한 보호: 분당 최대 1200회 (제한 도달 시 5초 대기)")
		print(f"🔥 청산 임계값: {self.config.liquidation_min_count}개, ${self.config.liquidation_min_value:,.0f}")
		print(f"🔮 예측 설정: 밀도 {self.config.prediction_min_density}개, 연쇄 {self.config.prediction_cascade_threshold}개")
		print(f"⏰ 최대 보유시간: {self.config.timing_max_hold_time_hours}시간 (스캘핑용)")
		print("=" * 60)
		print("💡 실시간 분석 중... 신호가 나올 때만 알림을 표시합니다.")
		print("💡 거래량 급증은 3.0x 이상일 때만 감지됩니다 (노이즈 감소).")
		print("💡 거래량 급증은 30초마다 요약해서 표시됩니다.")
		print("💡 스캘핑 최적화: 0.1% 가격 변동 감지, 10초마다 분석, 8초 쿨다운")
		print("💡 API 제한 보호: 분당 1200회 초과 시 자동으로 5초 대기")
		print("=" * 60)
		
		self.running = True
		
		# 웹소켓 백그라운드 시작
		self.websocket.start_background()
		
		# 하이브리드 분석 스레드 (옵션)
		if self.config.use_periodic_hybrid:
			self.hybrid_thread = threading.Thread(target=self._run_hybrid_analysis, daemon=True)
			self.hybrid_thread.start()
		
		# 메인 루프
		try:
			last_technical_analysis = None
			api_call_count = 0
			last_api_reset = datetime.datetime.now()
			max_api_calls_per_minute = 1200  # 바이낸스 분당 최대 호출 제한 (안전하게 설정)
			
			while self.running:
				now = datetime.datetime.now()
				
				# API 호출 제한 체크 (1분마다 리셋)
				if (now - last_api_reset).total_seconds() >= 60:
					api_call_count = 0
					last_api_reset = now
				
				# 주기적 기술적 분석 (10초마다 - 스캘핑용, API 제한 고려)
				if (not last_technical_analysis or 
					(now - last_technical_analysis).total_seconds() > 10):
					
					# API 호출 제한 체크
					if api_call_count < max_api_calls_per_minute:
						self._analyze_realtime_technical()
						last_technical_analysis = now
						api_call_count += 1
					else:
						# API 제한 도달 시 5초 대기
						if not last_technical_analysis or (now - last_technical_analysis).total_seconds() > 5:
							print(f"⚠️ API 호출 제한 도달, 5초 대기 중... ({api_call_count}/분)")
							self._analyze_realtime_technical()
							last_technical_analysis = now
							api_call_count += 1
			
				# 통계 출력 (5분마다)
				if (not self.last_signal_time or 
					now - self.last_signal_time > datetime.timedelta(minutes=5)):
					
					self._print_status()
					time.sleep(300)  # 5분 대기
				else:
					time.sleep(1)
					
		except KeyboardInterrupt:
			print("\n⏹️ 사용자에 의해 중지됨")
		finally:
			self.stop()
	
	def stop(self):
		"""트레이더 중지"""
		self.running = False
		self.websocket.stop()
		print("🛑 통합 스마트 자동 트레이더 중지됨")


def main():
	"""메인 함수"""
	config = IntegratedConfig()
	trader = IntegratedSmartTrader(config)
	trader.start()

if __name__ == "__main__":
	main()
