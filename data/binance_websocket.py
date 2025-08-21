import json
import asyncio
import websockets
import threading
import time
from typing import Dict, List, Callable, Optional
from datetime import datetime
import logging

class BinanceWebSocket:
    """바이낸스 웹소켓 클라이언트 - 실시간 청산 데이터 수집"""
    
    def __init__(self, symbol: str = "ETHUSDT"):
        self.symbol = symbol.lower()
        self.ws_url = "wss://fstream.binance.com/ws"
        self.running = False
        self.callbacks = {
            'liquidation': [],
            'price': [],
            'volume': [],
            'kline': []
        }
        
        # 데이터 저장소
        self.liquidations = []
        self.price_history = []
        self.volume_spikes = []
        self.kline_data = []
        
        # 설정
        self.max_liquidations = 1000  # 최대 저장 청산 데이터 수
        self.max_price_history = 1000  # 최대 저장 가격 데이터 수
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_callback(self, event_type: str, callback: Callable):
        """콜백 함수 등록"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable):
        """콜백 함수 제거"""
        if event_type in self.callbacks:
            if callback in self.callbacks[event_type]:
                self.callbacks[event_type].remove(callback)
    
    async def connect_liquidation_stream(self):
        """청산 데이터 스트림 연결"""
        uri = f"{self.ws_url}/{self.symbol}@forceOrder"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.logger.info(f"청산 스트림 연결됨: {self.symbol}")
                
                async for message in websocket:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        await self.process_liquidation(data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON 파싱 오류: {e}")
                    except Exception as e:
                        self.logger.error(f"청산 데이터 처리 오류: {e}")
                        
        except Exception as e:
            self.logger.error(f"청산 스트림 연결 오류: {e}")
    
    async def connect_kline_stream(self):
        """K라인 데이터 스트림 연결"""
        uri = f"{self.ws_url}/{self.symbol}@kline_1m"
        
        try:
            async with websockets.connect(uri) as websocket:
                self.logger.info(f"K라인 스트림 연결됨: {self.symbol}")
                
                async for message in websocket:
                    if not self.running:
                        break
                    
                    try:
                        data = json.loads(message)
                        await self.process_kline(data)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON 파싱 오류: {e}")
                    except Exception as e:
                        self.logger.error(f"K라인 데이터 처리 오류: {e}")
                        
        except Exception as e:
            self.logger.error(f"K라인 스트림 연결 오류: {e}")
    
    async def process_liquidation(self, data: Dict):
        """청산 데이터 처리"""
        try:
            if 'o' in data:  # 청산 이벤트
                # qty_usd 계산 (수량 × 가격)
                qty_usd = float(data['o']['q']) * float(data['o']['p'])
                
                liquidation = {
                    'timestamp': datetime.now(),
                    'symbol': data['o']['s'],
                    'side': data['o']['S'],  # BUY/SELL
                    'quantity': float(data['o']['q']),
                    'price': float(data['o']['p']),
                    'qty_usd': qty_usd,  # USD 기준 청산 금액
                    'time': data['o']['T']
                }
                
                # 데이터 저장
                self.liquidations.append(liquidation)
                if len(self.liquidations) > self.max_liquidations:
                    self.liquidations.pop(0)
                
                # 콜백 실행
                for callback in self.callbacks['liquidation']:
                    try:
                        callback(liquidation)
                    except Exception as e:
                        self.logger.error(f"청산 콜백 실행 오류: {e}")
                                
        except Exception as e:
            self.logger.error(f"청산 데이터 처리 오류: {e}")
    
    async def process_kline(self, data: Dict):
        """K라인 데이터 처리"""
        try:
            if 'k' in data:  # K라인 데이터
                kline = data['k']
                kline_data = {
                    'timestamp': datetime.now(),
                    'open_time': kline['t'],
                    'close_time': kline['T'],
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'is_closed': kline['x']
                }
                
                # 데이터 저장
                self.kline_data.append(kline_data)
                if len(self.kline_data) > self.max_price_history:
                    self.kline_data.pop(0)
                
                # 가격 데이터 저장
                price_data = {
                    'timestamp': datetime.now(),
                    'price': kline_data['close'],
                    'volume': kline_data['volume']
                }
                self.price_history.append(price_data)
                if len(self.price_history) > self.max_price_history:
                    self.price_history.pop(0)
                
                # 거래량 급증 감지
                if len(self.price_history) >= 20:  # 더 긴 기간으로 평균 계산
                    recent_volumes = [p['volume'] for p in self.price_history[-20:]]
                    current_volume = recent_volumes[-1]
                    
                    # 최근 5개 vs 이전 15개 평균 비교
                    recent_avg = sum(recent_volumes[-5:]) / 5
                    earlier_avg = sum(recent_volumes[:-5]) / 15
                    
                    if current_volume > earlier_avg * 3.0:  # 거래량 3.0배 이상 급증 (기존 1.8x에서 조정)
                        # 가격 방향성 분석 (더 민감하게)
                        recent_prices = [p['price'] for p in self.price_history[-5:]]
                        price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
                        
                        # 거래량 방향성 판단 (임계값 낮춤)
                        if price_change > 0.0005:  # 0.05% 이상 상승
                            direction = "📈 상승 압력"
                            trend = "SHORT_LIQUIDATION"  # 가격 상승 시 숏 청산
                        elif price_change < -0.0005:  # 0.05% 이상 하락
                            direction = "📉 하락 압력"
                            trend = "LONG_LIQUIDATION"  # 가격 하락 시 롱 청산
                        else:
                            # 가격 변화가 미미할 때는 거래량 패턴으로 판단
                            if current_volume > recent_avg * 2.5:  # 매우 강한 거래량
                                # 최근 가격 움직임으로 미세한 방향성 파악
                                last_3_prices = recent_prices[-3:]
                                if len(last_3_prices) >= 3:
                                    micro_trend = (last_3_prices[-1] - last_3_prices[0]) / last_3_prices[0]
                                    if micro_trend > 0.0002:  # 0.02% 이상
                                        direction = "📈 약한 상승 압력"
                                        trend = "SHORT_LIQUIDATION"
                                    elif micro_trend < -0.0002:  # 0.02% 이상
                                        direction = "📉 약한 하락 압력"
                                        trend = "LONG_LIQUIDATION"
                                    else:
                                        direction = "➡️ 중립 압력"
                                        trend = "NEUTRAL"
                                else:
                                    direction = "➡️ 중립 압력"
                                    trend = "NEUTRAL"
                            else:
                                direction = "➡️ 중립 압력"
                                trend = "NEUTRAL"
                        
                        volume_spike = {
                            'timestamp': datetime.now(),
                            'price': price_data['price'],
                            'volume': current_volume,
                            'avg_volume': earlier_avg,
                            'ratio': current_volume / earlier_avg,
                            'direction': direction,
                            'trend': trend,
                            'price_change_pct': price_change * 100,
                            'recent_avg': recent_avg,
                            'micro_trend': (recent_prices[-1] - recent_prices[-3]) / recent_prices[-3] * 100 if len(recent_prices) >= 3 else 0
                        }
                        self.volume_spikes.append(volume_spike)
                        
                        # 콜백 실행
                        for callback in self.callbacks['volume']:
                            try:
                                callback(volume_spike)
                            except Exception as e:
                                self.logger.error(f"거래량 급증 콜백 실행 오류: {e}")
                
                # 콜백 실행
                for callback in self.callbacks['kline']:
                    try:
                        callback(kline_data)
                    except Exception as e:
                        self.logger.error(f"K라인 콜백 실행 오류: {e}")
                
                # 가격 콜백 실행
                for callback in self.callbacks['price']:
                    try:
                        callback(price_data)
                    except Exception as e:
                        self.logger.error(f"가격 콜백 실행 오류: {e}")
                        
        except Exception as e:
            self.logger.error(f"K라인 데이터 처리 오류: {e}")
    
    def get_recent_liquidations(self, minutes: int = 5) -> List[Dict]:
        """최근 N분간의 청산 데이터 반환"""
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        return [liq for liq in self.liquidations if liq['timestamp'].timestamp() > cutoff_time]
    
    def get_liquidation_stats(self, minutes: int = 5) -> Dict:
        """청산 통계 반환"""
        recent_liquidations = self.get_recent_liquidations(minutes)
        
        if not recent_liquidations:
            return {
                'total_count': 0,
                'buy_count': 0,
                'sell_count': 0,
                'total_quantity': 0,
                'avg_price': 0,
                'total_value': 0
            }
        
        buy_liquidations = [liq for liq in recent_liquidations if liq['side'] == 'BUY']
        sell_liquidations = [liq for liq in recent_liquidations if liq['side'] == 'SELL']
        
        total_quantity = sum(liq['quantity'] for liq in recent_liquidations)
        total_value = sum(liq['quantity'] * liq['price'] for liq in recent_liquidations)
        avg_price = total_value / total_quantity if total_quantity > 0 else 0
        
        return {
            'total_count': len(recent_liquidations),
            'buy_count': len(buy_liquidations),
            'sell_count': len(sell_liquidations),
            'total_quantity': total_quantity,
            'avg_price': avg_price,
            'total_value': total_value,
            'buy_ratio': len(buy_liquidations) / len(recent_liquidations) if recent_liquidations else 0,
            'sell_ratio': len(sell_liquidations) / len(recent_liquidations) if recent_liquidations else 0
        }
    
    def get_volume_analysis(self, minutes: int = 5) -> Dict:
        """거래량 분석 반환"""
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        recent_prices = [p for p in self.price_history if p['timestamp'].timestamp() > cutoff_time]
        
        if len(recent_prices) < 2:
            return {
                'volume_trend': 'neutral',
                'volume_ratio': 1.0,
                'price_volatility': 0.0
            }
        
        volumes = [p['volume'] for p in recent_prices]
        prices = [p['price'] for p in recent_prices]
        
        # 거래량 트렌드
        recent_avg = sum(volumes[-5:]) / len(volumes[-5:]) if len(volumes) >= 5 else volumes[-1]
        earlier_avg = sum(volumes[:-5]) / len(volumes[:-5]) if len(volumes) >= 10 else volumes[0]
        
        volume_ratio = recent_avg / earlier_avg if earlier_avg > 0 else 1.0
        
        if volume_ratio > 1.5:
            volume_trend = 'increasing'
        elif volume_ratio < 0.7:
            volume_trend = 'decreasing'
        else:
            volume_trend = 'stable'
        
        # 가격 변동성
        price_changes = [abs(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        price_volatility = sum(price_changes) / len(price_changes) if price_changes else 0
        
        return {
            'volume_trend': volume_trend,
            'volume_ratio': volume_ratio,
            'price_volatility': price_volatility,
            'recent_volume': recent_avg,
            'earlier_volume': earlier_avg
        }
    
    def get_liquidation_density_analysis(self, current_price: float, range_pct: float = 3.0) -> Dict:
        """현재 가격 ±N% 이내의 청산 밀도 분석"""
        if not self.liquidations:
            return {
                'current_price': current_price,
                'range_pct': range_pct,
                'total_liquidations': 0,
                'price_levels': [],
                'max_density_price': None,
                'max_density_volume': 0
            }
        
        # 가격 범위 계산
        min_price = current_price * (1 - range_pct / 100)
        max_price = current_price * (1 + range_pct / 100)
        
        # 범위 내 청산 데이터 필터링
        range_liquidations = [
            liq for liq in self.liquidations 
            if min_price <= liq['price'] <= max_price
        ]
        
        if not range_liquidations:
            return {
                'current_price': current_price,
                'range_pct': range_pct,
                'total_liquidations': 0,
                'price_levels': [],
                'max_density_price': None,
                'max_density_volume': 0
            }
        
        # 가격별 청산 물량 집계 (0.1% 단위로 그룹화)
        price_bins = {}
        bin_size = current_price * 0.001  # 0.1% 단위
        
        for liq in range_liquidations:
            # 가격을 0.1% 단위로 반올림
            bin_price = round(liq['price'] / bin_size) * bin_size
            bin_key = f"{bin_price:.2f}"
            
            if bin_key not in price_bins:
                price_bins[bin_key] = {
                    'price': bin_price,
                    'total_volume': 0,
                    'long_volume': 0,
                    'short_volume': 0,
                    'long_count': 0,
                    'short_count': 0,
                    'total_value': 0
                }
            
            volume = liq['quantity']
            price_bins[bin_key]['total_volume'] += volume
            price_bins[bin_key]['total_value'] += volume * liq['price']
            
            if liq['side'] == 'BUY':  # 숏 청산
                price_bins[bin_key]['short_volume'] += volume
                price_bins[bin_key]['short_count'] += 1
            else:  # 롱 청산
                price_bins[bin_key]['long_volume'] += volume
                price_bins[bin_key]['long_count'] += 1
        
        # 청산 밀도 순으로 정렬
        price_levels = sorted(
            price_bins.values(), 
            key=lambda x: x['total_volume'], 
            reverse=True
        )
        
        # 상위 10개만 반환
        top_levels = price_levels[:10]
        
        # 최대 밀도 가격 찾기
        max_density_level = max(price_levels, key=lambda x: x['total_volume'])
        
        return {
            'current_price': current_price,
            'range_pct': range_pct,
            'total_liquidations': len(range_liquidations),
            'price_levels': top_levels,
            'max_density_price': max_density_level['price'],
            'max_density_volume': max_density_level['total_volume'],
            'max_density_value': max_density_level['total_value'],
            'range_min': min_price,
            'range_max': max_price
        }
    
    async def start(self):
        """웹소켓 스트림 시작"""
        self.running = True
        self.logger.info("웹소켓 스트림 시작")
        
        # 여러 스트림을 동시에 실행
        tasks = [
            self.connect_liquidation_stream(),
            self.connect_kline_stream()
        ]
        
        await asyncio.gather(*tasks)
    
    def stop(self):
        """웹소켓 스트림 중지"""
        self.running = False
        self.logger.info("웹소켓 스트림 중지")
    
    def start_background(self):
        """백그라운드에서 웹소켓 실행"""
        def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start())
        
        self.thread = threading.Thread(target=run_async, daemon=True)
        self.thread.start()
        self.logger.info("백그라운드 웹소켓 시작됨")
    
    def start_liquidation_stream(self):
        """청산 스트림만 시작 (실시간 청산 데이터 수집)"""
        def run_liquidation():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.connect_liquidation_stream())
        
        self.liquidation_thread = threading.Thread(target=run_liquidation, daemon=True)
        self.liquidation_thread.start()
        self.logger.info("청산 스트림 시작됨")
