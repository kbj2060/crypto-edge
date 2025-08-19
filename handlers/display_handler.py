#!/usr/bin/env python3
"""
출력 및 표시 핸들러
"""

import datetime
from typing import Dict, List
from data.binance_websocket import BinanceWebSocket


class DisplayHandler:
    """출력 및 표시 핸들러"""
    
    def __init__(self, websocket: BinanceWebSocket):
        self.websocket = websocket
    
    def print_volume_spike_summary(self, volume_buffer: List[Dict]):
        """거래량 급증 요약 출력"""
        if not volume_buffer:
            return
        
        # 방향성 분석
        long_liquidation = 0
        short_liquidation = 0
        neutral_pressure = 0
        
        for spike in volume_buffer:
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
    
    def print_liquidation_density_analysis(self):
        """청산 밀도 분석 출력"""
        if not self.websocket.price_history:
            print("⚠️ 가격 데이터가 없습니다.")
            return
        
        current_price = self.websocket.price_history[-1]['price']
        
        # 디버깅: 청산 데이터 상태 확인
        total_liquidations = len(self.websocket.liquidations) if hasattr(self.websocket, 'liquidations') else 0
        print(f"🔍 청산 데이터 상태: 총 {total_liquidations}개 수집됨")
        
        if total_liquidations == 0:
            print("⚠️ 청산 데이터가 수집되지 않았습니다.")
            return
        
        density_analysis = self.websocket.get_liquidation_density_analysis(current_price, 3.0)
        
        if density_analysis['total_liquidations'] == 0:
            print("⚠️ 현재 가격 ±3% 범위 내 청산 데이터가 없습니다.")
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
                
                # 청산 밀도 해석
                self._interpret_liquidation_density(long_vol, short_vol)
    
    def _interpret_liquidation_density(self, long_vol: float, short_vol: float):
        """청산 밀도 해석"""
        # 롱 청산 우세 (현재 호가보다 아래에서 청산)
        if long_vol > 0:
            print(f"       🔍 해석: 롱 청산 (현재 호가보다 아래에서 청산 = 매도압력 증가)")
        
        # 숏 청산 우세 (현재 호가보다 위에서 청산)
        elif short_vol > 0:
            print(f"       🔍 해석: 숏 청산 (현재 호가보다 위에서 청산 = 매수압력 증가)")
        
        # 둘 다 0인 경우 (청산 없음)
        elif long_vol == 0 and short_vol == 0:
            print(f"       🔍 해석: 청산 없음")
        
        # 예외적인 경우 (둘 다 0이 아닌 경우 - 이론적으로 불가능)
        else:
            print(f"       🔍 해석: 예외 상황 (롱: {long_vol}, 숏: {short_vol})")
    
    def print_current_liquidation_density(self):
        """현재 호가 ±3% 범위 청산 밀도 분석 출력"""
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
            print(f"📈 롱청산최고: ${max_long_liquidation_level['price']:.2f} ({long_distance_sign}{long_distance_pct:.2f}%) | {max_long_liquidation_level['long_volume']:.1f} ETH | 💡 가격하락시 롱청산 = 매도압력")
