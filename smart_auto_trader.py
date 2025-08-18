#!/usr/bin/env python3
"""
ETHUSDT 스마트 자동 트레이더
언제 사고 팔아야 할지 판단하는 알고리즘 + 실시간 알림 시스템
"""

import time
import datetime
import pandas as pd
from typing import Dict, Any, Optional
from data.loader import build_df
from indicators.vpvr import vpvr_key_levels
from signals.hybrid_strategy import make_hybrid_trade_plan, HybridConfig
from signals.timing_strategy import TimingStrategy, TimingConfig

class SmartAutoTrader:
    def __init__(self):
        self.symbol = "ETHUSDT"
        self.last_signal = None
        self.signal_history = []
        self.position_history = []
        
        # 하이브리드 전략 설정 (완전한 설정)
        self.hybrid_cfg = HybridConfig(
            min_hybrid_confidence=0.20,
            min_vpvr_headroom=0.001,
            trend_weight=0.4,
            entry_weight=0.6,
            atr_len=14,
            atr_stop_mult=1.0,
            atr_tp1_mult=1.5,
            atr_tp2_mult=2.5,
            vpvr_bins=50,
            vpvr_lookback=200
        )
        
        # 타이밍 전략 설정 (더 완화된 조건)
        self.timing_cfg = TimingConfig(
            entry_confidence_min=0.20,  # 0.25 → 0.20으로 완화
            entry_rr_min=0.15,  # 0.2 → 0.15로 완화
            entry_score_threshold=0.35,  # 0.4 → 0.35로 완화
            max_hold_time_hours=24,
            trailing_stop_atr=2.0
        )
        
        # 타이밍 전략 인스턴스
        self.timing_strategy = TimingStrategy(self.timing_cfg)
        
        # 알림 설정
        self.notification_enabled = True
        self.alert_sound = True  # 소리 알림 (향후 구현)
        
    def send_notification(self, message: str, signal_type: str = "INFO", urgency: str = "NORMAL"):
        """향상된 알림 전송 시스템"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 이모지와 색상 구분
        if signal_type == "STRONG_BUY":
            emoji = "🚀"
            prefix = f"[{emoji} 강력한 매수 신호!]"
            urgency_level = "🔥🔥🔥"
        elif signal_type == "BUY":
            emoji = "📈"
            prefix = f"[{emoji} 매수 신호]"
            urgency_level = "🔥🔥"
        elif signal_type == "STRONG_SELL":
            emoji = "📉"
            prefix = f"[{emoji} 강력한 매도 신호!]"
            urgency_level = "🔥🔥🔥"
        elif signal_type == "SELL":
            emoji = "📉"
            prefix = f"[{emoji} 매도 신호]"
            urgency_level = "🔥🔥"
        elif signal_type == "EXIT":
            emoji = "💰"
            prefix = f"[{emoji} 청산 신호]"
            urgency_level = "🔥"
        elif signal_type == "ALERT":
            emoji = "⚠️"
            prefix = f"[{emoji} 경고]"
            urgency_level = "⚠️"
        else:
            emoji = "ℹ️"
            prefix = f"[{emoji} 정보]"
            urgency_level = "ℹ️"
        
        # 긴급도에 따른 구분선
        if urgency == "HIGH":
            separator = "🔥" * 20
        elif urgency == "MEDIUM":
            separator = "=" * 60
        else:
            separator = "-" * 60
        
        print(f"\n{prefix} {timestamp}")
        print(f"{urgency_level} {urgency} {urgency_level}")
        print(f"{separator}")
        print(message)
        print(f"{separator}")
        
        # 신호 히스토리에 저장
        if signal_type in ["STRONG_BUY", "BUY", "STRONG_SELL", "SELL", "EXIT"]:
            self.signal_history.append({
                "timestamp": timestamp,
                "signal": signal_type,
                "urgency": urgency,
                "message": message
            })
    
    def analyze_market(self) -> Optional[Dict[str, Any]]:
        """시장 분석 및 거래 신호 생성"""
        try:
            # 데이터 로드
            df_15m = build_df(
                self.symbol, 
                self.hybrid_cfg.interval_15m, 
                self.hybrid_cfg.limit_15m, 
                self.hybrid_cfg.atr_len, 
                market="futures", 
                price_source="last", 
                ma_type="ema"
            )
            
            df_5m = build_df(
                self.symbol, 
                self.hybrid_cfg.interval_5m, 
                self.hybrid_cfg.limit_5m, 
                self.hybrid_cfg.atr_len, 
                market="futures", 
                price_source="last", 
                ma_type="ema"
            )
            
            if df_15m.empty or df_5m.empty:
                print("❌ 데이터가 비어있음")
                self.send_notification("데이터 로딩 실패", "ALERT", "HIGH")
                return None
            
            # VPVR 레벨 계산
            vpvr_levels = vpvr_key_levels(
                df_15m, 
                bins=self.hybrid_cfg.vpvr_bins, 
                lookback=min(self.hybrid_cfg.vpvr_lookback, len(df_15m)), 
                topn=8
            )
            
            # 하이브리드 거래 계획 생성
            plan = make_hybrid_trade_plan(df_15m, df_5m, vpvr_levels, self.hybrid_cfg)
            
            return plan
            
        except Exception as e:
            print(f"❌ 시장 분석 중 오류: {e}")
            self.send_notification(f"시장 분석 중 오류 발생: {str(e)}", "ALERT", "HIGH")
            return None
    
    def analyze_timing(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """타이밍 전략 분석"""
        if not plan:
            return None
        
        # 새로운 신호 형식에 맞춰 수정
        final_signal = plan.get('final_signal', 'NEUTRAL')
        current_price = plan.get('current_price', 0)
        confidence = plan.get('confidence', 0)
        risk_reward = plan.get('risk_reward', 0)
        
        # NEUTRAL 신호인 경우 대기
        if final_signal == 'NEUTRAL':
            return {
                "action": "WAIT",
                "reason": "NEUTRAL 신호 - 거래 조건 미충족",
                "entry_score": 0.0
            }
        
        # 신뢰도 및 리스크/보상 조건 확인
        if confidence < self.timing_cfg.entry_confidence_min:
            return {
                "action": "WAIT",
                "reason": f"신뢰도 부족: {confidence:.3f} < {self.timing_cfg.entry_confidence_min}",
                "entry_score": confidence
            }
        
        if risk_reward < self.timing_cfg.entry_rr_min:
            return {
                "action": "WAIT",
                "reason": f"리스크/보상 부족: {risk_reward:.3f} < {self.timing_cfg.entry_rr_min}",
                "entry_score": confidence
            }
        
        # 진입 점수 계산
        entry_score = (confidence * 0.6 + risk_reward * 0.4)
        
        # 진입 결정
        if entry_score >= self.timing_cfg.entry_score_threshold:
            action = "BUY" if final_signal == "BUY" else "SELL"
            return {
                "action": action,
                "reason": f"진입 조건 충족 (점수: {entry_score:.3f})",
                "entry_score": entry_score,
                "bias": final_signal,
                "entry_price": current_price,
                "stop_loss": plan.get('stop_loss'),
                "take_profit1": plan.get('take_profit1'),
                "take_profit2": plan.get('take_profit2'),
                "atr": plan.get('atr'),
                "timestamp": datetime.datetime.now()
            }
        else:
            return {
                "action": "WAIT",
                "reason": f"진입 점수 부족: {entry_score:.3f} < {self.timing_cfg.entry_score_threshold}",
                "entry_score": entry_score
            }
    
    def execute_trades(self, timing_analysis: Dict[str, Any]):
        """거래 실행 및 알림"""
        # 진입 신호 처리
        entry = timing_analysis.get('entry', {})
        if entry.get('action') in ['STRONG_BUY', 'BUY', 'STRONG_SELL', 'SELL']:
            self._handle_entry_signal(entry)
        
        # 청산 신호 처리
        exits = timing_analysis.get('exits', [])
        for exit_signal in exits:
            self._handle_exit_signal(exit_signal)
    
    def _handle_entry_signal(self, entry_signal: Dict[str, Any]):
        """진입 신호 처리"""
        action = entry_signal.get('action')
        bias = entry_signal.get('bias')
        confidence = entry_signal.get('confidence', 0)
        rr_ratio = entry_signal.get('risk_reward_ratio', 0)
        entry_price = entry_signal.get('entry_price', 0)
        
        # 포지션 오픈
        position_id = self.timing_strategy.open_position(entry_signal)
        
        # 알림 메시지 구성
        if action in ['STRONG_BUY', 'BUY']:
            urgency = "HIGH" if action == "STRONG_BUY" else "MEDIUM"
            message = self._format_buy_signal(entry_signal, position_id)
            self.send_notification(message, action, urgency)
        else:  # SELL
            urgency = "HIGH" if action == "STRONG_SELL" else "MEDIUM"
            message = self._format_sell_signal(entry_signal, position_id)
            self.send_notification(message, action, urgency)
    
    def _handle_exit_signal(self, exit_signal: Dict[str, Any]):
        """청산 신호 처리"""
        position_id = exit_signal['position_id']
        analysis = exit_signal['analysis']
        
        action = analysis.get('action')
        reason = analysis.get('reason', '')
        pnl = analysis.get('pnl', 0)
        
        # 포지션 클로즈
        position = self.timing_strategy.close_position(position_id, reason, pnl)
        
        if position:
            # 알림 메시지 구성
            message = self._format_exit_signal(analysis, position, pnl)
            urgency = "HIGH" if abs(pnl) > 0.02 else "MEDIUM"  # 2% 이상 손익 시 높은 긴급도
            self.send_notification(message, "EXIT", urgency)
    
    def _format_buy_signal(self, entry_signal: Dict[str, Any], position_id: str) -> str:
        """매수 신호 메시지 포맷"""
        action = entry_signal.get('action')
        confidence = entry_signal.get('confidence', 0)
        rr_ratio = entry_signal.get('risk_reward_ratio', 0)
        entry_price = entry_signal.get('entry_price', 0)
        stop_loss = entry_signal.get('stop_loss', 0)
        tp1 = entry_signal.get('take_profit1', 0)
        tp2 = entry_signal.get('take_profit2', 0)
        entry_score = entry_signal.get('entry_score', 0)
        
        message = f"""
🚀 ETHUSDT {action} 신호!

💰 현재 가격: ${entry_price:.2f}
📊 신뢰도: {confidence:.2f} ({confidence*100:.1f}%)
⚖️ 리스크/보상: {rr_ratio:.2f}
🎯 진입 점수: {entry_score:.2f}

🎯 진입 레벨: ${entry_price:.2f}
🛑 손절가: ${stop_loss:.2f}
💎 익절1: ${tp1:.2f}
💎 익절2: ${tp2:.2f}

📈 포지션 ID: {position_id}
⏰ 진입 시간: {entry_signal.get('timestamp', 'N/A')}

💡 거래 전략:
  • 즉시 진입: ${entry_price:.2f}
  • 손절: ${stop_loss:.2f} (손실 위험: ${(entry_price - stop_loss):.2f})
  • 익절1: ${tp1:.2f} (수익: ${(tp1 - entry_price):.2f})
  • 익절2: ${tp2:.2f} (수익: ${(tp2 - entry_price):.2f})

🔍 분석 이유:
"""
        
        # 분석 이유 추가
        reasons = entry_signal.get('reasons', [])
        for i, reason in enumerate(reasons[:5], 1):
            message += f"  {i}. {reason}\n"
        
        if len(reasons) > 5:
            message += f"  ... 및 {len(reasons)-5}개 더\n"
        
        return message
    
    def _format_sell_signal(self, entry_signal: Dict[str, Any], position_id: str) -> str:
        """매도 신호 메시지 포맷"""
        action = entry_signal.get('action')
        confidence = entry_signal.get('confidence', 0)
        rr_ratio = entry_signal.get('risk_reward_ratio', 0)
        entry_price = entry_signal.get('entry_price', 0)
        stop_loss = entry_signal.get('stop_loss', 0)
        tp1 = entry_signal.get('take_profit1', 0)
        tp2 = entry_signal.get('take_profit2', 0)
        entry_score = entry_signal.get('entry_score', 0)
        
        message = f"""
📉 ETHUSDT {action} 신호!

💰 현재 가격: ${entry_price:.2f}
📊 신뢰도: {confidence:.2f} ({confidence*100:.1f}%)
⚖️ 리스크/보상: {rr_ratio:.2f}
🎯 진입 점수: {entry_score:.2f}

🎯 진입 레벨: ${entry_price:.2f}
🛑 손절가: ${stop_loss:.2f}
💎 익절1: ${tp1:.2f}
💎 익절2: ${tp2:.2f}

📈 포지션 ID: {position_id}
⏰ 진입 시간: {entry_signal.get('timestamp', 'N/A')}

💡 거래 전략:
  • 즉시 진입: ${entry_price:.2f}
  • 손절: ${stop_loss:.2f} (손실 위험: ${(stop_loss - entry_price):.2f})
  • 익절1: ${tp1:.2f} (수익: ${(entry_price - tp1):.2f})
  • 익절2: ${tp2:.2f} (수익: ${(entry_price - tp2):.2f})

🔍 분석 이유:
"""
        
        # 분석 이유 추가
        reasons = entry_signal.get('reasons', [])
        for i, reason in enumerate(reasons[:5], 1):
            message += f"  {i}. {reason}\n"
        
        if len(reasons) > 5:
            message += f"  ... 및 {len(reasons)-5}개 더\n"
        
        return message
    
    def _format_exit_signal(self, analysis: Dict[str, Any], position: Dict[str, Any], pnl: float) -> str:
        """청산 신호 메시지 포맷"""
        action = analysis.get('action')
        reason = analysis.get('reason', '')
        bias = position.get('bias', 'UNKNOWN')
        entry_price = position.get('entry_price', 0)
        current_price = position.get('current_price', entry_price)
        hold_time = position.get('hold_time', 'N/A')  # 보유 시간 가져오기
        
        pnl_percent = pnl * 100
        pnl_emoji = "📈" if pnl > 0 else "📉"
        
        message = f"""
💰 ETHUSDT 청산 신호!

📊 청산 유형: {action}
🎯 청산 이유: {reason}
📈 포지션: {bias}

💰 진입가: ${entry_price:.2f}
💵 청산가: ${current_price:.2f}
{pnl_emoji} 손익: {pnl_percent:.2f}% (${pnl:.4f})

⏰ 보유 시간: {hold_time}
📅 진입 시간: {position.get('timestamp', 'N/A')}

💡 청산 전략:
  • {action} 실행
  • 포지션 완전 청산
  • 손익 실현
"""
        
        return message
    
    def run_analysis(self):
        """5분봉 분석 실행"""
        try:
            print(f"\n⏰ {datetime.datetime.now().strftime('%H:%M:%S')} - 분석 시작")
            
            # 데이터 로딩
            df_15m = build_df('ETHUSDT', '15m', 200, 14, market='futures', price_source='last', ma_type='ema')
            df_5m = build_df('ETHUSDT', '5m', 300, 14, market='futures', price_source='last', ma_type='ema')
            
            if df_15m.empty or df_5m.empty:
                print("❌ 데이터 로딩 실패")
                return
            
            # VPVR 레벨 계산
            vpvr_levels = vpvr_key_levels(df_15m, self.hybrid_cfg.vpvr_bins, self.hybrid_cfg.vpvr_lookback, topn=8)
            
            # 하이브리드 전략 분석
            plan = make_hybrid_trade_plan(df_15m, df_5m, vpvr_levels, self.hybrid_cfg)
            
            if not plan:
                print("❌ 전략 분석 실패")
                return
            
            final_signal = plan.get('final_signal')
            confidence = plan.get('confidence', 0)
            risk_reward = plan.get('risk_reward', 0)
            
            # 신호 조건 확인
            if (final_signal != "NEUTRAL" and 
                confidence >= self.timing_cfg.entry_confidence_min and 
                risk_reward >= self.timing_cfg.entry_rr_min):
                
                # 타이밍 분석
                timing_analysis = self.analyze_timing(plan)
                
                if timing_analysis and timing_analysis.get('action') in ['BUY', 'SELL']:
                    self.execute_trade(plan, timing_analysis)
                else:
                    print("⏳ 타이밍 조건 미충족")
            else:
                print(f"⏳ 신호 조건 미충족: {final_signal} | {confidence:.1%} | {risk_reward:.1f}")
                
        except Exception as e:
            print(f"❌ 분석 오류: {e}")
    
    def execute_trade(self, plan: Dict[str, Any], timing_analysis: Dict[str, Any]):
        """거래 실행"""
        action = timing_analysis.get('action')
        
        if action == "BUY":
            self.send_buy_signal(plan, timing_analysis)
            position_id = self.timing_strategy.open_position(timing_analysis)
            print(f"📈 포지션 오픈: {position_id}")
        elif action == "SELL":
            self.send_sell_signal(plan, timing_analysis)
            position_id = self.timing_strategy.open_position(timing_analysis)
            print(f"📉 포지션 오픈: {position_id}")
        
        # 포지션 요약
        position_summary = self.timing_strategy.get_position_summary()
        if position_summary['active_positions'] > 0:
            print(f"📊 활성 포지션: {position_summary['active_positions']}개 | 💰 일일 손익: {position_summary['daily_pnl']:.4f}")
        
        print(f"✅ {datetime.datetime.now().strftime('%H:%M:%S')} - 분석 완료")
    
    def get_next_5min_candle_time(self) -> datetime.datetime:
        """다음 5분봉 업데이트 시점 계산"""
        now = datetime.datetime.now()
        
        # 현재 분을 5로 나눈 몫에 1을 더하고 5를 곱해서 다음 5분 단위 시점 계산
        next_minute = ((now.minute // 5) + 1) * 5
        
        if next_minute >= 60:
            next_minute = 0
            next_hour = now.hour + 1
            if next_hour >= 24:
                next_hour = 0
                next_day = now.day + 1
            else:
                next_day = now.day
        else:
            next_hour = now.hour
            next_day = now.day
        
        # 다음 5분봉 시점 (초는 0으로 설정)
        next_candle = now.replace(
            day=next_day,
            hour=next_hour,
            minute=next_minute,
            second=0,
            microsecond=0
        )
        
        return next_candle
    
    def wait_until_next_candle(self):
        """다음 5분봉까지 대기"""
        next_candle = self.get_next_5min_candle_time()
        now = datetime.datetime.now()
        
        # 다음 5분봉까지의 대기 시간 계산 (초 단위)
        wait_seconds = (next_candle - now).total_seconds()
        
        if wait_seconds > 0:
            print(f"⏳ 다음 5분봉까지 대기 중... ({next_candle.strftime('%H:%M:%S')})")
            
            # 실시간으로 대기시간 카운트다운 (같은 줄에서 업데이트)
            while wait_seconds > 0:
                # 1초마다 업데이트
                time.sleep(1)
                wait_seconds -= 1
                
                # 남은 시간을 같은 줄에서 업데이트
                if wait_seconds > 60:
                    minutes = int(wait_seconds // 60)
                    seconds = int(wait_seconds % 60)
                    print(f"\r   남은 시간: {minutes:02d}:{seconds:02d}", end="", flush=True)
                else:
                    print(f"\r   남은 시간: {wait_seconds:.0f}초", end="", flush=True)
            
            # 카운트다운 완료 후 줄바꿈
            print()
        
        # 5분봉 업데이트 후 1초 대기 (데이터 안정화)
        print(f"🔄 5분봉 업데이트 완료! 1초 후 분석 시작...")
        
        # 1초 카운트다운 (같은 줄에서 업데이트)
        print(f"\r   분석 시작까지: 1초", end="", flush=True)
        time.sleep(1)
        
        # 카운트다운 완료 후 줄바꿈
        print()
    
    def start(self):
        """스마트 자동 트레이더 시작"""
        print(f"\n🚀 ETHUSDT 스마트 자동 트레이더 시작!")
        print(f"📊 심볼: {self.symbol}")
        print(f"⏰ 실행 주기: 매 5분봉 업데이트 후 1초")
        print(f"🎯 진입 조건: 신뢰도 ≥{self.timing_cfg.entry_confidence_min}, R/R ≥{self.timing_cfg.entry_rr_min}")
        print(f"💰 리스크 관리: 최대 포지션 {self.timing_cfg.max_position_size*100}%, 일일 손실 제한 {self.timing_cfg.max_daily_loss*100}%")
        print(f"🔄 트레일링 스탑: {'사용' if self.timing_cfg.trailing_stop else '사용 안함'}")
        print(f"{'='*60}")
        
        # 즉시 첫 번째 분석 실행
        self.run_analysis()
        
        # 무한 루프로 5분봉마다 실행
        try:
            while True:
                # 다음 5분봉까지 대기
                self.wait_until_next_candle()
                
                # 분석 실행
                self.run_analysis()
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️ 스마트 자동 트레이더 중지됨")
            self._print_summary()
    
    def _print_summary(self):
        """신호 히스토리 및 포지션 요약 출력"""
        # 신호 히스토리
        if self.signal_history:
            print(f"\n📊 신호 히스토리 요약:")
            print(f"{'='*60}")
            
            strong_buy = [s for s in self.signal_history if s['signal'] == 'STRONG_BUY']
            buy = [s for s in self.signal_history if s['signal'] == 'BUY']
            strong_sell = [s for s in self.signal_history if s['signal'] == 'STRONG_SELL']
            sell = [s for s in self.signal_history if s['signal'] == 'SELL']
            exits = [s for s in self.signal_history if s['signal'] == 'EXIT']
            
            print(f"🚀 강력한 매수: {len(strong_buy)}개")
            print(f"📈 매수: {len(buy)}개")
            print(f"📉 강력한 매도: {len(strong_sell)}개")
            print(f"📉 매도: {len(sell)}개")
            print(f"💰 청산: {len(exits)}개")
            print(f"📈 총 신호: {len(self.signal_history)}개")
            
            if self.signal_history:
                print(f"\n🕐 최근 신호:")
                for signal in self.signal_history[-5:]:  # 최근 5개
                    print(f"  {signal['timestamp']} - {signal['signal']} ({signal['urgency']})")
        else:
            print(f"\n📊 신호 히스토리: 없음")
        
        # 포지션 요약
        position_summary = self.timing_strategy.get_position_summary()
        print(f"\n💰 포지션 요약:")
        print(f"  활성 포지션: {position_summary['active_positions']}개")
        print(f"  일일 손익: {position_summary['daily_pnl']:.4f}")

    def send_buy_signal(self, plan: Dict[str, Any], timing_analysis: Dict[str, Any]):
        """BUY 신호 알림"""
        current_price = timing_analysis.get('entry_price', 0)
        confidence = plan.get('confidence', 0)
        risk_reward = plan.get('risk_reward', 0)
        stop_loss = timing_analysis.get('stop_loss', 0)
        take_profit1 = timing_analysis.get('take_profit1', 0)
        take_profit2 = timing_analysis.get('take_profit2', 0)
        
        print(f"\n📈 BUY 신호 - {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"💰 ${current_price:.2f} | 📊 {confidence:.1%} | ⚖️ {risk_reward:.1f}")
        print(f"🛑 ${stop_loss:.2f} | 💎 ${take_profit1:.2f} | 💎 ${take_profit2:.2f}")
    
    def send_sell_signal(self, plan: Dict[str, Any], timing_analysis: Dict[str, Any]):
        """SELL 신호 알림"""
        current_price = timing_analysis.get('entry_price', 0)
        confidence = plan.get('confidence', 0)
        risk_reward = plan.get('risk_reward', 0)
        stop_loss = timing_analysis.get('stop_loss', 0)
        take_profit1 = timing_analysis.get('take_profit1', 0)
        take_profit2 = timing_analysis.get('take_profit2', 0)
        
        print(f"\n📉 SELL 신호 - {datetime.datetime.now().strftime('%H:%M:%S')}")
        print(f"💰 ${current_price:.2f} | 📊 {confidence:.1%} | ⚖️ {risk_reward:.1f}")
        print(f"🛑 ${stop_loss:.2f} | 💎 ${take_profit1:.2f} | 💎 ${take_profit2:.2f}")

def main():
    """메인 함수"""
    trader = SmartAutoTrader()
    trader.start()

if __name__ == "__main__":
    main()
