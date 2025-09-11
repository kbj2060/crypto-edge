#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
실시간 미래 예측 대시보드
- 웹 기반 실시간 업데이트 대시보드
- 3분봉 데이터와 전략 예측을 실시간으로 시각화
"""

import asyncio
import json
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# 웹 서버 관련
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import plotly.graph_objs as go
import plotly.utils
from plotly.subplots import make_subplots

# 프로젝트 컴포넌트들
from utils.future_predictor import FuturePredictor, PredictionPoint
from data.strategy_executor import StrategyExecutor
from engines.trade_decision_engine import TradeDecisionEngine
from core.trader_core import TraderCore
from config.integrated_config import IntegratedConfig

class RealtimeDashboard:
    """실시간 미래 예측 대시보드"""
    
    def __init__(self, symbol: str = "ETHUSDC"):
        self.symbol = symbol
        import os
        template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
        self.app = Flask(__name__, template_folder=template_dir)
        self.app.config['SECRET_KEY'] = 'crypto_edge_dashboard'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # 컴포넌트들
        self.predictor = FuturePredictor()
        self.strategy_executor = StrategyExecutor()
        self.decision_engine = TradeDecisionEngine()
        self.trader_core = None
        self.websocket = None
        
        # 데이터 저장소
        self.current_data = {
            'price': 0.0,
            'timestamp': None,
            'predictions': [],
            'signals': {},
            'decision': {},
            'historical_data': pd.DataFrame()
        }
        
        # 업데이트 상태
        self.is_running = False
        self.update_interval = 3  # 3분마다 업데이트
        
        self._setup_routes()
        self._setup_socket_events()
    
    def _setup_routes(self):
        """Flask 라우트 설정"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html', symbol=self.symbol)
        
        @self.app.route('/api/data')
        def get_data():
            """현재 데이터 API"""
            return jsonify(self.current_data)
        
        @self.app.route('/api/predictions')
        def get_predictions():
            """예측 데이터 API"""
            return jsonify({
                'predictions': [
                    {
                        'timestamp': p.timestamp.isoformat(),
                        'price': p.price,
                        'confidence': p.confidence,
                        'strategy_type': p.strategy_type,
                        'action': p.action,
                        'market_context': p.market_context,
                        'net_score': p.net_score
                    }
                    for p in self.current_data['predictions']
                ]
            })
        
        @self.app.route('/api/update', methods=['POST'])
        def force_update():
            """강제 업데이트 API"""
            self._update_predictions()
            return jsonify({'status': 'success'})
    
    def _setup_socket_events(self):
        """SocketIO 이벤트 설정"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"🔌 클라이언트 연결됨: {request.sid}")
            emit('status', {'message': '대시보드에 연결되었습니다.'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"🔌 클라이언트 연결 해제됨: {request.sid}")
        
        @self.socketio.on('request_update')
        def handle_update_request():
            self._update_predictions()
            emit('data_updated', self.current_data)
    
    def _update_predictions(self):
        """예측 데이터 업데이트"""
        try:
            # 현재 신호 가져오기
            signals = self.strategy_executor.get_signals()
            
            # 의사결정 엔진 실행
            decision = self.decision_engine.decide_trade_realtime(signals)
            
            # 현재 가격 (실제로는 웹소켓에서 가져와야 함)
            current_price = self.current_data.get('price', 3000.0)
            
            # 미래 예측 생성
            predictions = self.predictor.generate_predictions(signals, current_price)
            
            # 데이터 업데이트
            self.current_data.update({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'signals': signals,
                'decision': decision,
                'predictions': predictions
            })
            
            # 클라이언트에 업데이트 전송
            self.socketio.emit('data_updated', {
                'timestamp': self.current_data['timestamp'],
                'price': current_price,
                'predictions_count': len(predictions),
                'decision': decision
            })
            
            print(f"✅ 예측 업데이트 완료: {len(predictions)}개 예측 포인트")
            
        except Exception as e:
            print(f"❌ 예측 업데이트 오류: {e}")
            self.socketio.emit('error', {'message': f'업데이트 오류: {str(e)}'})
    
    def _create_plotly_charts(self) -> Dict[str, str]:
        """Plotly 차트 생성"""
        charts = {}
        
        # 1. 메인 가격 차트
        if not self.current_data['historical_data'].empty:
            hist_data = self.current_data['historical_data'].tail(100)
            
            fig = go.Figure()
            
            # 과거 가격 라인
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=hist_data['close'],
                mode='lines',
                name='과거 가격',
                line=dict(color='blue', width=2)
            ))
            
            # 현재 가격 라인
            current_price = hist_data['close'].iloc[-1]
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="red",
                annotation_text=f"현재 가격: ${current_price:.2f}"
            )
            
            # 예측 포인트들
            if self.current_data['predictions']:
                pred_df = pd.DataFrame([(p.timestamp, p.price, p.strategy_type, p.confidence, p.action) 
                                      for p in self.current_data['predictions']],
                                     columns=['timestamp', 'price', 'strategy_type', 'confidence', 'action'])
                
                colors = {'SHORT_TERM': 'green', 'MEDIUM_TERM': 'orange', 'LONG_TERM': 'purple'}
                
                for strategy_type in ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']:
                    strategy_data = pred_df[pred_df['strategy_type'] == strategy_type]
                    if not strategy_data.empty:
                        fig.add_trace(go.Scatter(
                            x=strategy_data['timestamp'],
                            y=strategy_data['price'],
                            mode='markers',
                            name=f'{strategy_type} 예측',
                            marker=dict(
                                color=colors[strategy_type],
                                size=8,
                                opacity=strategy_data['confidence'].values
                            ),
                            hovertemplate=f'<b>{strategy_type}</b><br>' +
                                        '시간: %{x}<br>' +
                                        '가격: $%{y:.2f}<br>' +
                                        '신뢰도: %{marker.opacity:.2f}<extra></extra>'
                        ))
            
            fig.update_layout(
                title='🚀 미래 가격 예측',
                xaxis_title='시간',
                yaxis_title='가격 (USDC)',
                hovermode='x unified',
                template='plotly_white'
            )
            
            charts['main_chart'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # 2. 신뢰도 히트맵
        if self.current_data['predictions']:
            strategies = ['SHORT_TERM', 'MEDIUM_TERM', 'LONG_TERM']
            time_slots = pd.date_range(
                start=datetime.now(timezone.utc),
                end=datetime.now(timezone.utc) + timedelta(hours=24),
                freq='3min'
            )
            
            confidence_matrix = np.zeros((len(strategies), len(time_slots)))
            
            for pred in self.current_data['predictions']:
                strategy_idx = strategies.index(pred.strategy_type)
                time_idx = time_slots.get_indexer([pred.timestamp], method='nearest')[0]
                if 0 <= time_idx < len(time_slots):
                    confidence_matrix[strategy_idx, time_idx] = pred.confidence
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=confidence_matrix,
                x=time_slots,
                y=strategies,
                colorscale='RdYlGn',
                zmin=0,
                zmax=1,
                hovertemplate='시간: %{x}<br>전략: %{y}<br>신뢰도: %{z:.2f}<extra></extra>'
            ))
            
            fig_heatmap.update_layout(
                title='🔥 예측 신뢰도 히트맵',
                xaxis_title='시간',
                yaxis_title='전략',
                template='plotly_white'
            )
            
            charts['heatmap'] = json.dumps(fig_heatmap, cls=plotly.utils.PlotlyJSONEncoder)
        
        return charts
    
    def start_data_loader(self):
        """데이터 로더 시작"""
        def data_loader_callback():
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
                
                print("✅ 데이터 로더 초기화 완료")
                
                # 4. 초기 데이터 로드 및 예측 생성
                self._load_and_update_predictions()
                
            except Exception as e:
                print(f"❌ 데이터 로더 초기화 오류: {e}")
                import traceback
                traceback.print_exc()
        
        # 데이터 로더를 별도 스레드에서 실행
        dl_thread = threading.Thread(target=data_loader_callback, daemon=True)
        dl_thread.start()
    
    def _load_and_update_predictions(self):
        """데이터 로드 및 예측 업데이트"""
        try:
            # 최근 24시간 3분봉 데이터 로드
            df = self.data_loader.fetch_recent_3m(self.symbol, 24)
            
            if df is None or df.empty:
                print("❌ 데이터 로드 실패")
                return
            
            # 현재 가격 업데이트
            current_price = df['close'].iloc[-1]
            self.current_data['price'] = current_price
            self.current_data['timestamp'] = datetime.now(timezone.utc).isoformat()
            
            # 과거 데이터를 예측기에 추가
            self.predictor.add_historical_data(df)
            
            # 전략 신호 생성
            self.strategy_executor.execute_all_strategies()
            signals = self.strategy_executor.get_signals()
            
            # 예측 업데이트
            self._update_predictions()
            
            print(f"✅ 데이터 로드 및 예측 업데이트 완료: ${current_price:.2f}")
            
        except Exception as e:
            print(f"❌ 데이터 로드 및 예측 업데이트 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def start_auto_update(self):
        """자동 업데이트 시작"""
        def auto_update_loop():
            while self.is_running:
                time.sleep(self.update_interval * 60)  # 분 단위
                if self.is_running:
                    self._load_and_update_predictions()
        
        self.is_running = True
        update_thread = threading.Thread(target=auto_update_loop, daemon=True)
        update_thread.start()
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """대시보드 실행"""
        print(f"🚀 실시간 미래 예측 대시보드 시작...")
        print(f"📊 심볼: {self.symbol}")
        print(f"🌐 URL: http://{host}:{port}")
        
        # 데이터 로더 시작
        self.start_data_loader()
        
        # 자동 업데이트 시작
        self.start_auto_update()
        
        # Flask 앱 실행
        self.socketio.run(self.app, host=host, port=port, debug=debug)

def create_dashboard_template():
    """대시보드 HTML 템플릿 생성"""
    template_content = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 미래 예측 대시보드</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .dashboard {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .info-panel {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .status {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #4CAF50;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .metric-label {
            font-weight: bold;
            color: #666;
        }
        .metric-value {
            color: #333;
        }
        .btn {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 미래 예측 대시보드</h1>
        <p>실시간 3분봉 데이터와 AI 전략 예측</p>
    </div>

    <div class="status">
        <div>
            <span class="status-indicator"></span>
            <span id="status-text">연결됨</span>
        </div>
        <div>
            <button class="btn" onclick="requestUpdate()">🔄 업데이트</button>
            <button class="btn" onclick="toggleAutoUpdate()">⏸️ 자동업데이트</button>
        </div>
    </div>

    <div class="dashboard">
        <div class="chart-container">
            <h3>📈 가격 예측 차트</h3>
            <div id="main-chart"></div>
        </div>
        
        <div class="info-panel">
            <h3>📊 실시간 정보</h3>
            <div class="metric">
                <span class="metric-label">현재 가격:</span>
                <span class="metric-value" id="current-price">$0.00</span>
            </div>
            <div class="metric">
                <span class="metric-label">예측 포인트:</span>
                <span class="metric-value" id="prediction-count">0</span>
            </div>
            <div class="metric">
                <span class="metric-label">마지막 업데이트:</span>
                <span class="metric-value" id="last-update">-</span>
            </div>
            <div class="metric">
                <span class="metric-label">활성 신호:</span>
                <span class="metric-value" id="active-signals">0</span>
            </div>
        </div>
    </div>

    <div class="chart-container">
        <h3>🔥 신뢰도 히트맵</h3>
        <div id="heatmap"></div>
    </div>

    <script>
        const socket = io();
        let autoUpdate = true;

        // 소켓 이벤트 리스너
        socket.on('connect', function() {
            console.log('서버에 연결됨');
            updateStatus('연결됨', true);
        });

        socket.on('disconnect', function() {
            console.log('서버 연결 해제');
            updateStatus('연결 해제', false);
        });

        socket.on('data_updated', function(data) {
            console.log('데이터 업데이트:', data);
            updateDisplay(data);
        });

        socket.on('error', function(error) {
            console.error('오류:', error);
            updateStatus('오류 발생', false);
        });

        // 상태 업데이트
        function updateStatus(message, connected) {
            document.getElementById('status-text').textContent = message;
            const indicator = document.querySelector('.status-indicator');
            indicator.style.backgroundColor = connected ? '#4CAF50' : '#f44336';
        }

        // 디스플레이 업데이트
        function updateDisplay(data) {
            document.getElementById('current-price').textContent = `$${data.price.toFixed(2)}`;
            document.getElementById('prediction-count').textContent = data.predictions_count;
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            
            if (data.decision && data.decision.decisions) {
                const totalSignals = Object.values(data.decision.decisions)
                    .reduce((sum, cat) => sum + (cat.strategies_used || 0), 0);
                document.getElementById('active-signals').textContent = totalSignals;
            }
        }

        // 업데이트 요청
        function requestUpdate() {
            socket.emit('request_update');
        }

        // 자동 업데이트 토글
        function toggleAutoUpdate() {
            autoUpdate = !autoUpdate;
            const btn = event.target;
            btn.textContent = autoUpdate ? '⏸️ 자동업데이트' : '▶️ 자동업데이트';
            btn.style.backgroundColor = autoUpdate ? '#007bff' : '#28a745';
        }

        // 초기 데이터 로드
        fetch('/api/data')
            .then(response => response.json())
            .then(data => {
                updateDisplay(data);
                loadCharts();
            });

        // 차트 로드
        function loadCharts() {
            fetch('/api/predictions')
                .then(response => response.json())
                .then(data => {
                    // 여기에 Plotly 차트 생성 코드 추가
                    console.log('예측 데이터:', data);
                });
        }
    </script>
</body>
</html>
    """
    
    # templates 디렉토리 생성
    import os
    os.makedirs('templates', exist_ok=True)
    
    with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    print("✅ 대시보드 템플릿 생성 완료")

def main():
    """메인 실행 함수"""
    # 템플릿 생성
    create_dashboard_template()
    
    # 대시보드 실행
    dashboard = RealtimeDashboard("ETHUSDC")
    dashboard.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main()
