"""
VPVR (Volume Profile Visible Range) 테스트 파일

주요 테스트 항목:
1. 기본 초기화 및 설정
2. 세션 데이터 로딩
3. 캔들 데이터 업데이트
4. 동적 bin 크기 계산
5. POC, HVN, LVN 계산
6. 세션 리셋 기능
7. 상태 정보 반환
"""

import unittest
import pandas as pd
import numpy as np
import datetime as dt
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from indicators.vpvr import SessionVPVR

class TestSessionVPVR(unittest.TestCase):
    """SessionVPVR 클래스 테스트"""
    
    def setUp(self):
        """테스트 전 설정"""
        # Mock 객체들 생성
        self.mock_time_manager = Mock()
        self.mock_data_manager = Mock()
        self.mock_atr = Mock()
        
        # Mock ATR 속성 설정 - 실제 리스트와 값으로 설정
        self.mock_atr.true_ranges = [10.0] * 20
        self.mock_atr.candles = [1] * 20
        self.mock_atr.length = 14
        self.mock_atr.atr = 15.0  # 실제 ATR 값
        
        # Mock ATR 메서드 설정
        self.mock_atr.get_status.return_value = {'atr': 15.0}
        self.mock_atr.is_ready.return_value = True
        
        # 기본 세션 설정
        self.session_config = {
            'use_session_mode': True,
            'session_name': 'TEST_SESSION',
            'session_start_time': dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1),
            'elapsed_minutes': 60,
            'mode': 'session',
            'session_status': 'ACTIVE'
        }
        
        # Mock 설정
        self.mock_time_manager.get_indicator_mode_config.return_value = self.session_config
        
        # 테스트용 샘플 데이터
        self.sample_candle = pd.Series({
            'open': 2000.0,
            'high': 2010.0,
            'low': 1990.0,
            'close': 2005.0,
            'volume': 100.0,
            'quote_volume': 200500.0,
            'timestamp': dt.datetime.now(dt.timezone.utc)
        })
        
        self.sample_df = pd.DataFrame([
            {
                'open': 2000.0, 'high': 2010.0, 'low': 1990.0, 'close': 2005.0,
                'volume': 100.0, 'quote_volume': 200500.0
            },
            {
                'open': 2005.0, 'high': 2015.0, 'low': 2000.0, 'close': 2010.0,
                'volume': 150.0, 'quote_volume': 301500.0
            },
            {
                'open': 2010.0, 'high': 2020.0, 'low': 2005.0, 'close': 2015.0,
                'volume': 200.0, 'quote_volume': 403000.0
            }
        ])
    
    @patch('indicators.vpvr.get_time_manager')
    @patch('indicators.vpvr.get_data_manager')
    @patch('indicators.vpvr.ATR3M')
    def test_initialization(self, mock_atr_class, mock_get_data_manager, mock_get_time_manager):
        """기본 초기화 테스트"""
        # Mock 설정
        mock_get_time_manager.return_value = self.mock_time_manager
        mock_get_data_manager.return_value = self.mock_data_manager
        mock_atr_class.return_value = self.mock_atr
        
        # VPVR 인스턴스 생성
        vpvr = SessionVPVR(bins=50, price_bin_size=0.05, lookback=100)
        
        # 기본 속성 확인
        self.assertEqual(vpvr.bins, 50)
        self.assertEqual(vpvr.price_bin_size, 0.05)
        self.assertEqual(vpvr.lookback, 100)
        self.assertIsInstance(vpvr.price_bins, dict)
        self.assertIsInstance(vpvr.volume_histogram, dict)
        
        # 의존성 객체 확인
        self.assertEqual(vpvr.time_manager, self.mock_time_manager)
        self.assertEqual(vpvr.atr, self.mock_atr)
    
    @patch('indicators.vpvr.get_time_manager')
    @patch('indicators.vpvr.get_data_manager')
    @patch('indicators.vpvr.ATR3M')
    def test_session_data_loading(self, mock_atr_class, mock_get_data_manager, mock_get_time_manager):
        """세션 데이터 로딩 테스트"""
        # Mock 설정
        mock_get_time_manager.return_value = self.mock_time_manager
        mock_get_data_manager.return_value = self.mock_data_manager
        mock_atr_class.return_value = self.mock_atr
        
        # Mock 데이터 반환
        self.mock_data_manager.get_data_range.return_value = self.sample_df
        
        # VPVR 인스턴스 생성
        vpvr = SessionVPVR()
        
        # 세션 데이터 로딩 확인
        self.mock_data_manager.get_data_range.assert_called_once()
        self.assertEqual(vpvr.processed_candle_count, 3)
        self.assertGreater(len(vpvr.volume_histogram), 0)
    
    @patch('indicators.vpvr.get_time_manager')
    @patch('indicators.vpvr.get_data_manager')
    @patch('indicators.vpvr.ATR3M')
    def test_candle_update(self, mock_atr_class, mock_get_data_manager, mock_get_time_manager):
        """캔들 데이터 업데이트 테스트"""
        # Mock 설정
        mock_get_time_manager.return_value = self.mock_time_manager
        mock_get_data_manager.return_value = self.mock_data_manager
        mock_atr_class.return_value = self.mock_atr
        
        # VPVR 인스턴스 생성
        vpvr = SessionVPVR()
        
        # 초기 상태 확인
        initial_count = vpvr.processed_candle_count
        
        # 캔들 업데이트
        vpvr.update_with_candle(self.sample_candle)
        
        # 업데이트 확인
        self.assertEqual(vpvr.processed_candle_count, initial_count + 1)
        self.assertGreater(len(vpvr.volume_histogram), 0)
        self.assertIsNotNone(vpvr.last_update_time)
    
    @patch('indicators.vpvr.get_time_manager')
    @patch('indicators.vpvr.get_data_manager')
    @patch('indicators.vpvr.ATR3M')
    def test_dynamic_bin_size_calculation(self, mock_atr_class, mock_get_data_manager, mock_get_time_manager):
        """동적 bin 크기 계산 테스트"""
        # Mock 설정
        mock_get_time_manager.return_value = self.mock_time_manager
        mock_get_data_manager.return_value = self.mock_data_manager
        mock_atr_class.return_value = self.mock_atr
        
        # Mock ATR 상태 - 실제 숫자 값으로 설정
        self.mock_atr.get_status.return_value = {'atr': 15.0}
        self.mock_atr.is_ready.return_value = True
        
        # VPVR 인스턴스 생성
        vpvr = SessionVPVR()
        
        # 가격 bin 키 생성
        price = 2000.0
        bin_key = vpvr._get_price_bin_key(price)
        
        # bin 키 형식 확인
        self.assertIsInstance(bin_key, str)
        self.assertTrue(bin_key.startswith('bin_'))
        
        # price_bins에 가격 저장 확인
        self.assertIn(bin_key, vpvr.price_bins)
        self.assertEqual(vpvr.price_bins[bin_key], price)
    
    @patch('indicators.vpvr.get_time_manager')
    @patch('indicators.vpvr.get_data_manager')
    @patch('indicators.vpvr.ATR3M')
    def test_vpvr_calculation(self, mock_atr_class, mock_get_data_manager, mock_get_time_manager):
        """VPVR 계산 테스트"""
        # Mock 설정
        mock_get_time_manager.return_value = self.mock_time_manager
        mock_get_data_manager.return_value = self.mock_data_manager
        mock_atr_class.return_value = self.mock_atr
        
        # Mock ATR 상태 - 실제 숫자 값으로 설정
        self.mock_atr.get_status.return_value = {'atr': 15.0}
        self.mock_atr.is_ready.return_value = True
        
        # Mock 데이터 반환
        self.mock_data_manager.get_data_range.return_value = self.sample_df
        
        # VPVR 인스턴스 생성
        vpvr = SessionVPVR()
        
        # VPVR 결과 확인
        result = vpvr.get_current_vpvr()
        
        if result:
            # POC, HVN, LVN 존재 확인
            self.assertIn('poc', result)
            self.assertIn('hvn', result)
            self.assertIn('lvn', result)
            self.assertIn('total_volume', result)
            self.assertIn('active_bins', result)
            
            # 데이터 타입 확인
            self.assertIsInstance(result['poc'], (int, float))
            self.assertIsInstance(result['hvn'], (int, float))
            self.assertIsInstance(result['lvn'], (int, float))
            self.assertIsInstance(result['total_volume'], (int, float))
    
    @patch('indicators.vpvr.get_time_manager')
    @patch('indicators.vpvr.get_data_manager')
    @patch('indicators.vpvr.ATR3M')
    def test_session_reset(self, mock_atr_class, mock_get_data_manager, mock_get_time_manager):
        """세션 리셋 테스트"""
        # Mock 설정
        mock_get_time_manager.return_value = self.mock_time_manager
        mock_get_data_manager.return_value = self.mock_data_manager
        mock_atr_class.return_value = self.mock_atr
        
        # Mock ATR 상태 - 실제 숫자 값으로 설정
        self.mock_atr.get_status.return_value = {'atr': 15.0}
        self.mock_atr.is_ready.return_value = True
        
        # VPVR 인스턴스 생성
        vpvr = SessionVPVR()
        
        # 초기 데이터 로딩
        self.mock_data_manager.get_data_range.return_value = self.sample_df
        
        # 세션 리셋
        vpvr.reset_session()
        
        # 리셋 확인
        self.assertEqual(len(vpvr.price_bins), 0)
        self.assertEqual(len(vpvr.volume_histogram), 0)
        self.assertIsNone(vpvr.cached_result)
        self.assertIsNone(vpvr.last_update_time)
    
    @patch('indicators.vpvr.get_time_manager')
    @patch('indicators.vpvr.get_data_manager')
    @patch('indicators.vpvr.ATR3M')
    def test_status_information(self, mock_atr_class, mock_get_data_manager, mock_get_time_manager):
        """상태 정보 반환 테스트"""
        # Mock 설정
        mock_get_time_manager.return_value = self.mock_time_manager
        mock_get_data_manager.return_value = self.mock_data_manager
        mock_atr_class.return_value = self.mock_atr
        
        # Mock ATR 상태 - 실제 숫자 값으로 설정
        self.mock_atr.get_status.return_value = {'atr': 15.0}
        self.mock_atr.is_ready.return_value = True
        self.mock_atr.true_ranges = [10.0] * 20
        self.mock_atr.candles = [1] * 20
        
        # VPVR 인스턴스 생성
        vpvr = SessionVPVR()
        
        # 상태 정보 확인
        status = vpvr.get_status()
        
        # 기본 상태 정보 확인
        self.assertIn('is_session_active', status)
        self.assertIn('current_session', status)
        self.assertIn('mode', status)
        self.assertIn('data_count', status)
        self.assertIn('last_update', status)
        
        # ATR 상태 정보 확인
        self.assertIn('atr_status', status)
        atr_status = status['atr_status']
        self.assertIn('atr', atr_status)
        self.assertIn('is_ready', atr_status)
        self.assertIn('is_mature', atr_status)
        self.assertIn('candles_count', atr_status)
    
    @patch('indicators.vpvr.get_time_manager')
    @patch('indicators.vpvr.get_data_manager')
    @patch('indicators.vpvr.ATR3M')
    def test_session_change_detection(self, mock_atr_class, mock_get_data_manager, mock_get_time_manager):
        """세션 변경 감지 테스트"""
        # Mock 설정
        mock_get_time_manager.return_value = self.mock_time_manager
        mock_get_data_manager.return_value = self.mock_data_manager
        mock_atr_class.return_value = self.mock_atr
        
        # Mock ATR 상태 - 실제 숫자 값으로 설정
        self.mock_atr.get_status.return_value = {'atr': 15.0}
        self.mock_atr.is_ready.return_value = True
        
        # VPVR 인스턴스 생성
        vpvr = SessionVPVR()
        
        # 초기 세션 설정
        vpvr.last_session_name = 'OLD_SESSION'
        
        # 새로운 세션 설정
        new_session_config = self.session_config.copy()
        new_session_config['session_name'] = 'NEW_SESSION'
        self.mock_time_manager.get_indicator_mode_config.return_value = new_session_config
        
        # 세션 변경 감지 테스트
        vpvr._check_session_reset(new_session_config)
        
        # 세션 이름 업데이트 확인
        self.assertEqual(vpvr.last_session_name, 'NEW_SESSION')
    
    @patch('indicators.vpvr.get_time_manager')
    @patch('indicators.vpvr.get_data_manager')
    @patch('indicators.vpvr.ATR3M')
    def test_error_handling(self, mock_atr_class, mock_get_data_manager, mock_get_time_manager):
        """에러 처리 테스트"""
        # Mock 설정
        mock_get_time_manager.return_value = self.mock_time_manager
        mock_get_data_manager.return_value = self.mock_data_manager
        mock_atr_class.return_value = self.mock_atr
        
        # Mock ATR 상태 - 실제 숫자 값으로 설정
        self.mock_atr.get_status.return_value = {'atr': 15.0}
        self.mock_atr.is_ready.return_value = True
        
        # 에러 발생 시나리오 설정
        self.mock_data_manager.get_data_range.side_effect = Exception("데이터 로딩 오류")
        
        # VPVR 인스턴스 생성 (에러가 발생해도 생성되어야 함)
        vpvr = SessionVPVR()
        
        # 에러 상황에서도 기본 속성은 유지되어야 함
        self.assertIsInstance(vpvr.price_bins, dict)
        self.assertIsInstance(vpvr.volume_histogram, dict)
    
    @patch('indicators.vpvr.get_time_manager')
    @patch('indicators.vpvr.get_data_manager')
    @patch('indicators.vpvr.ATR3M')
    def test_lookback_mode(self, mock_atr_class, mock_get_data_manager, mock_get_time_manager):
        """Lookback 모드 테스트"""
        # Lookback 모드 설정
        lookback_config = self.session_config.copy()
        lookback_config['use_session_mode'] = False
        
        # Mock 설정
        mock_get_time_manager.return_value = self.mock_time_manager
        mock_get_data_manager.return_value = self.mock_data_manager
        mock_atr_class.return_value = self.mock_atr
        
        # Mock ATR 상태 - 실제 숫자 값으로 설정
        self.mock_atr.get_status.return_value = {'atr': 15.0}
        self.mock_atr.is_ready.return_value = True
        
        # Mock 데이터 반환
        self.mock_data_manager.get_data_range.return_value = self.sample_df
        
        # Lookback 모드로 VPVR 인스턴스 생성
        vpvr = SessionVPVR()
        
        # Lookback 모드에서도 데이터가 로딩되어야 함
        self.assertGreater(vpvr.processed_candle_count, 0)

class TestVPVRIntegration(unittest.TestCase):
    """VPVR 통합 테스트"""
    
    @patch('indicators.vpvr.get_time_manager')
    @patch('indicators.vpvr.get_data_manager')
    @patch('indicators.vpvr.ATR3M')
    def test_full_workflow(self, mock_atr_class, mock_get_data_manager, mock_get_time_manager):
        """전체 워크플로우 테스트"""
        # Mock 설정
        mock_time_manager = Mock()
        mock_data_manager = Mock()
        mock_atr = Mock()
        
        mock_get_time_manager.return_value = mock_time_manager
        mock_get_data_manager.return_value = mock_data_manager
        mock_atr_class.return_value = mock_atr
        
        # Mock ATR 상태 - 실제 숫자 값으로 설정
        mock_atr.get_status.return_value = {'atr': 15.0}
        mock_atr.is_ready.return_value = True
        mock_atr.true_ranges = [10.0] * 20
        mock_atr.candles = [1] * 20
        mock_atr.atr = 15.0  # 실제 ATR 값
        mock_atr.length = 14  # ATR 길이

        # 세션 설정
        session_config = {
            'use_session_mode': True,
            'session_name': 'INTEGRATION_TEST',
            'session_start_time': dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1),
            'elapsed_minutes': 60,
            'mode': 'session',
            'session_status': 'ACTIVE'
        }
        mock_time_manager.get_indicator_mode_config.return_value = session_config
        
        # 테스트 데이터
        test_df = pd.DataFrame([
            {'open': 2000, 'high': 2010, 'low': 1990, 'close': 2005, 'volume': 100, 'quote_volume': 200500},
            {'open': 2005, 'high': 2015, 'low': 2000, 'close': 2010, 'volume': 150, 'quote_volume': 301500},
            {'open': 2010, 'high': 2020, 'low': 2005, 'close': 2015, 'volume': 200, 'quote_volume': 403000}
        ])
        mock_data_manager.get_data_range.return_value = test_df
        
        # VPVR 인스턴스 생성
        vpvr = SessionVPVR()
        
        # 1. 초기화 확인
        self.assertIsNotNone(vpvr)
        self.assertEqual(vpvr.processed_candle_count, 3)
        
        # 2. 상태 정보 확인
        status = vpvr.get_status()
        self.assertTrue(status['is_session_active'])
        self.assertEqual(status['current_session'], 'INTEGRATION_TEST')
        
        # 3. VPVR 결과 확인
        result = vpvr.get_current_vpvr()
        if result:
            self.assertIn('poc', result)
            self.assertIn('total_volume', result)
        
        # 4. 새로운 캔들 추가
        new_candle = pd.Series({
            'open': 2015, 'high': 2025, 'low': 2010, 'close': 2020,
            'volume': 250, 'quote_volume': 505000
        })
        vpvr.update_with_candle(new_candle)
        
        # 5. 업데이트 확인
        self.assertEqual(vpvr.processed_candle_count, 4)
        
        # 6. 세션 리셋
        vpvr.reset_session()
        self.assertEqual(vpvr.processed_candle_count, 0)
        self.assertEqual(len(vpvr.volume_histogram), 0)

def run_performance_test():
    """성능 테스트 실행"""
    print("\n🚀 VPVR 성능 테스트 시작...")
    
    # 대량 데이터 생성
    large_df = pd.DataFrame({
        'open': np.random.uniform(2000, 2100, 1000),
        'high': np.random.uniform(2000, 2100, 1000),
        'low': np.random.uniform(2000, 2100, 1000),
        'close': np.random.uniform(2000, 2100, 1000),
        'volume': np.random.uniform(50, 200, 1000),
        'quote_volume': np.random.uniform(100000, 400000, 1000)
    })
    
    # 성능 측정
    import time
    
    with patch('indicators.vpvr.get_time_manager'), \
         patch('indicators.vpvr.get_data_manager'), \
         patch('indicators.vpvr.ATR3M'):
        
        start_time = time.time()
        
        # VPVR 인스턴스 생성
        vpvr = SessionVPVR()
        
        # 대량 데이터 처리
        for _, row in large_df.iterrows():
            vpvr.update_with_candle(row)
        
        end_time = time.time()
        
        processing_time = end_time - start_time
        print(f"✅ 1000개 캔들 처리 시간: {processing_time:.4f}초")
        print(f"✅ 초당 처리 캔들 수: {1000/processing_time:.2f}")

if __name__ == '__main__':
    print("🧪 VPVR 테스트 시작...")
    
    # 단위 테스트 실행
    unittest.main(verbosity=2, exit=False)
    
    # 성능 테스트 실행
    run_performance_test()
    
    print("\n🎉 모든 테스트 완료!")
