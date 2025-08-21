#!/usr/bin/env python3
"""
외부 데이터 로더 테스트 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from integrated_smart_trader import ExternalDataLoader
from config.integrated_config import IntegratedConfig

def test_external_data_loader():
    """외부 데이터 로더 테스트"""
    print("🧪 외부 데이터 로더 테스트를 시작합니다...")
    
    # 설정 로드
    config = IntegratedConfig()
    
    # 외부 데이터 로더 초기화
    data_loader = ExternalDataLoader(
        server_url=config.external_server_url,
        api_key=config.external_api_key
    )
    
    print(f"📡 서버 URL: {config.external_server_url}")
    print(f"🔑 API 키: {'설정됨' if config.external_api_key else '설정되지 않음'}")
    print(f"⏰ 초기 데이터 시간: {config.initial_data_hours}시간")
    
    # 외부 서버에서 데이터 가져오기 테스트
    print("\n🔄 외부 서버에서 데이터를 가져오는 중...")
    liquidation_data = data_loader.fetch_initial_liquidation_data(
        symbol=config.symbol,
        hours_back=config.initial_data_hours
    )
    
    if liquidation_data:
        print(f"✅ 성공적으로 {len(liquidation_data)}개의 데이터를 가져왔습니다.")
        
        # 첫 번째 데이터 샘플 출력
        if len(liquidation_data) > 0:
            print("\n📊 첫 번째 데이터 샘플:")
            sample = liquidation_data[0]
            for key, value in sample.items():
                print(f"  {key}: {value}")
        
        # 로컬 데이터베이스에 저장 테스트
        print("\n💾 로컬 데이터베이스에 저장하는 중...")
        data_loader.save_to_local_database(liquidation_data)
        
    else:
        print("❌ 외부 서버에서 데이터를 가져올 수 없습니다.")
        print("💡 외부 서버 URL과 API 키를 확인해주세요.")

def test_mock_data():
    """모의 데이터로 테스트"""
    print("\n🧪 모의 데이터로 테스트를 시작합니다...")
    
    # 모의 청산 데이터 생성
    import time
    mock_data = []
    current_time = int(time.time())
    
    for i in range(10):
        mock_data.append({
            'timestamp': current_time - (i * 3600),  # 1시간씩 이전
            'symbol': 'ETHUSDT',
            'side': 'long' if i % 2 == 0 else 'short',
            'qty_usd': 50000 + (i * 10000),
            'price': 2000 + (i * 10)
        })
    
    print(f"📊 모의 데이터 {len(mock_data)}개 생성 완료")
    
    # 로컬 데이터베이스에 저장 테스트
    data_loader = ExternalDataLoader()
    data_loader.save_to_local_database(mock_data, "test_liquidation_data.db")
    
    print("✅ 모의 데이터 테스트 완료")

if __name__ == "__main__":
    try:
        # 실제 외부 서버 테스트
        test_external_data_loader()
        
        # 모의 데이터 테스트
        test_mock_data()
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
