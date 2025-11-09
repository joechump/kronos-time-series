#!/usr/bin/env python3
"""
专门测试2025年3月29日及前后日期的预测API修复效果
"""

import requests
import json
from datetime import datetime, timedelta

def test_specific_dates():
    """测试特定日期"""
    base_url = "http://localhost:8080"
    stock_code = "000858"
    
    # 测试日期列表
    test_cases = [
        {"date": "2025-03-28", "expected": "工作日，应该成功"},
        {"date": "2025-03-29", "expected": "周六，非交易日，应该自动调整"},
        {"date": "2025-03-30", "expected": "周日，非交易日，应该自动调整"},
        {"date": "2025-03-31", "expected": "周一，工作日，应该成功"},
        {"date": "2025-04-01", "expected": "周二，工作日，应该成功"}
    ]
    
    print("🚀 开始测试2025年3月29日及前后日期的预测API修复")
    print("=" * 70)
    
    for test_case in test_cases:
        date_str = test_case["date"]
        expected = test_case["expected"]
        
        print(f"\n📅 测试日期: {date_str}")
        print(f"📊 预期结果: {expected}")
        
        # 1. 先检查交易日状态
        print("\n1. 交易日检查:")
        try:
            check_url = f"{base_url}/api/trading-calendar/check?date={date_str}"
            response = requests.get(check_url, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                is_trading = result.get('is_trading_day', False)
                print(f"   ✅ 交易日检查成功")
                print(f"   📊 是否为交易日: {is_trading}")
            else:
                print(f"   ❌ 交易日检查失败: {response.status_code}")
                is_trading = False
        except Exception as e:
            print(f"   ❌ 交易日检查异常: {e}")
            is_trading = False
        
        # 2. 测试预测API
        print("\n2. 预测API测试:")
        try:
            predict_url = f"{base_url}/api/predict"
            params = {
                "file_path": f"stock_{stock_code}_live",
                "lookback": 120,
                "pred_len": 30,
                "start_date": date_str
            }
            
            print(f"   📋 请求参数: {json.dumps(params, indent=6)}")
            
            response = requests.post(predict_url, json=params, timeout=30)
            
            print(f"   📡 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ 预测成功!")
                print(f"   💬 消息: {result.get('message', 'N/A')}")
                
                if 'error' in result:
                    print(f"   ⚠️  警告信息: {result['error']}")
                
                # 检查是否包含交易日历信息
                if 'trading_calendar' in result:
                    trading_cal = result['trading_calendar']
                    print(f"   📅 交易日历信息: {len(trading_cal.get('trading_days', []))} 个交易日")
                
            elif response.status_code == 400:
                result = response.json()
                print(f"   ❌ 预测失败 (400错误)")
                print(f"   📋 错误信息: {result.get('error', 'N/A')}")
                print(f"   💡 建议: {result.get('suggestion', 'N/A')}")
            else:
                print(f"   ❌ 预测失败 (状态码: {response.status_code})")
                print(f"   📋 响应内容: {response.text[:200]}...")
                
        except requests.exceptions.Timeout:
            print(f"   ❌ 请求超时")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ 连接错误 - 请检查服务器是否运行")
        except Exception as e:
            print(f"   ❌ 未知异常: {e}")
        
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("📊 测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_specific_dates()