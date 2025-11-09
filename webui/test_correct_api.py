#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试正确的预测API参数格式
"""

import requests
import json

def test_correct_predict_api():
    """测试正确的预测API参数格式"""
    
    base_url = "http://localhost:8080"
    
    # 正确的测试数据 - 使用file_path而不是stock_code
    test_cases = [
        {
            "name": "使用2025-11-07开始日期",
            "data": {
                "file_path": "stock_600159_live",
                "lookback": 400,
                "pred_len": 120,
                "start_date": "2025-11-07"
            }
        },
        {
            "name": "不使用开始日期（最新数据）",
            "data": {
                "file_path": "stock_600159_live",
                "lookback": 400,
                "pred_len": 120
            }
        },
        {
            "name": "使用2024-01-01开始日期",
            "data": {
                "file_path": "stock_600159_live",
                "lookback": 400,
                "pred_len": 120,
                "start_date": "2024-01-01"
            }
        }
    ]
    
    print("=== 正确格式的预测API测试 ===")
    
    for test_case in test_cases:
        print(f"\n--- 测试: {test_case['name']} ---")
        print(f"请求数据: {json.dumps(test_case['data'], indent=2, ensure_ascii=False)}")
        
        try:
            response = requests.post(
                f"{base_url}/api/predict",
                json=test_case['data'],
                timeout=30
            )
            
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 请求成功")
                print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
            else:
                print("❌ 请求失败")
                try:
                    error_data = response.json()
                    print(f"错误信息: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
                except:
                    print(f"原始响应: {response.text}")
                    
        except requests.exceptions.Timeout:
            print("⏰ 请求超时")
        except requests.exceptions.ConnectionError:
            print("🔌 连接错误 - 请确保服务器正在运行")
        except Exception as e:
            print(f"💥 其他错误: {str(e)}")

if __name__ == "__main__":
    test_correct_predict_api()