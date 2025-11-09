#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的预测API逻辑
验证start_date处理逻辑是否正确：从指定日期往前取历史数据
"""

import requests
import json
from datetime import datetime, timedelta

def test_fixed_prediction_logic():
    """测试修复后的预测API逻辑"""
    base_url = "http://localhost:8080"
    
    # 测试用例：使用2025-03-31作为开始日期
    test_cases = [
        {
            "stock_code": "000858",
            "start_date": "2025-03-31",
            "pred_len": 30,
            "description": "测试2025-03-31开始日期（之前会报数据不足错误）"
        },
        {
            "stock_code": "000858",
            "start_date": "2025-03-28",
            "pred_len": 30,
            "description": "测试2025-03-28开始日期（之前能正常工作）"
        },
        {
            "stock_code": "000858",
            "start_date": "2025-04-01",
            "pred_len": 30,
            "description": "测试2025-04-01开始日期（之前会报数据不足错误）"
        }
    ]
    
    print("=== 测试修复后的预测API逻辑 ===")
    print("验证start_date处理逻辑：从指定日期往前取历史数据\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"测试用例 {i}: {test_case['description']}")
        print(f"股票代码: {test_case['stock_code']}")
        print(f"开始日期: {test_case['start_date']}")
        print(f"预测长度: {test_case['pred_len']}")
        
        # 构建请求数据
        data = {
            "stock_code": test_case["stock_code"],
            "start_date": test_case["start_date"],
            "pred_len": test_case["pred_len"]
        }
        
        try:
            # 发送预测请求
            response = requests.post(f"{base_url}/api/predict", json=data)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 预测成功!")
                print(f"   预测类型: {result.get('prediction_type', 'N/A')}")
                print(f"   预测结果: {result.get('prediction', 'N/A')}")
                print(f"   时间戳数量: {len(result.get('timestamps', []))}")
            elif response.status_code == 400:
                error_info = response.json()
                print(f"❌ 预测失败 (400错误)")
                print(f"   错误信息: {error_info.get('error', 'N/A')}")
                print(f"   建议: {error_info.get('suggestion', 'N/A')}")
            else:
                print(f"❌ 请求失败，状态码: {response.status_code}")
                print(f"   响应内容: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到服务器，请确保Web服务正在运行")
        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")
        
        print("-" * 60)

def test_data_availability():
    """测试数据可用性"""
    print("\n=== 测试数据可用性 ===")
    
    base_url = "http://localhost:8080"
    
    # 测试不同日期的数据可用性
    test_dates = [
        "2025-03-27",  # 应该有足够数据
        "2025-03-28",  # 应该有足够数据  
        "2025-03-29",  # 之前会报数据不足
        "2025-03-31",  # 之前会报数据不足
        "2025-04-01",  # 之前会报数据不足
        "2025-04-02"   # 之前会报数据不足
    ]
    
    for date in test_dates:
        print(f"\n测试日期: {date}")
        
        data = {
            "stock_code": "000858",
            "start_date": date,
            "pred_len": 30
        }
        
        try:
            response = requests.post(f"{base_url}/api/predict", json=data)
            
            if response.status_code == 200:
                print("   ✅ 预测成功 - 数据充足")
            elif response.status_code == 400:
                error_info = response.json()
                error_msg = error_info.get('error', '')
                if "数据不足" in error_msg:
                    print(f"   ❌ 预测失败 - 数据不足")
                    print(f"      错误: {error_msg}")
                else:
                    print(f"   ❌ 预测失败 - 其他错误: {error_msg}")
            else:
                print(f"   ❌ 请求失败，状态码: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ 请求异常: {str(e)}")

if __name__ == "__main__":
    # 测试修复后的逻辑
    test_fixed_prediction_logic()
    
    # 测试数据可用性
    test_data_availability()
    
    print("\n=== 测试完成 ===")