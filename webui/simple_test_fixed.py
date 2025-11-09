#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试修复后的预测API逻辑
"""

import requests
import json

def test_simple_prediction():
    """简单测试预测API"""
    base_url = "http://localhost:8080"
    
    # 测试用例：使用实时股票代码
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
        }
    ]
    
    print("=== 简单测试修复后的预测API逻辑 ===")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test_case['description']}")
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
                print(f"   预测结果长度: {len(result.get('prediction', []))}")
                print(f"   时间戳数量: {len(result.get('timestamps', []))}")
                
                # 检查预测类型是否正确
                pred_type = result.get('prediction_type', '')
                if "往前" in pred_type or "历史数据" in pred_type:
                    print("✅ 预测类型正确 - 使用历史数据")
                else:
                    print("⚠️  预测类型可能不正确")
                    
            elif response.status_code == 400:
                error_info = response.json()
                error_msg = error_info.get('error', '')
                print(f"❌ 预测失败 (400错误)")
                print(f"   错误信息: {error_msg}")
                
                # 分析错误类型
                if "数据不足" in error_msg:
                    print("❌ 数据不足错误仍然存在")
                elif "文件路径" in error_msg:
                    print("❌ 文件路径错误")
                else:
                    print("❌ 其他错误")
                    
            else:
                print(f"❌ 请求失败，状态码: {response.status_code}")
                print(f"   响应内容: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到服务器，请确保Web服务正在运行")
        except Exception as e:
            print(f"❌ 请求异常: {str(e)}")

if __name__ == "__main__":
    test_simple_prediction()
    print("\n=== 测试完成 ===")