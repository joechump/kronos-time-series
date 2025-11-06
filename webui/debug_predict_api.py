#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试预测API 400错误问题
"""

import requests
import json
import time

def test_predict_api():
    """测试预测API"""
    url = "http://localhost:7070/api/predict"
    
    # 测试不同的参数组合
    test_cases = [
        {
            "name": "标准股票代码",
            "data": {
                "stock_code": "000001",
                "days": 30,
                "model_name": "kronos-small"
            }
        },
        {
            "name": "带特殊参数",
            "data": {
                "stock_code": "600523",
                "lookback": 400,
                "pred_len": 120,
                "temperature": 1.3,
                "top_p": 1,
                "model_name": "kronos-small"
            }
        },
        {
            "name": "简化参数",
            "data": {
                "stock_code": "600523",
                "days": 30,
                "model_name": "kronos-small"
            }
        }
    ]
    
    print("=== 预测API调试测试 ===")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"服务器地址: {url}")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"测试 {i}: {test_case['name']}")
        print(f"请求参数: {json.dumps(test_case['data'], ensure_ascii=False, indent=2)}")
        
        try:
            response = requests.post(url, json=test_case['data'], timeout=30)
            
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 请求成功")
                print(f"预测状态: {result.get('status', 'unknown')}")
                if 'prediction' in result:
                    print(f"预测结果: {result['prediction']}")
            else:
                print("❌ 请求失败")
                print(f"错误信息: {response.text}")
                
                # 尝试获取更详细的错误信息
                try:
                    error_data = response.json()
                    print(f"错误详情: {error_data}")
                except:
                    pass
                    
        except requests.exceptions.RequestException as e:
            print(f"❌ 请求异常: {e}")
        except Exception as e:
            print(f"❌ 其他异常: {e}")
        
        print("-" * 50)
        time.sleep(2)  # 避免请求过于频繁

def test_system_status():
    """测试系统状态"""
    print("\n=== 系统状态检查 ===")
    
    # 测试系统信息API
    try:
        response = requests.get("http://localhost:7070/api/system-info", timeout=10)
        if response.status_code == 200:
            info = response.json()
            print("✅ 系统信息API正常")
            print(f"模型可用性: {info.get('model_available', False)}")
            print(f"临时文件目录: {info.get('temp_file_dir', 'N/A')}")
        else:
            print("❌ 系统信息API异常")
            print(f"状态码: {response.status_code}")
    except Exception as e:
        print(f"❌ 系统状态检查失败: {e}")

def test_model_loading():
    """测试模型加载"""
    print("\n=== 模型加载测试 ===")
    
    # 测试模型加载API
    try:
        response = requests.post(
            "http://localhost:7070/api/load-model",
            json={"model_name": "kronos-small"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 模型加载成功")
            print(f"加载结果: {result}")
        else:
            print("❌ 模型加载失败")
            print(f"状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")

if __name__ == "__main__":
    print("等待服务器启动...")
    time.sleep(5)  # 给服务器启动时间
    
    # 测试系统状态
    test_system_status()
    
    # 测试模型加载
    test_model_loading()
    
    # 测试预测API
    test_predict_api()
    
    print("\n=== 调试测试完成 ===")