#!/usr/bin/env python3
"""
诊断400错误测试脚本
模拟前端请求并查看具体错误信息
"""

import requests
import json

def test_predict_api():
    """测试预测API接口"""
    url = "http://localhost:7070/api/predict"
    
    # 模拟前端发送的参数
    params = {
        "file_path": "stock_600519_live",  # 前端设置的格式
        "lookback": 400,
        "pred_len": 120,
        "temperature": 1.3,
        "top_p": 1,
        "sample_count": 2,
        "start_date": "2025-11-05",
        "trading_mode": "auto"
    }
    
    print("🚀 测试预测API接口")
    print(f"📊 请求参数: {json.dumps(params, indent=2)}")
    
    try:
        response = requests.post(url, json=params, timeout=30)
        
        print(f"📡 响应状态码: {response.status_code}")
        print(f"📡 响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ 请求成功!")
            print(f"📄 响应内容: {response.text}")
        else:
            print(f"❌ 请求失败! 状态码: {response.status_code}")
            print(f"📄 错误响应: {response.text}")
            
            # 尝试获取更多错误信息
            if response.status_code == 400:
                print("🔍 分析400错误原因:")
                
                # 检查响应内容
                try:
                    error_data = response.json()
                    print(f"📋 错误详情: {error_data}")
                except:
                    print(f"📋 原始错误信息: {response.text}")
                    
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {e}")
    except Exception as e:
        print(f"❌ 其他异常: {e}")

def test_alternative_params():
    """测试不同的参数组合"""
    print("\n🔧 测试不同参数组合")
    
    test_cases = [
        {
            "name": "基础参数",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 100,
                "pred_len": 30
            }
        },
        {
            "name": "带扩展名",
            "params": {
                "file_path": "stock_600519_live.csv",
                "lookback": 100,
                "pred_len": 30
            }
        },
        {
            "name": "绝对路径",
            "params": {
                "file_path": "c:\\kron\\data\\stock_600519_live.csv",
                "lookback": 100,
                "pred_len": 30
            }
        },
        {
            "name": "最小参数",
            "params": {
                "file_path": "stock_600519_live"
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 测试: {test_case['name']}")
        print(f"📊 参数: {test_case['params']}")
        
        try:
            response = requests.post(
                "http://localhost:7070/api/predict", 
                json=test_case['params'], 
                timeout=10
            )
            print(f"📡 状态码: {response.status_code}")
            if response.status_code != 200:
                print(f"📄 响应: {response.text[:200]}")
        except Exception as e:
            print(f"❌ 异常: {e}")

if __name__ == "__main__":
    print("🔍 开始诊断400错误...")
    test_predict_api()
    test_alternative_params()