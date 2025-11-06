#!/usr/bin/env python3
"""
测试前端发送的预测参数格式
"""

import json
import requests

def test_prediction_api():
    """测试预测API参数"""
    
    # 模拟前端发送的参数
    prediction_params = {
        "file_path": "stock_600159_live",
        "lookback": 400,
        "pred_len": 120,
        "start_date": None,
        "temperature": 1.3,
        "top_p": 0.98,
        "sample_count": 2
    }
    
    print("🚀 测试预测API参数:")
    print(json.dumps(prediction_params, indent=2, ensure_ascii=False))
    
    try:
        # 发送预测请求
        response = requests.post(
            "http://localhost:7070/api/predict",
            json=prediction_params,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\n📊 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 预测成功!")
            print(f"预测类型: {result.get('prediction_type', 'N/A')}")
            print(f"消息: {result.get('message', 'N/A')}")
            print(f"预测点数: {len(result.get('prediction_results', []))}")
            print(f"实际数据点数: {len(result.get('actual_data', []))}")
        else:
            error_data = response.json()
            print(f"❌ 预测失败: {error_data.get('error', '未知错误')}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接失败 - 请确保后端服务器正在运行在7070端口")
    except requests.exceptions.Timeout:
        print("❌ 请求超时 - 预测可能需要更长时间")
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")

def test_with_different_params():
    """测试不同的参数组合"""
    
    test_cases = [
        {
            "name": "实时股票数据",
            "params": {
                "file_path": "stock_600159_live",
                "lookback": 400,
                "pred_len": 120,
                "start_date": None,
                "temperature": 1.3,
                "top_p": 0.98,
                "sample_count": 2
            }
        },
        {
            "name": "文件数据",
            "params": {
                "file_path": "data/stock_600159.csv",
                "lookback": 400,
                "pred_len": 120,
                "start_date": None,
                "temperature": 1.3,
                "top_p": 0.98,
                "sample_count": 2
            }
        },
        {
            "name": "指定开始日期",
            "params": {
                "file_path": "stock_600159_live",
                "lookback": 400,
                "pred_len": 120,
                "start_date": "2024-01-01",
                "temperature": 1.3,
                "top_p": 0.98,
                "sample_count": 2
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"测试: {test_case['name']}")
        print(f"{'='*50}")
        
        try:
            response = requests.post(
                "http://localhost:7070/api/predict",
                json=test_case['params'],
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ 成功")
                print(f"预测点数: {len(result.get('prediction_results', []))}")
            else:
                error_data = response.json()
                print(f"❌ 失败: {error_data.get('error', '未知错误')}")
                
        except Exception as e:
            print(f"❌ 异常: {str(e)}")

if __name__ == "__main__":
    print("🧪 开始测试预测API参数")
    print("="*60)
    
    # 测试基本参数
    test_prediction_api()
    
    print("\n" + "="*60)
    print("🧪 测试不同参数组合")
    
    # 测试不同参数组合
    test_with_different_params()
    
    print("\n" + "="*60)
    print("🧪 测试完成")