#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断400错误的具体原因
"""

import requests
import json
import sys

def test_predict_api_with_detailed_error():
    """测试预测API并获取详细错误信息"""
    print("=== 诊断预测API 400错误 ===")
    
    # 模拟前端发送的预测请求参数
    prediction_params = {
        "file_path": "stock_600519_live",
        "lookback": 400,
        "pred_len": 120,
        "temperature": 1.3,
        "top_p": 1,
        "sample_count": 1,
        "trading_mode": "calendar",
        "start_date": "2024-01-01"  # 使用修复后的YYYY-MM-DD格式
    }
    
    url = "http://localhost:8080/api/predict"
    
    print(f"请求URL: {url}")
    print(f"请求参数: {json.dumps(prediction_params, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(url, json=prediction_params, timeout=60)
        print(f"\n状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 预测API请求成功！")
            print(f"预测类型: {result.get('prediction_type', 'N/A')}")
            print(f"消息: {result.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ 预测API返回错误: {response.status_code}")
            
            # 尝试获取详细的错误信息
            try:
                error_data = response.json()
                print(f"错误信息: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
                
                # 分析错误类型
                if response.status_code == 400:
                    print("错误类型: 客户端错误 (400)")
                    if "模型未加载" in error_data.get('error', ''):
                        print("具体原因: Kronos模型未加载")
                    elif "数据长度不足" in error_data.get('error', ''):
                        print("具体原因: 数据量不足")
                    elif "无效的开始日期格式" in error_data.get('error', ''):
                        print("具体原因: 日期格式错误")
                    elif "文件路径不能为空" in error_data.get('error', ''):
                        print("具体原因: 文件路径为空")
                    else:
                        print("具体原因: 其他参数错误")
                        
            except:
                print(f"原始错误响应: {response.text}")
            
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保Web服务器正在运行")
        return False
    except requests.exceptions.Timeout:
        print("❌ 请求超时，服务器可能正在处理大量数据")
        return False
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def test_model_status():
    """测试模型状态"""
    print("\n=== 测试模型状态 ===")
    
    try:
        response = requests.get("http://localhost:8080/api/model-status", timeout=10)
        
        if response.status_code == 200:
            status = response.json()
            print("✅ 模型状态检查成功")
            print(f"模型状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ 模型状态检查失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 模型状态检查异常: {e}")
        return False

def test_data_provider():
    """测试数据提供者状态"""
    print("\n=== 测试数据提供者状态 ===")
    
    try:
        response = requests.post(
            "http://localhost:8080/api/akshare/get-stock-data",
            json={
                "symbol": "600519",
                "period": "daily",
                "start_date": "20200101",
                "end_date": "20241106"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ 数据提供者工作正常")
                data_info = data.get('data_info', {})
                print(f"数据量: {data_info.get('rows', 0)} 条记录")
                return True
            else:
                print(f"❌ 数据提供者返回失败: {data.get('error', 'N/A')}")
                return False
        else:
            print(f"❌ 数据提供者请求失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 数据提供者检查异常: {e}")
        return False

def test_different_parameters():
    """测试不同参数组合"""
    print("\n=== 测试不同参数组合 ===")
    
    test_cases = [
        {
            "name": "标准参数（无start_date）",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 400,
                "pred_len": 120,
                "temperature": 1.3,
                "top_p": 1,
                "sample_count": 1
            }
        },
        {
            "name": "较小参数（减少数据量要求）",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 100,
                "pred_len": 30,
                "temperature": 1.3,
                "top_p": 1,
                "sample_count": 1
            }
        },
        {
            "name": "最小参数（最小数据量）",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 50,
                "pred_len": 10,
                "temperature": 1.3,
                "top_p": 1,
                "sample_count": 1
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 测试: {test_case['name']}")
        print(f"参数: {json.dumps(test_case['params'], indent=2, ensure_ascii=False)}")
        
        try:
            response = requests.post("http://localhost:8080/api/predict", json=test_case['params'], timeout=30)
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ 请求成功")
            else:
                try:
                    error_data = response.json()
                    print(f"错误: {error_data.get('error', '未知错误')}")
                except:
                    print(f"错误响应: {response.text}")
                    
        except Exception as e:
            print(f"❌ 请求异常: {e}")

def main():
    """主诊断函数"""
    print("开始诊断预测API 400错误问题...")
    
    # 测试模型状态
    model_ok = test_model_status()
    
    # 测试数据提供者状态
    data_provider_ok = test_data_provider()
    
    # 测试预测API
    predict_ok = test_predict_api_with_detailed_error()
    
    # 测试不同参数组合
    test_different_parameters()
    
    # 总结诊断结果
    print("\n=== 诊断结果总结 ===")
    print(f"模型状态: {'✅ 正常' if model_ok else '❌ 异常'}")
    print(f"数据提供者状态: {'✅ 正常' if data_provider_ok else '❌ 异常'}")
    print(f"预测API: {'✅ 正常' if predict_ok else '❌ 异常'}")
    
    if not predict_ok:
        print("\n⚠️ 预测API存在问题，需要进一步调试")
        print("建议检查:")
        print("1. 模型是否正确加载")
        print("2. 数据提供者是否正常工作")
        print("3. 请求参数是否符合要求")
        print("4. 服务器日志中的详细错误信息")
    else:
        print("\n🎉 预测API工作正常！")

if __name__ == "__main__":
    main()