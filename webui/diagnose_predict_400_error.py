#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断预测API 400错误问题
"""

import requests
import json
import sys

def test_predict_api_with_detailed_error():
    """测试预测API并获取详细错误信息"""
    print("\n=== 诊断预测API 400错误 ===")
    
    # 模拟前端发送的预测请求参数
    prediction_params = {
        "file_path": "stock_600519_live",
        "lookback": 400,
        "pred_len": 120,
        "temperature": 1.3,
        "top_p": 1,
        "sample_count": 1,
        "trading_mode": "calendar",
        "start_date": ""
    }
    
    url = "http://localhost:7070/api/predict"
    
    print(f"请求URL: {url}")
    print(f"请求参数: {json.dumps(prediction_params, indent=2, ensure_ascii=False)}")
    
    try:
        response = requests.post(url, json=prediction_params, timeout=60)
        print(f"\n状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
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
                    if "文件路径不能为空" in error_data.get('error', ''):
                        print("问题: file_path参数为空")
                    elif "缺少必需的列" in error_data.get('error', ''):
                        print("问题: 数据文件缺少必需的列")
                    elif "数据长度不足" in error_data.get('error', ''):
                        print("问题: 数据量不足")
                    elif "无效的开始日期格式" in error_data.get('error', ''):
                        print("问题: start_date参数格式错误")
                    elif "无法找到日期列" in error_data.get('error', ''):
                        print("问题: 数据文件缺少日期列")
                    else:
                        print("问题: 其他参数验证错误")
                elif response.status_code == 500:
                    print("错误类型: 服务器内部错误 (500)")
                    if "Kronos模型预测失败" in error_data.get('error', ''):
                        print("问题: 模型预测过程中出错")
                    elif "Kronos模型未加载" in error_data.get('error', ''):
                        print("问题: 模型未正确加载")
                    else:
                        print("问题: 服务器处理过程中出错")
                elif response.status_code == 503:
                    print("错误类型: 服务不可用 (503)")
                    if "Akshare数据提供者不可用" in error_data.get('error', ''):
                        print("问题: 数据提供者未初始化")
                    elif "网络连接失败" in error_data.get('error', ''):
                        print("问题: 网络连接问题")
                    else:
                        print("问题: 服务暂时不可用")
                        
            except:
                print(f"原始响应内容: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保Flask服务器正在运行")
        return False
    except requests.exceptions.Timeout:
        print("❌ 请求超时，预测可能需要更长时间")
        return False
    except Exception as e:
        print(f"❌ 请求过程中发生异常: {e}")
        return False

def test_model_status():
    """测试模型状态API"""
    print("\n=== 测试模型状态 ===")
    
    try:
        response = requests.get("http://localhost:7070/api/model-status", timeout=10)
        print(f"模型状态API状态码: {response.status_code}")
        
        if response.status_code == 200:
            status_data = response.json()
            print("✅ 模型状态API正常")
            print(f"模型状态: {status_data.get('status', 'N/A')}")
            print(f"可用模型: {status_data.get('available_models', [])}")
            return True
        else:
            print("❌ 模型状态API异常")
            try:
                error_data = response.json()
                print(f"错误信息: {error_data}")
            except:
                print(f"响应内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 模型状态API请求异常: {e}")
        return False

def test_data_provider():
    """测试数据提供者状态"""
    print("\n=== 测试数据提供者状态 ===")
    
    try:
        response = requests.get("http://localhost:7070/api/akshare/status", timeout=10)
        print(f"数据提供者状态API状态码: {response.status_code}")
        
        if response.status_code == 200:
            status_data = response.json()
            print("✅ 数据提供者状态API正常")
            print(f"数据提供者状态: {status_data.get('status', 'N/A')}")
            return True
        else:
            print("❌ 数据提供者状态API异常")
            try:
                error_data = response.json()
                print(f"错误信息: {error_data}")
            except:
                print(f"响应内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 数据提供者状态API请求异常: {e}")
        return False

def test_different_parameters():
    """测试不同的参数组合"""
    print("\n=== 测试不同参数组合 ===")
    
    test_cases = [
        {
            "name": "标准股票代码",
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
            "name": "简化参数",
            "params": {
                "file_path": "stock_600519_live",
                "lookback": 100,
                "pred_len": 30,
                "temperature": 1.0,
                "top_p": 0.9,
                "sample_count": 1
            }
        },
        {
            "name": "不同股票代码",
            "params": {
                "file_path": "stock_000001_live",
                "lookback": 400,
                "pred_len": 120,
                "temperature": 1.3,
                "top_p": 1,
                "sample_count": 1
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n测试: {test_case['name']}")
        print(f"参数: {json.dumps(test_case['params'], ensure_ascii=False)}")
        
        try:
            response = requests.post(
                "http://localhost:7070/api/predict",
                json=test_case['params'],
                timeout=30
            )
            
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                print("✅ 请求成功")
            else:
                print("❌ 请求失败")
                try:
                    error_data = response.json()
                    print(f"错误信息: {error_data.get('error', 'N/A')}")
                except:
                    print(f"响应内容: {response.text}")
                    
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