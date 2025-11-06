#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前端请求调试脚本 - 模拟前端发送的预测请求，找出400错误原因
"""

import requests
import json
import sys

def debug_frontend_request():
    """调试前端发送的预测请求"""
    
    print("=== 前端请求调试测试 ===")
    
    # 模拟前端发送的预测请求参数
    prediction_params = {
        "file_path": "stock_600159_live",
        "lookback": 400,
        "pred_len": 120,
        "start_date": None,
        "temperature": 1.3,
        "top_p": 0.98,
        "sample_count": 2
    }
    
    print("1. 模拟前端请求参数:")
    print(json.dumps(prediction_params, indent=2, ensure_ascii=False))
    
    # 测试不同的API端点
    endpoints = [
        "http://127.0.0.1:7070/api/predict",
        "http://localhost:7070/api/predict"
    ]
    
    for endpoint in endpoints:
        print(f"\n2. 测试端点: {endpoint}")
        
        try:
            # 发送预测请求
            response = requests.post(
                endpoint,
                json=prediction_params,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            print(f"   响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("   ✅ 请求成功!")
                print(f"   预测类型: {result.get('prediction_type', 'N/A')}")
                print(f"   预测点数: {len(result.get('prediction_results', []))}")
                return True
            else:
                print(f"   ❌ 请求失败，状态码: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   错误信息: {json.dumps(error_data, indent=2, ensure_ascii=False)}")
                    
                    # 分析错误类型
                    error_msg = error_data.get('error', '')
                    if "文件路径不能为空" in error_msg:
                        print("   🔍 问题: file_path参数为空")
                    elif "数据长度不足" in error_msg:
                        print("   🔍 问题: 数据量不足")
                    elif "模型未加载" in error_msg:
                        print("   🔍 问题: 模型未正确加载")
                    elif "Akshare数据提供者不可用" in error_msg:
                        print("   🔍 问题: 数据提供者不可用")
                    else:
                        print("   🔍 问题: 其他错误")
                        
                except:
                    print(f"   原始响应内容: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            print("   ❌ 连接失败 - 服务器未运行或端口错误")
        except requests.exceptions.Timeout:
            print("   ❌ 请求超时")
        except Exception as e:
            print(f"   ❌ 请求异常: {e}")
    
    return False

def test_model_status():
    """测试模型状态"""
    
    print("\n3. 测试模型状态:")
    
    endpoints = [
        "http://127.0.0.1:7070/api/model-status",
        "http://localhost:7070/api/model-status"
    ]
    
    for endpoint in endpoints:
        print(f"   测试端点: {endpoint}")
        
        try:
            response = requests.get(endpoint, timeout=10)
            print(f"   响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("   ✅ 模型状态正常")
                print(f"   模型信息: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return True
            else:
                print(f"   ❌ 模型状态异常: {response.status_code}")
                print(f"   响应内容: {response.text}")
                
        except Exception as e:
            print(f"   ❌ 模型状态检查失败: {e}")
    
    return False

def main():
    """主调试函数"""
    
    print("开始调试前端400错误...")
    
    # 先测试模型状态
    model_ok = test_model_status()
    
    # 测试前端请求
    request_ok = debug_frontend_request()
    
    # 总结调试结果
    print("\n=== 调试结果总结 ===")
    
    if model_ok and request_ok:
        print("✅ 所有测试通过! 前端请求应该正常工作")
        print("🔍 可能的问题: 前端代码中的参数构建逻辑有误")
    elif model_ok and not request_ok:
        print("❌ 模型状态正常，但预测请求失败")
        print("🔍 可能的问题: 预测API参数验证失败或数据处理错误")
    elif not model_ok and request_ok:
        print("❌ 模型状态异常，但预测请求成功")
        print("🔍 可能的问题: 模型状态API有问题")
    else:
        print("❌ 模型状态和预测请求都失败")
        print("🔍 可能的问题: 服务器未正确启动或端口被占用")
    
    return model_ok and request_ok

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 调试完成!")
        sys.exit(0)
    else:
        print("\n⚠️ 需要进一步调试")
        sys.exit(1)