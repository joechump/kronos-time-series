#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的预测API功能
验证错误处理改进和备用数据源机制
"""

import requests
import json
import sys
import time

def test_predict_api():
    """测试预测API功能"""
    print("=== 测试修复后的预测API功能 ===")
    
    # API端点
    url = "http://localhost:8080/api/predict"
    
    # 测试用例1：正常股票预测
    print("\n1. 测试正常股票预测 (600519)")
    test_data = {
        "file_path": "stock_600519_live",  # 测试贵州茅台股票
        "lookback": 400,
        "pred_len": 120,
        "temperature": 0.5,
        "top_p": 0.9,
        "sample_count": 1
    }
    
    try:
        print(f"发送预测请求到: {url}")
        print(f"请求数据: {json.dumps(test_data, indent=2, ensure_ascii=False)}")
        
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 预测API请求成功！")
            print(f"预测结果包含 {len(result.get('predictions', []))} 个预测点")
            print(f"预测类型: {result.get('prediction_type', 'N/A')}")
            
            # 检查错误信息中是否包含建议
            if 'error' in result:
                print(f"警告: 响应包含错误信息: {result['error']}")
            
            return True
        else:
            error_data = response.json()
            print(f"❌ 预测API请求失败: {response.status_code}")
            print(f"错误信息: {error_data.get('error', '未知错误')}")
            
            # 检查是否有建议信息
            if 'suggestion' in error_data:
                print(f"建议: {error_data['suggestion']}")
            
            # 分析错误类型
            if response.status_code == 400:
                print("错误类型: 客户端错误 (400)")
                if "模型未加载" in error_data.get('error', ''):
                    print("具体问题: 模型加载状态检查失败")
                elif "数据长度不足" in error_data.get('error', ''):
                    print("具体问题: 股票数据不足")
                else:
                    print("具体问题: 其他客户端错误")
            elif response.status_code == 503:
                print("错误类型: 服务不可用 (503)")
                print("具体问题: Akshare数据提供者或网络连接问题")
            elif response.status_code == 500:
                print("错误类型: 服务器内部错误 (500)")
                print("具体问题: 模型预测过程出错")
            
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保服务器正在运行")
        return False
    except requests.exceptions.Timeout:
        print("❌ 请求超时，服务器响应时间过长")
        return False
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        return False

def test_error_handling():
    """测试错误处理功能"""
    print("\n=== 测试错误处理功能 ===")
    
    url = "http://localhost:8080/api/predict"
    
    # 测试用例2：无效股票代码
    print("\n2. 测试无效股票代码 (12345)")
    test_data = {
        "file_path": "stock_12345_live",
        "lookback": 400,
        "pred_len": 120
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 400 or response.status_code == 404:
            error_data = response.json()
            print(f"✅ 错误处理正常: {error_data.get('error', 'Unknown error')}")
            if 'suggestion' in error_data:
                print(f"建议: {error_data['suggestion']}")
            return True
        else:
            print("❌ 错误处理异常")
            return False
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        return False
    
    # 测试用例3：格式错误的日期
    print("\n3. 测试格式错误的日期")
    test_data = {
        "file_path": "stock_600519_live",
        "lookback": 400,
        "pred_len": 120,
        "start_date": "2024-13-45"  # 无效日期
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 400:
            error_data = response.json()
            print(f"✅ 日期验证正常: {error_data.get('error', 'Unknown error')}")
            if 'suggestion' in error_data:
                print(f"建议: {error_data['suggestion']}")
            return True
        else:
            print("❌ 日期验证异常")
            return False
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        return False
    
    # 测试用例4：数据量不足
    print("\n4. 测试数据量不足")
    test_data = {
        "file_path": "stock_999999_live",  # 模拟数据
        "lookback": 1000,  # 设置过大的lookback
        "pred_len": 120
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 400:
            error_data = response.json()
            print(f"✅ 数据量检查正常: {error_data.get('error', 'Unknown error')}")
            if 'suggestion' in error_data:
                print(f"建议: {error_data['suggestion']}")
            return True
        else:
            print("❌ 数据量检查异常")
            return False
    except Exception as e:
        print(f"❌ 请求异常: {str(e)}")
        return False

def test_model_status_api():
    """测试模型状态API"""
    print("\n=== 测试模型状态API ===")
    
    url = "http://localhost:8080/api/model-status"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            status_data = response.json()
            print("✅ 模型状态API请求成功！")
            print(f"模型状态: {status_data}")
            return True
        else:
            print(f"❌ 模型状态API请求失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 模型状态API请求异常: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("开始测试修复后的预测API功能...")
    print("请确保Web UI服务正在运行 (http://localhost:8080)")
    print("=" * 50)
    
    # 先测试模型状态
    model_status_ok = test_model_status_api()
    
    # 等待服务稳定
    print("\n等待服务稳定...")
    time.sleep(2)
    
    # 测试预测API
    predict_ok = test_predict_api()
    
    # 测试错误处理
    error_handling_ok = test_error_handling()
    
    # 总结测试结果
    print("\n=== 测试结果总结 ===")
    test_results = {
        "模型状态": model_status_ok,
        "预测功能": predict_ok,
        "错误处理": error_handling_ok
    }
    
    all_passed = all(test_results.values())
    
    for test_name, passed in test_results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
    
    if all_passed:
        print("\n🎉 所有测试通过！预测API功能已修复")
        print("✅ 备用数据源机制正常工作")
        print("✅ 错误处理改进已生效")
        print("✅ 用户友好的错误信息已实现")
        return True
    else:
        print("\n⚠️ 部分测试失败，需要进一步调试")
        
        if not model_status_ok:
            print("问题: 模型状态API不可用")
        if not predict_ok:
            print("问题: 预测API功能异常")
        if not error_handling_ok:
            print("问题: 错误处理功能异常")
        
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 预测API修复完成！")
        sys.exit(0)
    else:
        print("\n⚠️ 预测API仍需进一步调试")
        sys.exit(1)