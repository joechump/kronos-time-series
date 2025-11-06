#!/usr/bin/env python3
"""
测试修复后的预测API功能
"""

import requests
import json
import sys

def test_predict_api():
    """测试预测API功能"""
    print("=== 测试修复后的预测API功能 ===")
    
    # API端点
    url = "http://localhost:8080/api/predict"
    
    # 测试数据
    test_data = {
        "file_path": "stock_000001_live",  # 测试平安银行股票
        "lookback": 400,
        "pred_len": 120,
        "temperature": 1.0,
        "top_p": 0.9,
        "sample_count": 1,
        "trading_mode": "calendar"
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
            return True
        else:
            error_data = response.json()
            print(f"❌ 预测API请求失败: {response.status_code}")
            print(f"错误信息: {error_data.get('error', '未知错误')}")
            
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
    
    # 先测试模型状态
    model_status_ok = test_model_status_api()
    
    # 测试预测API
    predict_ok = test_predict_api()
    
    # 总结测试结果
    print("\n=== 测试结果总结 ===")
    if model_status_ok and predict_ok:
        print("✅ 所有测试通过！预测API功能已修复")
        return True
    else:
        print("❌ 部分测试失败，需要进一步调试")
        
        if not model_status_ok:
            print("问题: 模型状态API不可用")
        if not predict_ok:
            print("问题: 预测API功能异常")
        
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🎉 预测API修复完成！")
        sys.exit(0)
    else:
        print("\n⚠️ 预测API仍需进一步调试")
        sys.exit(1)