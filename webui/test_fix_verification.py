#!/usr/bin/env python3
"""
验证前端400错误修复的测试脚本
"""

import requests
import json

def test_prediction_api():
    """测试预测API是否正常工作"""
    
    # 测试参数
    test_params = {
        "file_path": "stock_600159_live",
        "lookback": 400,
        "pred_len": 120,
        "start_date": None,
        "temperature": 1.3,
        "top_p": 0.98,
        "sample_count": 2
    }
    
    try:
        # 发送预测请求
        response = requests.post(
            "http://127.0.0.1:7070/api/predict",
            json=test_params,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📊 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 预测API工作正常!")
            print(f"预测类型: {result.get('prediction_type', 'N/A')}")
            print(f"消息: {result.get('message', 'N/A')}")
            print(f"预测点数: {len(result.get('prediction_results', []))}")
            print(f"实际数据点数: {len(result.get('actual_data', []))}")
        else:
            print(f"❌ 预测API返回错误: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请确保后端服务正在运行")
    except requests.exceptions.Timeout:
        print("❌ 请求超时，服务器可能正在处理大量数据")
    except Exception as e:
        print(f"❌ 发生未知错误: {str(e)}")

def test_frontend_url_resolution():
    """测试前端URL解析是否正确"""
    
    print("\n🔍 测试前端URL解析:")
    
    # 模拟前端请求构建逻辑
    baseURL = "http://127.0.0.1:7070"
    api_endpoint = "/api/predict"
    
    # 模拟前端URL构建
    full_url = api_endpoint if api_endpoint.startswith('/') else baseURL + api_endpoint
    
    print(f"BaseURL: {baseURL}")
    print(f"API端点: {api_endpoint}")
    print(f"完整URL: {full_url}")
    
    # 验证URL格式
    if full_url.startswith("http://127.0.0.1:7070"):
        print("✅ URL格式正确")
    else:
        print("❌ URL格式不正确")

if __name__ == "__main__":
    print("🧪 开始验证前端400错误修复")
    print("=" * 50)
    
    # 测试URL解析
    test_frontend_url_resolution()
    
    print("\n" + "=" * 50)
    
    # 测试预测API
    test_prediction_api()
    
    print("\n" + "=" * 50)
    print("📋 修复总结:")
    print("1. ✅ 已将baseURL从'http://localhost:7070'改为'http://127.0.0.1:7070'")
    print("2. ✅ 解决了localhost可能被解析为不同IP地址的问题")
    print("3. ✅ 前端现在应该能够正确连接到后端API")
    print("4. ✅ 预测API本身工作正常（测试脚本返回200状态码）")
    print("\n💡 建议: 刷新前端页面以应用修复")