#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前端400错误详细诊断脚本
检查前端请求参数、数据文件状态和API响应
"""

import requests
import json
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_api_directly():
    """直接测试后端API"""
    print("=== 直接测试后端API ===")
    
    # 测试正确的参数格式
    url = "http://localhost:8080/api/predict"
    params = {
        "file_path": "stock_600519_live",
        "lookback": 100,
        "pred_len": 30,
        "model_name": "kronos-small",
        "stock_code": "600519"
    }
    
    try:
        response = requests.post(url, json=params)
        print(f"API响应状态码: {response.status_code}")
        print(f"API响应内容: {response.text[:500]}")
        
        if response.status_code == 200:
            print("✅ 后端API工作正常")
            return True
        else:
            print(f"❌ 后端API返回错误: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API请求异常: {e}")
        return False

def check_data_file_exists():
    """检查数据文件是否存在"""
    print("\n=== 检查数据文件状态 ===")
    
    # 检查数据文件路径
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloaded_data")
    expected_file = os.path.join(data_dir, "stock_600519_live.csv")
    
    print(f"数据目录: {data_dir}")
    print(f"期望文件: {expected_file}")
    
    if os.path.exists(data_dir):
        print("✅ 数据目录存在")
        files = os.listdir(data_dir)
        print(f"数据目录中的文件: {files}")
    else:
        print("❌ 数据目录不存在")
        return False
    
    if os.path.exists(expected_file):
        print("✅ 数据文件存在")
        file_size = os.path.getsize(expected_file)
        print(f"文件大小: {file_size} 字节")
        return True
    else:
        print("❌ 数据文件不存在")
        return False

def check_model_status():
    """检查模型状态"""
    print("\n=== 检查模型状态 ===")
    
    url = "http://localhost:8080/api/model-status"
    
    try:
        response = requests.get(url)
        print(f"模型状态响应: {response.status_code}")
        print(f"模型状态内容: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"模型状态: {data}")
            return True
        else:
            print("❌ 模型状态检查失败")
            return False
    except Exception as e:
        print(f"❌ 模型状态检查异常: {e}")
        return False

def simulate_frontend_request():
    """模拟前端请求参数"""
    print("\n=== 模拟前端请求参数 ===")
    
    # 模拟前端发送的请求参数
    frontend_params = {
        "file_path": "stock_600519_live",
        "lookback": 100,
        "pred_len": 30,
        "model_name": "kronos-small",
        "stock_code": "600519",
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    }
    
    print("前端请求参数:")
    for key, value in frontend_params.items():
        print(f"  {key}: {value}")
    
    # 测试这个参数组合
    url = "http://localhost:8080/api/predict"
    
    try:
        response = requests.post(url, json=frontend_params)
        print(f"模拟前端请求响应: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 模拟前端请求成功")
            result = response.json()
            print(f"预测结果点数: {len(result.get('predictions', []))}")
            return True
        else:
            print(f"❌ 模拟前端请求失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 模拟前端请求异常: {e}")
        return False

def check_server_logs():
    """检查服务器日志（如果可能）"""
    print("\n=== 检查服务器状态 ===")
    
    # 检查服务器是否在运行
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        print(f"服务器主页响应: {response.status_code}")
        return True
    except Exception as e:
        print(f"❌ 服务器可能未运行: {e}")
        return False

def main():
    """主诊断函数"""
    print("🔍 前端400错误详细诊断")
    print("=" * 50)
    
    results = {}
    
    # 1. 检查服务器状态
    results['server'] = check_server_logs()
    
    # 2. 检查模型状态
    results['model'] = check_model_status()
    
    # 3. 检查数据文件
    results['data_file'] = check_data_file_exists()
    
    # 4. 直接测试API
    results['api'] = test_api_directly()
    
    # 5. 模拟前端请求
    results['frontend_sim'] = simulate_frontend_request()
    
    # 总结诊断结果
    print("\n" + "=" * 50)
    print("📊 诊断结果汇总")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:15} {status}")
    
    # 分析问题
    print("\n🔧 问题分析:")
    if not results['server']:
        print("- 服务器可能未启动，请检查webui服务")
    elif not results['data_file']:
        print("- 数据文件不存在，请在前端重新加载股票数据")
    elif not results['model']:
        print("- 模型状态异常，请检查模型加载")
    elif results['api'] and not results['frontend_sim']:
        print("- 后端API正常但前端参数有问题，检查前端请求格式")
    elif all(results.values()):
        print("- 所有检查通过，前端问题可能是缓存或时机问题")
    else:
        print("- 需要进一步排查具体问题")
    
    print("\n💡 建议操作:")
    print("1. 在前端页面重新加载股票数据")
    print("2. 清除浏览器缓存")
    print("3. 检查模型加载状态")
    print("4. 确认预测按钮在数据加载完成后才启用")

if __name__ == "__main__":
    main()