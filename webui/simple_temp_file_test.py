#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单临时文件功能测试脚本
"""

import requests
import json
import time

def test_basic_functionality():
    """测试基本功能"""
    
    base_url = "http://localhost:7070"
    
    print("=== 简单临时文件功能测试 ===")
    
    # 1. 测试系统状态
    print("1. 测试系统状态...")
    try:
        response = requests.get(f"{base_url}/api/system-info", timeout=10)
        if response.status_code == 200:
            print("   ✓ 系统状态正常")
        else:
            print(f"   ✗ 系统状态异常: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ 系统连接失败: {e}")
        return False
    
    # 2. 测试预测功能
    print("\n2. 测试预测功能...")
    try:
        prediction_data = {
            "file_path": "stock_000001_live",
            "lookback": 10,
            "prediction_days": 3,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        response = requests.post(f"{base_url}/api/predict", json=prediction_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("   ✓ 预测功能正常")
            print(f"   预测状态: {result.get('status', 'unknown')}")
            
            # 检查日志中是否有临时文件相关信息
            if 'logs' in result:
                for log in result['logs']:
                    if '临时文件' in log or 'temp' in log.lower():
                        print(f"   临时文件日志: {log}")
            
            return True
        else:
            print(f"   ✗ 预测失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ✗ 预测异常: {e}")
        return False

def test_cleanup_api():
    """测试清理API"""
    
    base_url = "http://localhost:7070"
    
    print("\n3. 测试临时文件清理API...")
    try:
        cleanup_data = {
            "hours_ago": 0.1  # 清理6分钟前的文件
        }
        
        response = requests.post(f"{base_url}/api/cleanup-temp-files", json=cleanup_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("   ✓ 清理API正常")
            print(f"   清理结果: {result}")
            return True
        else:
            print(f"   ✗ 清理API失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ✗ 清理API异常: {e}")
        return False

if __name__ == "__main__":
    print("等待服务器启动...")
    time.sleep(5)  # 等待服务器完全启动
    
    success1 = test_basic_functionality()
    success2 = test_cleanup_api()
    
    if success1 and success2:
        print("\n🎉 临时文件功能测试通过！")
    else:
        print("\n❌ 部分功能测试失败")
        
    print("\n测试完成")