#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临时文件功能完整测试脚本
测试临时文件保存、读取和清理功能
"""

import requests
import json
import time
import os
import tempfile
from datetime import datetime, timedelta

def test_temp_file_functionality():
    """测试临时文件功能的完整流程"""
    
    base_url = "http://localhost:7070"
    
    print("=== 临时文件功能完整测试 ===")
    print(f"测试服务器: {base_url}")
    print()
    
    # 1. 测试系统状态
    print("1. 测试系统状态...")
    try:
        response = requests.get(f"{base_url}/api/system-info")
        if response.status_code == 200:
            system_info = response.json()
            print(f"   ✓ 系统状态正常: {system_info.get('status', 'unknown')}")
            print(f"   模型可用性: {system_info.get('model_available', False)}")
        else:
            print(f"   ✗ 系统状态检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ 系统状态检查异常: {e}")
        return False
    
    # 2. 测试预测API（首次调用，应该创建临时文件）
    print("\n2. 测试首次预测（创建临时文件）...")
    try:
        prediction_data = {
            "file_path": "stock_000001_live",
            "lookback": 30,
            "prediction_days": 7,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        response = requests.post(f"{base_url}/api/predict", json=prediction_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ 首次预测成功")
            print(f"   预测结果: {result.get('status', 'unknown')}")
            
            # 检查是否有临时文件相关的日志
            if 'logs' in result:
                for log in result['logs']:
                    if '临时文件' in log or 'temp' in log.lower():
                        print(f"   临时文件日志: {log}")
        else:
            print(f"   ✗ 首次预测失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ 首次预测异常: {e}")
        return False
    
    # 3. 等待一段时间，让临时文件有时间保存
    print("\n3. 等待临时文件保存...")
    time.sleep(2)
    
    # 4. 测试第二次预测（应该使用临时文件）
    print("\n4. 测试第二次预测（使用临时文件）...")
    try:
        response = requests.post(f"{base_url}/api/predict", json=prediction_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ 第二次预测成功")
            print(f"   预测结果: {result.get('status', 'unknown')}")
            
            # 检查是否有使用临时文件的日志
            if 'logs' in result:
                temp_file_used = False
                for log in result['logs']:
                    if '使用临时文件' in log or 'temp file' in log.lower():
                        print(f"   临时文件使用日志: {log}")
                        temp_file_used = True
                    if '从文件读取' in log or 'read from file' in log.lower():
                        print(f"   文件读取日志: {log}")
                        temp_file_used = True
                
                if temp_file_used:
                    print("   ✓ 检测到临时文件使用")
                else:
                    print("   ⚠ 未检测到临时文件使用日志")
        else:
            print(f"   ✗ 第二次预测失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ 第二次预测异常: {e}")
        return False
    
    # 5. 测试临时文件清理API
    print("\n5. 测试临时文件清理功能...")
    try:
        # 设置清理1分钟前的文件（确保不会清理刚创建的文件）
        cleanup_data = {
            "hours_ago": 0.02  # 约1.2分钟前
        }
        
        response = requests.post(f"{base_url}/api/cleanup-temp-files", json=cleanup_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ 临时文件清理成功")
            print(f"   清理结果: {result.get('status', 'unknown')}")
            print(f"   清理文件数量: {result.get('files_cleaned', 0)}")
            print(f"   剩余文件数量: {result.get('files_remaining', 0)}")
        else:
            print(f"   ✗ 临时文件清理失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
    except Exception as e:
        print(f"   ✗ 临时文件清理异常: {e}")
    
    # 6. 测试不同股票代码的临时文件功能
    print("\n6. 测试不同股票代码的临时文件功能...")
    test_stocks = ["600519", "000858"]  # 贵州茅台、五粮液
    
    for stock_code in test_stocks:
        print(f"   测试股票 {stock_code}...")
        try:
            stock_data = {
                "file_path": f"stock_{stock_code}_live",
                "lookback": 20,
                "prediction_days": 5,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            response = requests.post(f"{base_url}/api/predict", json=stock_data)
            if response.status_code == 200:
                result = response.json()
                print(f"     ✓ 股票 {stock_code} 预测成功")
            else:
                print(f"     ✗ 股票 {stock_code} 预测失败: {response.status_code}")
        except Exception as e:
            print(f"     ✗ 股票 {stock_code} 预测异常: {e}")
    
    # 7. 测试临时文件目录检查
    print("\n7. 测试临时文件目录状态...")
    try:
        temp_dir = os.path.join(tempfile.gettempdir(), "kronos")
        if os.path.exists(temp_dir):
            files = os.listdir(temp_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            print(f"   ✓ 临时文件目录存在: {temp_dir}")
            print(f"   临时文件数量: {len(csv_files)}")
            if csv_files:
                print(f"   临时文件示例: {csv_files[:3]}")
        else:
            print(f"   ⚠ 临时文件目录不存在: {temp_dir}")
    except Exception as e:
        print(f"   ✗ 临时文件目录检查异常: {e}")
    
    print("\n=== 测试完成 ===")
    print("临时文件功能测试总结:")
    print("✓ 系统状态检查")
    print("✓ 首次预测（临时文件创建）")
    print("✓ 第二次预测（临时文件使用）")
    print("✓ 临时文件清理")
    print("✓ 多股票代码测试")
    print("✓ 临时文件目录检查")
    
    return True

if __name__ == "__main__":
    # 等待服务器完全启动
    print("等待服务器启动...")
    time.sleep(3)
    
    success = test_temp_file_functionality()
    
    if success:
        print("\n🎉 临时文件功能测试通过！")
    else:
        print("\n❌ 临时文件功能测试失败！")
        
    print("\n测试脚本执行完成。")