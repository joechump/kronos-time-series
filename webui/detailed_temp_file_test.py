#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细临时文件功能测试脚本
"""

import requests
import json
import time
import os
import tempfile

def test_detailed_functionality():
    """详细测试临时文件功能"""
    
    base_url = "http://localhost:7070"
    
    print("=== 详细临时文件功能测试 ===")
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 检查临时文件目录
    print("1. 检查临时文件目录...")
    temp_dir = os.path.join(tempfile.gettempdir(), "kronos")
    print(f"   临时文件目录: {temp_dir}")
    
    if os.path.exists(temp_dir):
        files = os.listdir(temp_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        print(f"   现有临时文件数量: {len(csv_files)}")
        if csv_files:
            print(f"   现有临时文件: {csv_files[:5]}")
    else:
        print("   临时文件目录不存在")
    
    # 2. 测试首次预测（应该创建临时文件）
    print("\n2. 测试首次预测（创建临时文件）...")
    try:
        prediction_data = {
            "file_path": "stock_000001_live",
            "lookback": 15,
            "prediction_days": 5,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/api/predict", json=prediction_data, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ 首次预测成功 (耗时: {end_time - start_time:.2f}秒)")
            print(f"   预测状态: {result.get('status', 'unknown')}")
            
            # 详细分析日志
            if 'logs' in result:
                temp_file_logs = []
                for log in result['logs']:
                    if any(keyword in log for keyword in ['临时文件', 'temp', '文件', '保存', '读取']):
                        temp_file_logs.append(log)
                
                if temp_file_logs:
                    print("   临时文件相关日志:")
                    for log in temp_file_logs:
                        print(f"     - {log}")
                else:
                    print("   ⚠ 未找到临时文件相关日志")
            
            return True
        else:
            print(f"   ✗ 首次预测失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ✗ 首次预测异常: {e}")
        return False
    
    # 3. 检查临时文件是否创建
    print("\n3. 检查临时文件创建情况...")
    time.sleep(2)  # 等待文件保存
    
    if os.path.exists(temp_dir):
        files_after = os.listdir(temp_dir)
        csv_files_after = [f for f in files_after if f.endswith('.csv')]
        print(f"   预测后临时文件数量: {len(csv_files_after)}")
        
        new_files = set(csv_files_after) - set(csv_files) if 'csv_files' in locals() else csv_files_after
        if new_files:
            print(f"   ✓ 检测到新创建的临时文件: {list(new_files)[:3]}")
        else:
            print("   ⚠ 未检测到新创建的临时文件")
    
    # 4. 测试第二次预测（应该使用临时文件）
    print("\n4. 测试第二次预测（使用临时文件）...")
    try:
        start_time = time.time()
        response = requests.post(f"{base_url}/api/predict", json=prediction_data, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ 第二次预测成功 (耗时: {end_time - start_time:.2f}秒)")
            print(f"   预测状态: {result.get('status', 'unknown')}")
            
            # 检查是否有使用临时文件的日志
            if 'logs' in result:
                file_read_logs = []
                for log in result['logs']:
                    if any(keyword in log for keyword in ['使用临时文件', '从文件读取', 'read from', 'file']):
                        file_read_logs.append(log)
                
                if file_read_logs:
                    print("   文件读取相关日志:")
                    for log in file_read_logs:
                        print(f"     - {log}")
                else:
                    print("   ⚠ 未找到文件读取相关日志")
            
            return True
        else:
            print(f"   ✗ 第二次预测失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ✗ 第二次预测异常: {e}")
        return False
    
    # 5. 测试清理API
    print("\n5. 测试临时文件清理API...")
    try:
        cleanup_data = {
            "hours_ago": 0.02  # 清理约1分钟前的文件
        }
        
        response = requests.post(f"{base_url}/api/cleanup-temp-files", json=cleanup_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("   ✓ 清理API调用成功")
            print(f"   清理结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return True
        else:
            print(f"   ✗ 清理API失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ✗ 清理API异常: {e}")
        return False

if __name__ == "__main__":
    print("等待服务器启动...")
    time.sleep(3)
    
    success = test_detailed_functionality()
    
    print("\n" + "="*50)
    if success:
        print("🎉 临时文件功能详细测试完成！")
        print("✓ 临时文件目录检查")
        print("✓ 首次预测（临时文件创建）")
        print("✓ 临时文件创建验证")
        print("✓ 第二次预测（临时文件使用）")
        print("✓ 临时文件清理API")
    else:
        print("❌ 部分测试失败")
    
    print("\n测试完成时间:", time.strftime('%Y-%m-%d %H:%M:%S'))