#!/usr/bin/env python3
"""
测试修复后的temp_file_path字段功能
"""

import requests
import json
import sys
import os

def test_temp_file_path_api():
    """测试temp_file_path字段功能"""
    
    # API端点
    url = "http://localhost:7070/api/akshare/get-stock-data"
    
    # 测试数据
    test_data = {
        "symbol": "000001",
        "start_date": "2024-01-01",
        "end_date": "2024-01-10",
        "save_to_temp_file": True
    }
    
    try:
        # 发送POST请求
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            
            # 检查temp_file_path字段是否存在
            if 'temp_file_path' in result:
                print("✅ temp_file_path字段已正确返回")
                print(f"临时文件路径: {result['temp_file_path']}")
                
                # 检查文件是否存在
                if os.path.exists(result['temp_file_path']):
                    print("✅ 临时文件已成功创建")
                    
                    # 检查文件大小
                    file_size = os.path.getsize(result['temp_file_path'])
                    print(f"文件大小: {file_size} 字节")
                    
                    # 读取文件内容验证
                    with open(result['temp_file_path'], 'r', encoding='utf-8') as f:
                        content = f.read()
                        print(f"文件内容预览: {content[:200]}...")
                else:
                    print("❌ 临时文件不存在")
                    
            else:
                print("❌ temp_file_path字段未返回")
                
            # 检查其他字段
            if 'data' in result:
                print(f"数据记录数: {len(result['data'])}")
            
            if 'error' in result:
                print(f"错误信息: {result['error']}")
                
        else:
            print(f"❌ API请求失败，状态码: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求异常: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
    except Exception as e:
        print(f"❌ 其他异常: {e}")

def test_without_temp_file():
    """测试不保存临时文件的情况"""
    
    url = "http://localhost:7070/api/akshare/get-stock-data"
    
    test_data = {
        "symbol": "000001",
        "start_date": "2024-01-01",
        "end_date": "2024-01-10",
        "save_to_temp_file": False
    }
    
    try:
        response = requests.post(url, json=test_data, timeout=30)
        
        print("\n=== 测试不保存临时文件 ===")
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if 'temp_file_path' in result:
                print("❌ temp_file_path字段不应该返回")
                print(f"临时文件路径: {result['temp_file_path']}")
            else:
                print("✅ temp_file_path字段正确未返回")
                
        else:
            print(f"❌ API请求失败，状态码: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")

if __name__ == "__main__":
    print("=== 开始测试temp_file_path字段功能 ===")
    
    # 测试保存临时文件
    test_temp_file_path_api()
    
    # 测试不保存临时文件
    test_without_temp_file()
    
    print("\n=== 测试完成 ===")