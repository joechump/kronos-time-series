#!/usr/bin/env python3
"""
测试修复后的temp_file_path字段功能 - 模拟数据测试
"""

import requests
import json
import sys
import os
import tempfile
import pandas as pd
from datetime import datetime, timedelta

def test_temp_file_path_with_simulation():
    """使用模拟数据测试temp_file_path字段功能"""
    
    # API端点
    url = "http://localhost:7070/api/akshare/get-stock-data"
    
    # 测试数据 - 使用有效的股票代码
    test_data = {
        "symbol": "600519",  # 贵州茅台，通常有数据
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
                    try:
                        with open(result['temp_file_path'], 'r', encoding='utf-8') as f:
                            content = f.read()
                            print(f"文件内容预览: {content[:200]}...")
                    except Exception as e:
                        print(f"文件读取错误: {e}")
                        
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
        "symbol": "600519",  # 贵州茅台
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

def test_different_stock_codes():
    """测试不同的股票代码"""
    
    stock_codes = ["600519", "000001", "000858", "601318"]
    
    for symbol in stock_codes:
        print(f"\n=== 测试股票代码: {symbol} ===")
        
        url = "http://localhost:7070/api/akshare/get-stock-data"
        test_data = {
            "symbol": symbol,
            "start_date": "2024-01-01",
            "end_date": "2024-01-10",
            "save_to_temp_file": True
        }
        
        try:
            response = requests.post(url, json=test_data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'temp_file_path' in result:
                    print(f"✅ {symbol}: temp_file_path字段已返回")
                    if os.path.exists(result['temp_file_path']):
                        print(f"   临时文件存在: {result['temp_file_path']}")
                    else:
                        print(f"   临时文件不存在")
                else:
                    print(f"❌ {symbol}: temp_file_path字段未返回")
                    
                if 'error' in result:
                    print(f"   错误信息: {result['error']}")
                    
            else:
                print(f"❌ {symbol}: API请求失败，状态码: {response.status_code}")
                
        except Exception as e:
            print(f"❌ {symbol}: 测试异常: {e}")

def test_akshare_directly():
    """直接测试akshare获取数据"""
    
    print("\n=== 直接测试akshare ===")
    
    try:
        import akshare as ak
        
        # 测试不同的股票代码
        test_codes = ["600519", "000001", "000858", "601318"]
        
        for symbol in test_codes:
            print(f"\n测试股票代码: {symbol}")
            
            try:
                # 尝试获取数据
                stock_data = ak.stock_zh_a_hist(
                    symbol=symbol, 
                    period='daily', 
                    start_date='20240101', 
                    end_date='20240110',
                    adjust="hfq"
                )
                
                if stock_data is not None and not stock_data.empty:
                    print(f"✅ 直接akshare获取成功，数据量: {len(stock_data)}")
                    print(f"  列名: {list(stock_data.columns)}")
                else:
                    print(f"❌ 直接akshare获取失败或数据为空")
                    
            except Exception as e:
                print(f"❌ 直接akshare获取异常: {e}")
                
    except ImportError:
        print("❌ akshare模块未安装")
    except Exception as e:
        print(f"❌ 直接测试异常: {e}")

if __name__ == "__main__":
    print("=== 开始测试temp_file_path字段功能 ===")
    
    # 测试直接akshare
    test_akshare_directly()
    
    # 测试保存临时文件
    test_temp_file_path_with_simulation()
    
    # 测试不保存临时文件
    test_without_temp_file()
    
    # 测试不同的股票代码
    test_different_stock_codes()
    
    print("\n=== 测试完成 ===")